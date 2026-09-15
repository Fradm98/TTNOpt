"""
z3_finegrained_gscan.py
-------------------------
Fine-resolution unpatched g-scan focused on the suspected vacuum
transition region (g in [-0.8,-0.5], step 0.01, 31 points), using the
FIXED tree topology (Z3_funcs/create_graph_file.py). Each g is cold-
started independently (matching the coarse 15-point scan's convention),
chi ladder [9,18,27,50] with sweep budget [3,3,20,20] (no separate
descend stage -- chi=27 and chi=50 both get a generous budget instead,
since this whole region is the slow-converging one identified earlier).

Writes to an ISOLATED drive_path (not the shared production tensors.hdf5)
to avoid any resume-collision with the old (pre-fix, wrong-topology)
checkpoints already saved for g=-0.500/-0.600/-0.700/-0.800 at chi 9/18/27
-- deleting those was blocked by the destructive-action guard, and an
isolated directory sidesteps the problem entirely without touching
anything shared.
"""
import os
import subprocess

import numpy as np
import yaml
from tqdm import tqdm

from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.hdf5_manager import tensor_exists

Lx, Ly, shape = 5, 5, "parallelogram"
precision = 3
N = nplaqs(Lx, Ly, shape)

g_values = np.round(np.arange(0.50, 0.80 + 1e-9, 0.01), precision)  # g_raw; actual g = -g_raw
chis = [9, 18, 27, 50]
sweeps = [3, 3, 20, 20]

drive_path = "/Users/fradm/Desktop/projects/5_Z3_finegrained"
folder = f"{drive_path}/shape_{shape}/_Lx{Lx}_Ly{Ly}/runs_vacuum"
os.makedirs(folder, exist_ok=True)
save_edges_file(Lx, Ly, shape, tensor_folder=folder)
init_tensor_file = f"{drive_path}/tensors.hdf5"

print(f"model: z3, shape: {shape}, Lx:{Lx}, Ly:{Ly}, isolated output: {drive_path}")
print(f"parameter space: {len(g_values)} points, g ext: {g_values[0]:.{precision}f}-{g_values[-1]:.{precision}f}")
print(f"bond dimensions: {chis}, sweeps: {sweeps}, unpatched: True, cold start per g", flush=True)

# resume support (safe here since this drive_path is exclusively this scan's own data)
resume_start_idx = 0
for g_raw in g_values:
    if tensor_exists(init_tensor_file, shape, Lx, Ly, None, None, float(-g_raw), precision, chis[-1]):
        resume_start_idx += 1
    else:
        break
if resume_start_idx > 0:
    print(f"Resuming: first {resume_start_idx} g-value(s) already have chi={chis[-1]} checkpoints.")

pbar = tqdm(g_values[resume_start_idx:], dynamic_ncols=True)
for g_raw in pbar:
    g = float(-g_raw)
    pbar.set_description(f"g={g:.{precision}f} (cold start)")

    run_folder = f"{folder}/g_{g:.{precision}f}"
    os.makedirs(run_folder, exist_ok=True)
    filename = os.path.join(run_folder, "EF_Z.dat")
    create_ids_coeffs_file(Lx, Ly, shape, g, None, filename=filename)  # vacuum

    numerics_dict = {
        "opt_structure": {"type": 0},
        "initial_bond_dimension": chis[0],
        "energy_convergence_threshold": 1e-5,
        "entanglement_convergence_threshold": 1e-10,
        "energy_degeneracy_threshold": 1e-13,
        "entanglement_degeneracy_threshold": 1e-8,
        "verbose_sweeps": True,
        "unpatched": True,
        "init_tree": 2,  # cold start
        "max_bond_dimensions": chis,
        "max_num_sweeps": sweeps,
        "lanczos_tol": [None] * len(chis),
        "lanczos_maxiter": [None] * len(chis),
    }

    input_dict = {
        "system": {
            "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape,
            "bound_state": None, "chargesx": None, "chargesy": None, "R": None,
            "model": {"type": "z3", "file": filename},
            "g": g, "MF_X": 1.0 / g, "EF_Z": filename, "precision": precision,
        },
        "numerics": numerics_dict,
        "output": {
            "dir": run_folder, "tensors": folder, "save_tensors": drive_path,
            "single_site": 0, "two_site": 0,
        },
    }

    inputfile = os.path.join(run_folder, "input.yml")
    with open(inputfile, "w") as f:
        yaml.dump(input_dict, f, sort_keys=False)

    subprocess.run(["gss", inputfile], check=True)

print("DONE")
