import subprocess
import yaml
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs, get_coord_charges
from Z3_funcs.lattice_plaquettes import label_links

## System ##
model_name = "z3"
Lx = 3
Ly = 5
shape = "hexagon"
# shape = "parallelogram"
bound_state = "baryon"
# bound_state = "meson"
# bound_state = None
chargesx, chargesy = None, None
# chargesx, chargesy = [2,6], [1,1]
chargesx, chargesy = [-1,-1,2], [1,4,1]
R = 1
if chargesx is not None:
    if bound_state == "meson":
        R = int(abs(chargesx[-1] - chargesx[0]))
    elif bound_state == "baryon":
        R = int(max(chargesy) - min(chargesy))
    
R = None if bound_state == None else R
N = nplaqs(Lx,Ly,shape)
link_plaquettes, bounds = label_links(Lx, Ly, shape)

## Numerics ##
g_values = [-2, -1.5, -1, -0.5]
g_values = np.linspace(0.5,1,6)
precision = 3
g_values = [10]

# chi ladder: ascend cheaply to chi_max, converge hard there, then descend
# through the same values for consistently-converged finite-chi scaling data
# (per advisor's suggestion: truncate down from the best available state
# rather than optimizing every chi independently from scratch).
chis = [9,20,40]
chis = [9,18,27]
SWEEP_ASCEND = 2      # cheap warm-up stages while building up to chi_max
SWEEP_CONVERGE = 4   # hard-converge stage at chi_max, threshold-governed
SWEEP_DESCEND = 4    # re-settle stages while truncating down from chi_max

# Loosen the per-two-site eigensolver's own tolerance (ttn_eigensolver's
# tol/maxiter, NOT the outer sweep-convergence thresholds above) only during
# the cheap ascend stages -- those exist just to build up a reasonable
# structure on the way to chi_max, so the local ground state at each step
# doesn't need to be solved to 1e-10. The converge-hard and descend stages
# keep the eigensolver's tight default (tol=1e-10, maxiter=300) since those
# are the stages whose results you actually rely on.
LANCZOS_TOL_ASCEND = 1e-6
LANCZOS_MAXITER_ASCEND = 150

# g-continuation: warm-start each g from the previous g's converged chi_max
# state instead of a fresh random tree every time. Seeded from deep in the
# confined phase (g=-1.0) and walked toward the transition, since a random
# start near criticality is the most likely to land on a metastable state.
# Set False to fall back to independent random-start runs at every g (e.g.
# for a clean comparison, or to check for hysteresis around the transition).
WARM_START_G = True

# Both convergence thresholds below are purely the outer-sweep stopping
# criteria (compares each edge's local two-tensor energy/entanglement
# between consecutive full sweeps) -- NOT the eigensolver's internal
# tolerance, which is a separate hardcoded value inside ttn_eigensolver()
# and is unaffected by these. 1e-11 was unreachable in practice (relative
# energy diff was barely reaching ~1e-1 near criticality with the old
# 2-sweep budget); 1e-5 on both puts them on the same scale to compare in
# each stage's convergence.csv.
energy_convergence_threshold = 1e-5
entanglement_convergence_threshold = 1e-10

device = "pc"

if device == "pc":
    drive_path = "D:work/projects/5_Z3"
elif device == "ngt":
    drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
elif device == "presto":
    drive_path = "/home/fradm/projects/5_Z3"


folder = f"{drive_path}/shape_{shape}/_Lx{Lx}_Ly{Ly}"
if bound_state == None:
    folder = f"{folder}/runs_vacuum"
else:
    if chargesx is None:
        chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state=bound_state, shape=shape, R=R)

    xs = "-".join(str(x) for x in chargesx)
    ys = "-".join(str(y) for y in chargesy)
    config = f"x{xs}_y{ys}"
    folder = f"{folder}/runs_{len(chargesx)}-q"
    folder = f"{folder}/{config}"

os.makedirs(folder, exist_ok=True)
save_edges_file(Lx, Ly, shape, tensor_folder=folder)

# Warm-start checkpoints are written here by gss itself (output.save_tensors)
init_tensor_file = f"{drive_path}/tensors.hdf5"


# ── Run gss ──────────────────────────────────────────────────────────────────
print(f"model: {model_name}, shape: {shape}, Lx:{Lx}, Ly:{Ly}")
print(f"N plaquette (dual lattice): {N}, L links (direct lattice): {len(link_plaquettes)}")
print(f"bound state: {bound_state}, R: {R}, parameter space:{len(g_values)}, g ext: {g_values[0]:.{precision}f}-{g_values[-1]:.{precision}f}")
print(f"bond dimensions: {chis} (ascend {SWEEP_ASCEND}/converge {SWEEP_CONVERGE}/descend {SWEEP_DESCEND} sweeps), device: {device}")
print(f"warm_start_g: {WARM_START_G}")

# Process g in confined-phase-first order (g=-1.0, largest |g|, first) so the
# warm-start chain is seeded from the easiest-to-converge, unambiguous region
# and walked toward the transition, rather than starting cold near
# criticality. g = -g_raw, and g_raw ascends in g_values, so g=-1.0 is at the
# end of g_values today -- reverse the iteration order to visit it first.
g_order = list(g_values[::-1])
descend_chis = chis[-2::-1]  # chis reversed, excluding chi_max (already the ascend/converge target)

pbar = tqdm(g_order, dynamic_ncols=True)
previous_g = None

for idx, g_raw in enumerate(pbar):
    g = float(-g_raw)
    use_warm_start = WARM_START_G and idx > 0

    pbar.set_description(
        f"g={g:.{precision}f} ("
        + (f"warm from g={previous_g:.{precision}f}" if use_warm_start else "cold start")
        + ")"
    )

    run_folder = f"{folder}/g_{g:.{precision}f}"
    os.makedirs(run_folder, exist_ok=True)
    filename = os.path.join(run_folder, "EF_Z.dat")

    create_ids_coeffs_file(Lx, Ly, shape, g, bound_state, chargesx=chargesx, chargesy=chargesy, filename=filename)

    mf = 1/g

    numerics_dict = {
        "opt_structure":
            {"type": 0},
        "initial_bond_dimension": chis[0],
        "energy_convergence_threshold": energy_convergence_threshold,
        "entanglement_convergence_threshold": entanglement_convergence_threshold,
        "energy_degeneracy_threshold": 1e-13,
        "entanglement_degeneracy_threshold": 1e-8,
        # Print one line per sweep (start + timing + energy/ee diff once
        # available) instead of going silent for the whole stage -- this is
        # what was missing, not tqdm; tqdm only tracks progress across g
        # values, it has no visibility into what a single gss subprocess is
        # doing sweep-by-sweep.
        "verbose_sweeps": True,
    }

    if use_warm_start:
        # No ascend leg here -- already starting from the previous g's
        # converged chi_max state, so every stage keeps the tight default.
        numerics_dict["init_tree"] = 3
        numerics_dict["init_tensor_file"] = init_tensor_file
        numerics_dict["init_g"] = previous_g
        numerics_dict["init_chi"] = chis[-1]
        numerics_dict["max_bond_dimensions"] = [chis[-1]] + descend_chis
        numerics_dict["max_num_sweeps"] = [SWEEP_CONVERGE] + [SWEEP_DESCEND] * len(descend_chis)
        numerics_dict["lanczos_tol"] = [None] * len(numerics_dict["max_bond_dimensions"])
        numerics_dict["lanczos_maxiter"] = [None] * len(numerics_dict["max_bond_dimensions"])
    else:
        numerics_dict["init_tree"] = 2
        numerics_dict["max_bond_dimensions"] = chis + descend_chis
        numerics_dict["max_num_sweeps"] = (
            [SWEEP_ASCEND] * (len(chis) - 1) + [SWEEP_CONVERGE] + [SWEEP_DESCEND] * len(descend_chis)
        )
        n_ascend = len(chis) - 1  # everything except the chis[-1] converge-hard stage
        n_rest = 1 + len(descend_chis)
        numerics_dict["lanczos_tol"] = [LANCZOS_TOL_ASCEND] * n_ascend + [None] * n_rest
        numerics_dict["lanczos_maxiter"] = [LANCZOS_MAXITER_ASCEND] * n_ascend + [None] * n_rest

    input_dict = {
    "system":
        {
        "N": N, # Number of spins
        "spin_size": '1',
        "Lx": Lx,
        "Ly": Ly,
        "shape": shape,
        "bound_state": bound_state,
        "chargesx": chargesx,
        "chargesy": chargesy,
        "R": R,

        "model":
            {
            "type": model_name,
            "file": filename,
            },

        "g": g,
        "MF_X": mf,
        "EF_Z": filename,
        "precision": precision,
        },

    "numerics": numerics_dict,

    "output":
        {
        "dir": run_folder,
        "tensors": folder,
        "save_tensors": drive_path,
        "single_site": 0,
        "two_site": 0,
        }
    }

    inputfile = os.path.join(run_folder, "input.yml")
    with open(inputfile, "w") as f:
        yaml.dump(input_dict, f, sort_keys=False)

    subprocess.run(["gss", inputfile], check=True)

    previous_g = g
