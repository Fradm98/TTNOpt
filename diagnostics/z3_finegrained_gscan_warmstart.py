"""
z3_finegrained_gscan_warmstart.py
------------------------------------
Same fine-resolution unpatched g-scan as z3_finegrained_gscan.py (g in
[-0.8,-0.5], step 0.01, chis=[9,18,27,50], sweeps=[3,3,20,20]), but each
point is warm-started from the PREVIOUS g's converged chi=50 state
instead of cold-started -- much cheaper when it works, since only a
handful of resettling sweeps should be needed for a 0.01 step.

Risk (per discussion): near the actual transition, a state converged on
one side can have LOWER overlap with the true ground state on the other
side than a plain random state would -- so a naive warm start could get
stuck/fail to converge right where it matters most, worse than cold.

Mitigation: track a rolling time budget from the last few successfully
completed points (median * TIMEOUT_MULTIPLIER, floored at TIMEOUT_FLOOR
seconds so early, still-noisy statistics don't trigger a false abort).
Each point's `gss` subprocess runs under that timeout; if it's exceeded,
the attempt is killed and immediately retried ONCE, cold-started
(init_tree=2, random state) with no timeout -- the trusted fallback.
Every point's outcome (warm-started / fell back to cold / which) is
logged and saved, since the *pattern* of fallbacks is itself informative
about where the transition actually sits.

Separate, isolated drive_path from both the coarse 15-point scan and the
cold-started fine scan, so none of these three runs can collide.
"""
import csv
import os
import subprocess
import time

import numpy as np
import yaml
from tqdm import tqdm

from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.hdf5_manager import tensor_exists

Lx, Ly, shape = 5, 5, "parallelogram"
precision = 3
N = nplaqs(Lx, Ly, shape)

g_values = np.round(np.arange(0.50, 0.80 + 1e-9, 0.01), precision)
chis = [9, 18, 27, 50]
sweeps = [3, 3, 20, 20]

TIMEOUT_MULTIPLIER = 3.0
TIMEOUT_FLOOR_SECONDS = 20 * 60  # don't time out below this even if early points were fast

drive_path = "/Users/fradm/Desktop/projects/5_Z3_finegrained_warmstart"
folder = f"{drive_path}/shape_{shape}/_Lx{Lx}_Ly{Ly}/runs_vacuum"
os.makedirs(folder, exist_ok=True)
save_edges_file(Lx, Ly, shape, tensor_folder=folder)
init_tensor_file = f"{drive_path}/tensors.hdf5"

# Confined-phase-first order: start deep (g=-0.80, least ambiguous) and walk
# toward the transition, matching the coarse scan's own rationale.
g_order = list(g_values[::-1])

print(f"model: z3, shape: {shape}, Lx:{Lx}, Ly:{Ly}, isolated output: {drive_path}")
print(f"parameter space: {len(g_order)} points, warm-started from previous g, "
      f"timeout={TIMEOUT_MULTIPLIER}x rolling median (floor {TIMEOUT_FLOOR_SECONDS}s), "
      f"cold-start fallback on timeout", flush=True)

resume_start_idx = 0
previous_g = None
for g_raw in g_order:
    g_check = float(-g_raw)
    if tensor_exists(init_tensor_file, shape, Lx, Ly, None, None, g_check, precision, chis[-1]):
        resume_start_idx += 1
        previous_g = g_check
    else:
        break
if resume_start_idx > 0:
    print(f"Resuming: first {resume_start_idx} g-value(s) in g_order already done "
          f"(last: g={previous_g:.{precision}f}).")

completed_times = []
outcomes = []


def build_input(g, run_folder, filename, use_warm_start, previous_g_val):
    numerics_dict = {
        "opt_structure": {"type": 0},
        "initial_bond_dimension": chis[0],
        "energy_convergence_threshold": 1e-5,
        "entanglement_convergence_threshold": 1e-10,
        "energy_degeneracy_threshold": 1e-13,
        "entanglement_degeneracy_threshold": 1e-8,
        "verbose_sweeps": True,
        "unpatched": True,
    }
    if use_warm_start:
        numerics_dict["init_tree"] = 3
        numerics_dict["init_tensor_file"] = init_tensor_file
        numerics_dict["init_g"] = previous_g_val
        numerics_dict["init_chi"] = chis[-1]
        numerics_dict["max_bond_dimensions"] = [chis[-1]]
        numerics_dict["max_num_sweeps"] = [sweeps[-1]]
        numerics_dict["lanczos_tol"] = [None]
        numerics_dict["lanczos_maxiter"] = [None]
    else:
        numerics_dict["init_tree"] = 2
        numerics_dict["max_bond_dimensions"] = chis
        numerics_dict["max_num_sweeps"] = sweeps
        numerics_dict["lanczos_tol"] = [None] * len(chis)
        numerics_dict["lanczos_maxiter"] = [None] * len(chis)

    return {
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


pbar = tqdm(g_order[resume_start_idx:], dynamic_ncols=True)
for g_raw in pbar:
    g = float(-g_raw)
    use_warm_start = previous_g is not None
    pbar.set_description(
        f"g={g:.{precision}f} (" + (f"warm from g={previous_g:.{precision}f}" if use_warm_start else "cold start") + ")"
    )

    run_folder = f"{folder}/g_{g:.{precision}f}"
    os.makedirs(run_folder, exist_ok=True)
    filename = os.path.join(run_folder, "EF_Z.dat")
    create_ids_coeffs_file(Lx, Ly, shape, g, None, filename=filename)

    if completed_times:
        timeout = max(TIMEOUT_MULTIPLIER * float(np.median(completed_times)), TIMEOUT_FLOOR_SECONDS)
    else:
        timeout = None  # no baseline yet -- let the very first point run to completion

    inputfile = os.path.join(run_folder, "input.yml")
    t0 = time.time()
    outcome = "cold"  # default for the not-warm-started case, or a successful warm start
    if use_warm_start:
        with open(inputfile, "w") as f:
            yaml.dump(build_input(g, run_folder, filename, True, previous_g), f, sort_keys=False)
        try:
            subprocess.run(["gss", inputfile], check=True, timeout=timeout)
            outcome = "warm"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"\n  [g={g:.{precision}f}] warm start exceeded timeout ({timeout:.0f}s, took >{elapsed:.0f}s) "
                  f"-- falling back to a cold (random) restart", flush=True)
            with open(inputfile, "w") as f:
                yaml.dump(build_input(g, run_folder, filename, False, previous_g), f, sort_keys=False)
            subprocess.run(["gss", inputfile], check=True)
            outcome = "cold_fallback"
    else:
        with open(inputfile, "w") as f:
            yaml.dump(build_input(g, run_folder, filename, False, previous_g), f, sort_keys=False)
        subprocess.run(["gss", inputfile], check=True)
        outcome = "cold"

    elapsed = time.time() - t0
    if outcome == "warm":
        completed_times.append(elapsed)
    outcomes.append({"g": g, "outcome": outcome, "elapsed_s": elapsed, "timeout_used": timeout})
    print(f"  [g={g:.{precision}f}] outcome={outcome}, elapsed={elapsed:.1f}s, "
          f"timeout_was={('n/a' if timeout is None else f'{timeout:.0f}s')}", flush=True)

    previous_g = g

csv_path = os.path.join(os.path.dirname(__file__), "z3_5x5_warmstart_results", "finegrained_gscan_warmstart_outcomes.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["g", "outcome", "elapsed_s", "timeout_used"])
    for r in outcomes:
        writer.writerow([r["g"], r["outcome"], r["elapsed_s"], r["timeout_used"]])
print(f"\nsaved outcomes: {csv_path}")
print("DONE")
