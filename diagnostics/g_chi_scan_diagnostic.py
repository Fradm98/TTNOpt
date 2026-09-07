"""
g_chi_scan_diagnostic.py
-------------------------
Repeats chi_scan_diagnostic.py's per-chi ref-edge diagnostics across a
sequence of g values, walking from deep-confined (fast, clean convergence
expected at every chi) toward the transition -- to see whether the
trunc_loss/lanczos_gain plateau found at g=-1.0 (ratio stuck near 1 from
chi=50 through chi=150, no improving trend) is specific to the near-
critical region, or already present further out.

Warm-starts across BOTH dimensions: within a given g, chi is warm-started
upward exactly as in chi_scan_diagnostic.py; across g, each g's chi_max
converged state seeds the next g's chi=chis[0] stage (as in sweep.py).

Resume support: checks tensors_hdf5 for the most advanced (g, chi) already
completed (this ladder is purely ascending -- no descend phase -- so the
last chi processed for a given g is simply chis[-1]) and continues from
there, so a killed/restarted job doesn't redo finished work.

Run from the repo root, e.g.:
    OMP_NUM_THREADS=<n> MKL_NUM_THREADS=<n> OPENBLAS_NUM_THREADS=<n> python diagnostics/g_chi_scan_diagnostic.py
"""

import os, tempfile, time

import numpy as np
from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine
from ttnopt.hamiltonian import hamiltonian
from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.create_ttn import get_rnd_tree
from Z3_funcs.hdf5_manager import save_tensor, load_tensor, tensor_exists

Lx, Ly, shape_ = 5, 5, "parallelogram"
g_values = [-5.0, -4.0, -3.0, -2.0]   # processed in this order, most-confined first
# chi=150 deliberately left out here to keep the total runtime across 4 g's
# manageable -- add it back (chis = [10, 50, 100, 150]) once you see where
# the trend is heading and want the full picture at a specific g.
chis = [10, 50, 100]
N_SWEEPS = 5
PRECISION = 3

CHECKPOINT_FILE = "../5_Z3/logs/g_chi_scan_checkpoint.hdf5"
os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

N = nplaqs(Lx, Ly, shape_)
tmpdir = tempfile.mkdtemp()
tensor_folder = os.path.join(tmpdir, "tensors")
os.makedirs(tensor_folder, exist_ok=True)
save_edges_file(Lx, Ly, shape_, tensor_folder=tensor_folder)

print(f"g-chi scan: Lx={Lx}, Ly={Ly}, vacuum, g_values={g_values}, chis={chis}, "
      f"{N_SWEEPS} sweeps each (warm-started across chi AND across g)", flush=True)

# Resume support: find the most advanced (g, chi) checkpoint already
# completed. chis[-1] (not chis[0] -- this ladder never descends) marks a
# fully-completed g.
resume_g_idx = 0
resume_chi_idx_within_g = 0
previous_g_for_warmstart = None
for gi, g in enumerate(g_values):
    if tensor_exists(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, chis[-1]):
        resume_g_idx = gi + 1
        previous_g_for_warmstart = g
        continue
    for ci, chi in enumerate(chis):
        if tensor_exists(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, chi):
            resume_chi_idx_within_g = ci + 1
        else:
            break
    break

if resume_g_idx > 0 or resume_chi_idx_within_g > 0:
    print(
        f"Resuming: g_values[0:{resume_g_idx}] fully completed"
        + (
            f", plus chis[0:{resume_chi_idx_within_g}] of g_values[{resume_g_idx}]"
            if resume_chi_idx_within_g > 0
            else ""
        )
        + ".",
        flush=True,
    )

ref_edge = None
all_results = {}  # all_results[g][chi] = {"local_update_diagnostics": ..., "convergence_history": ...}
first_chi_idx_this_run = resume_chi_idx_within_g  # only applies to the very first g processed below

for gi in range(resume_g_idx, len(g_values)):
    g = g_values[gi]
    filename = os.path.join(tmpdir, f"EF_Z_g{g:.{PRECISION}f}.dat")
    create_ids_coeffs_file(Lx, Ly, shape_, g, None, filename=filename)  # vacuum
    mf = 1 / g
    system_cfg = DotMap({
        "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
        "model": {"type": "z3", "file": filename},
        "MF_X": mf, "EF_Z": filename,
    })
    ham = hamiltonian(system_cfg)

    chi_start_idx = first_chi_idx_this_run if gi == resume_g_idx else 0
    first_chi_idx_this_run = 0

    if chi_start_idx > 0:
        psi, _ = load_tensor(
            CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, chis[chi_start_idx - 1]
        )
        print(f"g={g}: resuming its own chi ladder from chi={chis[chi_start_idx - 1]}", flush=True)
    elif previous_g_for_warmstart is not None:
        psi, _ = load_tensor(
            CHECKPOINT_FILE, shape_, Lx, Ly, None, None, previous_g_for_warmstart, PRECISION, chis[-1]
        )
        print(f"g={g}: warm-started from g={previous_g_for_warmstart}, chi={chis[-1]}", flush=True)
    else:
        psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=chis[0])
        print(f"g={g}: cold start (first g processed this run)", flush=True)

    start_chi = chis[chi_start_idx] if chi_start_idx > 0 else chis[0]
    gss = GroundStateSearch(psi, ham, init_bond_dim=start_chi, max_bond_dim=start_chi)
    patch_physics_engine(gss)
    if ref_edge is None:
        ref_edge = gss.psi.top_edge_id
        print(f"top_edge_id (reference edge) = {ref_edge}", flush=True)

    all_results.setdefault(g, {})

    for chi in chis[chi_start_idx:]:
        gss.max_bond_dim = chi
        gss.move_canonical_center(ref_edge)
        gss._prime_renormalized_operators()
        t0 = time.time()
        gss.run(
            opt_structure=0,
            max_num_sweep=N_SWEEPS,
            verbose=True,
            energy_convergence_threshold=0.0,
            entanglement_convergence_threshold=0.0,
        )
        elapsed = time.time() - t0
        print(f"\ng={g}, chi={chi} stage done in {elapsed:.0f}s total\n", flush=True)

        save_tensor(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, chi, gss.psi)
        print(f"checkpoint saved: g={g}, chi={chi} -> {CHECKPOINT_FILE}", flush=True)

        all_results[g][chi] = {"convergence_history": list(gss.convergence_history)}

        print(f"{'='*110}\n--- g={g}, chi={chi} reference-edge trace ---\n{'='*110}")
        print(f"{'sweep':>5} {'ref_energy':>18} {'ref_energy_reldiff':>19} {'ref_trunc_err':>15} "
              f"{'ref_lanczos_resid':>18} {'ref_retried':>11} {'sweep_retries':>13}")
        for rec in all_results[g][chi]["convergence_history"]:
            print(f"{rec['sweep']:>5} {rec['ref_energy']:>18.10f} {rec['ref_energy_reldiff']:>19.6e} "
                  f"{rec['ref_truncation_error']:>15.6e} {rec['ref_lanczos_residual']:>18.6e} "
                  f"{str(rec['ref_lanczos_retried']):>11} {rec['num_lanczos_retries']:>13}")
        print("", flush=True)

    previous_g_for_warmstart = g

# ── final cross-(g, chi) summary tables ─────────────────────────────────────
# Only covers g's actually processed in THIS invocation -- if this run was
# itself resumed from an earlier crash, collate across log files by hand for
# the full picture (the per-stage tables above are always printed in full,
# regardless of resume).
def _cross_table(title, field):
    print(f"\n{'='*110}\n--- {title} (final sweep of each stage) ---\n{'='*110}")
    header = f"{'g':>8}" + "".join(f"{('chi='+str(c)):>14}" for c in chis)
    print(header)
    for g in g_values:
        row = f"{g:>8}"
        if g not in all_results:
            row += "   (not run this invocation)"
            print(row)
            continue
        for chi in chis:
            hist = all_results.get(g, {}).get(chi, {}).get("convergence_history")
            if not hist:
                row += f"{'--':>14}"
                continue
            row += f"{hist[-1][field]:>14.4e}"
        print(row)

_cross_table("ref_energy_reldiff", "ref_energy_reldiff")
_cross_table("ref_trunc_err", "ref_truncation_error")
_cross_table("ref_lanczos_resid", "ref_lanczos_residual")

print("\nDONE")
