"""
g_fixed_chi_eigensolver_scan.py
---------------------------------
Two questions in one script, both at FIXED g and chi (isolating them from
the chi-truncation question entirely):

  1. Does ref_energy_reldiff keep shrinking with more sweeps, or does it
     plateau at a stationary oscillation level (as the 5-sweep budget in
     g_chi_scan_diagnostic.py suggested for every g checked so far)?
  2. Does the per-two-site eigensolver's own configuration -- tolerance,
     Krylov subspace size (ncv), or iteration budget (maxiter) -- actually
     move ref_energy_reldiff, or is it already so far below the scale that
     matters that tuning it further does nothing?

g=-2.0, chi=100 chosen because it's the hardest point already fully
converged in g_chi_scan_diagnostic.py's run -- its checkpoint is reused
directly as the common starting point for every configuration tested here,
so all runs start from bit-identical tensors (not chained off each other)
and any difference in outcome is attributable to the eigensolver
configuration alone, not to warm-start history.

Run from the repo root, e.g.:
    OMP_NUM_THREADS=<n> MKL_NUM_THREADS=<n> OPENBLAS_NUM_THREADS=<n> python diagnostics/g_fixed_chi_eigensolver_scan.py
"""

import os
import tempfile
import time

import numpy as np
from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine
from ttnopt.hamiltonian import hamiltonian
from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs
from Z3_funcs.hdf5_manager import load_tensor, tensor_exists

Lx, Ly, shape_ = 5, 5, "parallelogram"
G = -2.0          # hardest g already converged in g_chi_scan_diagnostic.py's batch
CHI = 100         # fixed -- isolates this test from the chi-truncation question
N_SWEEPS = 20     # generous budget: is reldiff still shrinking by sweep 20, or flat by sweep 8?
PRECISION = 3

# Reseeded before every config's run (see the loop below) so degeneracy-
# breaking in decompose_two_tensors's operate_degeneracy=True path
# (TwoSiteUpdater.py's np.random.choice) can't make later configs diverge
# from earlier ones purely because they consumed a different amount of
# global RNG state -- every config sees the identical sequence of random
# draws, so any difference in outcome is attributable to the eigensolver
# config alone.
SEED = 42

# Named eigensolver configurations to compare. "default" matches
# ttn_eigensolver's own defaults (tol=1e-10, ncv=None i.e. scipy's own
# sizing ~20, maxiter=300). Each of the others changes exactly one knob
# from "default" so any effect on ref_energy_reldiff is attributable to
# that one change. Tolerance and ncv each get enough points to see an
# actual trend rather than a couple of spot checks; ncv is capped at 160
# to stay well within the mem=350gb PBS request at chi=100 (dim=chi**4,
# so ncv=160 needs ~(160+4)*1.6GB =~ 262GB for the Krylov workspace alone).
CONFIGS = [
    {"label": "default",      "tol": 1e-10, "ncv": None, "maxiter": 300},
    {"label": "tol_1e-6",     "tol": 1e-6,  "ncv": None, "maxiter": 300},
    {"label": "tol_1e-8",     "tol": 1e-8,  "ncv": None, "maxiter": 300},
    {"label": "tol_1e-12",    "tol": 1e-12, "ncv": None, "maxiter": 300},
    {"label": "tol_1e-14",    "tol": 1e-14, "ncv": None, "maxiter": 300},
    {"label": "ncv_40",       "tol": 1e-10, "ncv": 40,   "maxiter": 300},
    {"label": "ncv_80",       "tol": 1e-10, "ncv": 80,   "maxiter": 300},
    {"label": "ncv_120",      "tol": 1e-10, "ncv": 120,  "maxiter": 300},
    {"label": "ncv_160",      "tol": 1e-10, "ncv": 160,  "maxiter": 300},
    {"label": "more_maxiter", "tol": 1e-10, "ncv": None, "maxiter": 1000},
]

# Must match wherever g_chi_scan_diagnostic.py actually wrote its
# checkpoint -- same file, same (g, chi) keying scheme.
CHECKPOINT_FILE = "../5_Z3/logs/g_chi_scan_checkpoint.hdf5"

if not tensor_exists(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, G, PRECISION, CHI):
    raise RuntimeError(
        f"No checkpoint found for g={G}, chi={CHI} in {CHECKPOINT_FILE} -- run "
        f"g_chi_scan_diagnostic.py first, or point CHECKPOINT_FILE at wherever "
        f"that run's output actually landed."
    )

N = nplaqs(Lx, Ly, shape_)
tmpdir = tempfile.mkdtemp()
filename = os.path.join(tmpdir, "EF_Z.dat")
create_ids_coeffs_file(Lx, Ly, shape_, G, None, filename=filename)  # vacuum
mf = 1 / G
system_cfg = DotMap({
    "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
    "model": {"type": "z3", "file": filename},
    "MF_X": mf, "EF_Z": filename,
})

print(
    f"Fixed-chi, more-sweeps + eigensolver-config scan: g={G}, chi={CHI}, {N_SWEEPS} "
    f"sweeps, configs={[c['label'] for c in CONFIGS]}, each warm-started fresh from "
    f"the same g_chi_scan_diagnostic.py checkpoint at (g={G}, chi={CHI})",
    flush=True,
)

all_histories = {}

for cfg in CONFIGS:
    label = cfg["label"]
    # Reload the SAME starting checkpoint for every config -- not chained
    # off the previous config's result -- so any difference in outcome is
    # attributable to the eigensolver config alone.
    psi, _ = load_tensor(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, G, PRECISION, CHI)
    ham = hamiltonian(system_cfg)
    gss = GroundStateSearch(psi, ham, init_bond_dim=CHI, max_bond_dim=CHI)
    patch_physics_engine(gss)
    ref_edge = gss.psi.top_edge_id

    print(
        f"\n{'='*90}\n--- {label}: tol={cfg['tol']:.1e}, ncv={cfg['ncv']}, "
        f"maxiter={cfg['maxiter']} ---\n{'='*90}",
        flush=True,
    )
    np.random.seed(SEED)
    t0 = time.time()
    gss.run(
        opt_structure=0,
        max_num_sweep=N_SWEEPS,
        verbose=True,
        diagnostic_edges=[ref_edge],
        lanczos_tol=cfg["tol"],
        lanczos_ncv=cfg["ncv"],
        lanczos_maxiter=cfg["maxiter"],
        energy_convergence_threshold=0.0,
        entanglement_convergence_threshold=0.0,
    )
    elapsed = time.time() - t0
    print(f"{label} stage done in {elapsed:.0f}s total", flush=True)

    all_histories[label] = list(gss.convergence_history)

# ── side-by-side trajectories across configs ────────────────────────────────
# convergence_history entries start at sweep 3 (sweeps 1-2 have no diff yet).
def _side_by_side(title, field):
    print(f"\n{'='*115}\n--- {title}, side by side across configs ---\n{'='*115}")
    header = f"{'sweep':>5}" + "".join(f"{c['label']:>16}" for c in CONFIGS)
    print(header)
    max_len = max(len(h) for h in all_histories.values())
    for i in range(max_len):
        row = f"{i + 3:>5}"
        for cfg in CONFIGS:
            h = all_histories[cfg["label"]]
            row += f"{h[i][field]:>16.4e}" if i < len(h) else f"{'--':>16}"
        print(row)

_side_by_side("ref_energy_reldiff trajectory", "ref_energy_reldiff")
_side_by_side("ref_lanczos_resid trajectory", "ref_lanczos_residual")
_side_by_side("ref_trunc_err trajectory", "ref_truncation_error")

print("\nDONE")
