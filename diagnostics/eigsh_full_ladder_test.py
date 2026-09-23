"""
eigsh_full_ladder_test.py
----------------------------
Fair, full-ladder test of whether eigsh's `tol` or `num_eigvals` explain
why the patched engine lands on a lower-entropy/higher-energy state than
the dense engine near the vacuum phase transition (see
compare_patched_vs_unpatched.py). Unlike eigsh_tol_numeigvals_test.py
(which starts from the ALREADY-correct converged state, and found tol/k
irrelevant there), this branches from a REALISTIC intermediate state: the
same chi=18 ascend-stage checkpoint every (tol, num_eigvals) config shares,
produced by the ordinary ascend(9)->ascend(18) part of the real
pipeline.g_sweep ladder (patched, default settings, 3 sweeps each -- "as
always").

For each (tol, num_eigvals) config: deep-copy that SAME chi=18 state (so
no config is contaminated by another's chi=27 sweeps), set max_bond_dim=27,
override the two-site eigensolver to use this config, and run exactly 10
sweeps (energy/entanglement convergence thresholds forced to 0 so every
config runs the full fixed budget, not an early stop) -- matching the
coarse-scan recipe's chi=27 stage. Reports final entropy/energy per config,
flagging which (if any) reach the unpatched reference's ~0.213 entropy
rather than getting stuck near ~0.176-0.179.

Usage:
  python diagnostics/eigsh_full_ladder_test.py --g -0.700 --ascend-chi 18
"""
import argparse
import copy

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine, ttn_eigensolver
from ttnopt.hamiltonian import hamiltonian as build_hamiltonian
from dotmap import DotMap
from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs
from Z3_funcs.hdf5_manager import load_tensor, tensor_exists

Lx, Ly, shape_ = 5, 5, "parallelogram"
DEFAULT_ASCEND_DRIVE_PATH = "/Users/fradm/Desktop/projects/5_Z3_eigsh_test"
PRECISION = 3

# NOT a "ground truth" -- a fresh, independent dense (unpatched) rerun of
# the SAME g=-0.700 recipe landed on a DIFFERENT value (S=0.176112,
# E=-52.874274, matching what this script's patched configs found) than
# the original single unpatched run (S=0.213082, E=-52.878764). Two
# "identical" dense runs gave two different answers -- this point has (at
# least) two competing near-degenerate local optima, and either engine can
# land on either one depending on run-to-run numerical noise (BLAS
# threading non-determinism), not on which engine is used. Kept only as a
# printed data point for context, not as a target every config "should"
# reach. The variationally meaningful comparison is which trajectory
# reaches the LOWER energy, not which one matches a specific prior run.
PRIOR_RUN_ENTROPY = 0.213082
PRIOR_RUN_ENERGY = -52.878764


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g", type=float, required=True, help="actual g (negative), e.g. -0.700")
    p.add_argument("--ascend-chi", type=int, default=18, help="chi of the shared starting checkpoint")
    p.add_argument("--target-chi", type=int, default=27)
    p.add_argument("--sweeps", type=int, default=10)
    p.add_argument("--tols", type=float, nargs="+", default=[1e-10, 1e-13])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--ascend-drive-path", default=DEFAULT_ASCEND_DRIVE_PATH,
                    help="drive_path holding the shared ascend-only checkpoint")
    args = p.parse_args()

    g, ascend_chi, chi = args.g, args.ascend_chi, args.target_chi
    ASCEND_DRIVE_PATH = args.ascend_drive_path
    ten_file = f"{ASCEND_DRIVE_PATH}/tensors.hdf5"
    if not tensor_exists(ten_file, shape_, Lx, Ly, None, None, g, PRECISION, ascend_chi):
        raise SystemExit(
            f"No ascend checkpoint for g={g}, chi={ascend_chi} in {ten_file} -- "
            f"run pipeline.g_sweep --chis <...> {ascend_chi} --patched --drive-path {ASCEND_DRIVE_PATH} first."
        )

    N = nplaqs(Lx, Ly, shape_)
    filename = f"/tmp/EF_Z_ladder_test_g{g}.dat"
    create_ids_coeffs_file(Lx, Ly, shape_, g, None, filename=filename)
    system_cfg = DotMap({
        "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
        "model": {"type": "z3", "file": filename},
        "MF_X": 1.0 / g, "EF_Z": filename,
    })
    ham = build_hamiltonian(system_cfg)

    print(f"Loading shared starting point: g={g}, chi={ascend_chi} (ascend-only, "
          f"never touched by a later descend stage), drive_path={ASCEND_DRIVE_PATH}", flush=True)
    psi_template, _ = load_tensor(ten_file, shape_, Lx, Ly, None, None, g, PRECISION, ascend_chi)

    print(f"Branching to chi={chi}, {args.sweeps} sweeps (forced, no early stop). "
          f"For context only (NOT ground truth -- see module docstring), a prior "
          f"unpatched run landed at S={PRIOR_RUN_ENTROPY:.6f}, E={PRIOR_RUN_ENERGY:.6f}\n",
          flush=True)
    print(f"{'tol':>10} {'k':>3} {'final S':>10} {'final E':>14}")

    for tol in args.tols:
        for k in args.ks:
            print(f"--- running tol={tol:.0e} k={k} ---", flush=True)
            psi = copy.deepcopy(psi_template)
            # entanglement_degeneracy_threshold MUST match pipeline.g_sweep's
            # explicit override (1e-10) -- GroundStateSearch's class default
            # is 1e-8, which silently gives different SVD-truncation
            # degeneracy handling than the actual production runs use.
            gss = GroundStateSearch(
                psi, ham, init_bond_dim=ascend_chi, max_bond_dim=ascend_chi,
                energy_degeneracy_threshold=1e-13, entanglement_degeneracy_threshold=1e-10,
            )
            patch_physics_engine(gss)
            gss.max_bond_dim = chi
            # Full explicit control over BOTH tol and num_eigvals -- bypasses
            # run()'s own lanczos_tol/lanczos_maxiter pass-through (which has
            # no num_eigvals hook at all) so this one test doesn't require
            # any production-code changes.
            gss.lanczos = lambda central_tensor_ids, tol=tol, k=k: ttn_eigensolver(
                gss, central_tensor_ids, num_eigvals=k, tol=tol, max_iter=args.max_iter
            )
            gss.run(
                opt_structure=0,
                energy_convergence_threshold=0.0,
                entanglement_convergence_threshold=0.0,
                max_num_sweep=args.sweeps,
                verbose=True,
            )
            ref_edge = gss.psi.top_edge_id
            s_final = gss.entanglement[ref_edge]
            e_final = gss.energy[ref_edge]
            print(f"{tol:10.0e} {k:3d} {s_final:10.6f} {e_final:14.6f}")


if __name__ == "__main__":
    main()
