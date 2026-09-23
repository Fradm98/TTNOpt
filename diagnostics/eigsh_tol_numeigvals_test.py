"""
eigsh_tol_numeigvals_test.py
-------------------------------
Controlled test of WHY the patched (eigsh) engine lands on a slightly
higher-energy/lower-entropy state than the dense engine near the vacuum
phase transition (see compare_patched_vs_unpatched.py's discrepancy plot,
worst at g=-0.600/-0.700).

Loads the REFERENCE (dense-verified, correct) converged chi=27 checkpoint
for a given g directly from tensors.hdf5 -- no YAML/gss subprocess round
trip -- primes the renormalized block Hamiltonians/spin operators from
those already-converged tensors, then calls BOTH eigensolvers on the exact
same two-site effective Hamiltonian at the canonical center:

  1. PhysicsEngine.lanczos() (dense, unpatched) -- the reference answer.
  2. ttn_eigensolver(..., tol=T, num_eigvals=K) for a grid of (T, K) --
     does tightening tol to 1e-13 (matching lanczos_tol's exponent, though
     that's not really the comparable dense parameter -- see below) or
     requesting more Ritz values change which state eigsh returns?

Both start from the IDENTICAL v0 (built from the same loaded, converged
tensors) and see the IDENTICAL H_eff (same block Hamiltonians/spin
operators, primed once from that same state) -- so any difference in the
returned energy is attributable to the eigensolver algorithm itself, not
to differing initial guesses or differing renormalized-operator histories.

Usage:
  python diagnostics/eigsh_tol_numeigvals_test.py --g -0.700 --chi 27
  python diagnostics/eigsh_tol_numeigvals_test.py --g -0.600 --chi 27
"""
import argparse

from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine, ttn_eigensolver
from ttnopt.hamiltonian import hamiltonian as build_hamiltonian
from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs
from Z3_funcs.hdf5_manager import load_tensor, tensor_exists

Lx, Ly, shape_ = 5, 5, "parallelogram"
DRIVE_PATH = "/Users/fradm/Desktop/projects/5_Z3"  # the recovered, dense-verified reference
PRECISION = 3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g", type=float, required=True, help="actual g (negative), e.g. -0.700")
    p.add_argument("--chi", type=int, default=27)
    p.add_argument("--tols", type=float, nargs="+", default=[1e-10, 1e-13])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--max-iter", type=int, default=300)
    args = p.parse_args()

    g, chi = args.g, args.chi
    ten_file = f"{DRIVE_PATH}/tensors.hdf5"
    if not tensor_exists(ten_file, shape_, Lx, Ly, None, None, g, PRECISION, chi):
        raise SystemExit(f"No checkpoint for g={g}, chi={chi} in {ten_file}")

    N = nplaqs(Lx, Ly, shape_)
    filename = f"/tmp/EF_Z_test_g{g}_chi{chi}.dat"
    create_ids_coeffs_file(Lx, Ly, shape_, g, None, filename=filename)
    system_cfg = DotMap({
        "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
        "model": {"type": "z3", "file": filename},
        "MF_X": 1.0 / g, "EF_Z": filename,
    })
    ham = build_hamiltonian(system_cfg)

    print(f"Loading reference checkpoint: g={g}, chi={chi}, drive_path={DRIVE_PATH}", flush=True)
    psi, _ = load_tensor(ten_file, shape_, Lx, Ly, None, None, g, PRECISION, chi)
    gss = GroundStateSearch(psi, ham, init_bond_dim=chi, max_bond_dim=chi)
    # Rebuild block_hamiltonians / edge_spin_operators from the ALREADY
    # converged tensors -- no re-initialization, no re-sweeping, just the
    # renormalized-operator cache needed to assemble H_eff at the
    # canonical center exactly as GroundStateSearch.run() would have it
    # mid-sweep.
    gss._prime_renormalized_operators()
    central_ids = gss.psi.central_tensor_ids()

    # ---- 1. Dense reference (unpatched) on this EXACT two-site problem ----
    _, e_dense = gss.lanczos(central_ids)
    print(f"\nDense (unpatched) lanczos() on this exact H_eff: E = {e_dense:.12f}")
    print("(this is the value the two-site update would keep, since this "
          "checkpoint IS the converged fixed point)")

    # ---- 2. Patched eigsh over a (tol, num_eigvals) grid ----
    patch_physics_engine(gss)  # only rebinds gss.lanczos; H_eff/tensors untouched
    print(f"\n{'tol':>10} {'k':>3} {'E_patched':>18} {'E_patched - E_dense':>22}")
    for tol in args.tols:
        for k in args.ks:
            _, e_patched = ttn_eigensolver(
                gss, central_ids, num_eigvals=k, tol=tol, max_iter=args.max_iter
            )
            diff = e_patched - e_dense
            flag = "  <-- MATCHES dense" if abs(diff) < 1e-9 else ""
            print(f"{tol:10.0e} {k:3d} {e_patched:18.12f} {diff:+22.6e}{flag}")


if __name__ == "__main__":
    main()
