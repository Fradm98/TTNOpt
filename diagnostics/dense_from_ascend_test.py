"""
dense_from_ascend_test.py
----------------------------
Companion to eigsh_full_ladder_test.py: branches from the EXACT SAME
chi=18 ascend-stage checkpoint (produced by the ordinary patched
ascend(9)->ascend(18) part of pipeline.g_sweep, ONLY that checkpoint --
see eigsh_full_ladder_test.py's docstring), but runs the ORIGINAL DENSE
PhysicsEngine.lanczos() (no patch_physics_engine()) for the chi=27 stage
instead of eigsh.

Answers the direct question: starting from the identical, already
"stuck-prone" intermediate state, does the dense engine reach the
unpatched reference's higher entropy (~0.213) where every (tol,
num_eigvals) eigsh config got stuck near ~0.171-0.176? If yes, the gap is
attributable to the eigensolver algorithm itself (dense's exhaustive,
unbounded-Krylov-plus-inverse-iteration approach vs eigsh's
ncv-bounded restarted Lanczos), not to some property of the starting
point shared by every patched config.

Usage:
  python diagnostics/dense_from_ascend_test.py --g -0.700 --ascend-chi 18 --target-chi 27 --sweeps 10
"""
import argparse

from ttnopt.src import GroundStateSearch
from ttnopt.hamiltonian import hamiltonian as build_hamiltonian
from dotmap import DotMap
from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs
from Z3_funcs.hdf5_manager import load_tensor, tensor_exists

Lx, Ly, shape_ = 5, 5, "parallelogram"
DEFAULT_ASCEND_DRIVE_PATH = "/Users/fradm/Desktop/projects/5_Z3_eigsh_test"
PRECISION = 3

# NOT ground truth -- see eigsh_full_ladder_test.py's docstring/comment for
# why: a fresh independent dense rerun of this same g landed on a different
# value than this one prior run did. Kept only as context, never a target.
PRIOR_RUN_ENTROPY = 0.213082
PRIOR_RUN_ENERGY = -52.878764


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g", type=float, required=True)
    p.add_argument("--ascend-chi", type=int, default=18)
    p.add_argument("--target-chi", type=int, default=27)
    p.add_argument("--sweeps", type=int, default=10)
    p.add_argument("--ascend-drive-path", default=DEFAULT_ASCEND_DRIVE_PATH,
                    help="drive_path holding the shared ascend-only checkpoint")
    args = p.parse_args()

    g, ascend_chi, chi = args.g, args.ascend_chi, args.target_chi
    ASCEND_DRIVE_PATH = args.ascend_drive_path
    ten_file = f"{ASCEND_DRIVE_PATH}/tensors.hdf5"
    if not tensor_exists(ten_file, shape_, Lx, Ly, None, None, g, PRECISION, ascend_chi):
        raise SystemExit(f"No ascend checkpoint for g={g}, chi={ascend_chi} in {ten_file}")

    N = nplaqs(Lx, Ly, shape_)
    filename = f"/tmp/EF_Z_dense_test_g{g}.dat"
    create_ids_coeffs_file(Lx, Ly, shape_, g, None, filename=filename)
    system_cfg = DotMap({
        "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
        "model": {"type": "z3", "file": filename},
        "MF_X": 1.0 / g, "EF_Z": filename,
    })
    ham = build_hamiltonian(system_cfg)

    print(f"Loading SAME shared starting point as eigsh_full_ladder_test.py: "
          f"g={g}, chi={ascend_chi}, drive_path={ASCEND_DRIVE_PATH}", flush=True)
    psi, _ = load_tensor(ten_file, shape_, Lx, Ly, None, None, g, PRECISION, ascend_chi)
    # entanglement_degeneracy_threshold MUST match pipeline.g_sweep's explicit
    # override (1e-10) -- GroundStateSearch's own class default is 1e-8, and
    # using that silently gives every branch here different SVD-truncation
    # degeneracy handling than the actual production runs use.
    gss = GroundStateSearch(
        psi, ham, init_bond_dim=ascend_chi, max_bond_dim=ascend_chi,
        energy_degeneracy_threshold=1e-13, entanglement_degeneracy_threshold=1e-10,
    )
    # No patch_physics_engine() call -- gss.lanczos stays the original dense
    # PhysicsEngine.lanczos().
    gss.max_bond_dim = chi

    print(f"Branching to chi={chi}, {args.sweeps} sweeps, DENSE (unpatched) engine. "
          f"For context only (NOT ground truth), a prior unpatched run landed at "
          f"S={PRIOR_RUN_ENTROPY:.6f}, E={PRIOR_RUN_ENERGY:.6f}\n", flush=True)
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
    print(f"\nfinal S={s_final:.6f}  final E={e_final:.6f}")


if __name__ == "__main__":
    main()
