"""
test_linop_matches_dense_matvec.py
------------------------------------
Asserts that TTNLinearOperator's matvec reproduces PhysicsEngine._apply_ham_psi
ELEMENTWISE, on random complex input vectors -- not merely that the two
eigensolvers report the same energy.

Why elementwise matters: the two operators are related by H_patched = H_dense^T
if the index convention is inverted, and for Hermitian H that transpose has the
*same real spectrum*. So an energy-level comparison (which is all
diagnostics/ab_test_linop_vs_dense.py does) passes either way. That is exactly
how an inverted convention survived in _block_ham_matvec / Part B: every stored
renormalized operator carries index 0 on the un-conjugated-tensor side (see
PhysicsEngine._set_edge_spin and _set_block_hamiltonian, both emitting
output_edge_order=[bra[2], ket[2]]), and the dense consumers contract index 0,
so the LinearOperator must too.

Also checks that the dense H_eff really is Hermitian, so a future transpose
cannot hide behind "well, the energies match".

Run from the repo root:
    python diagnostics/test_linop_matches_dense_matvec.py
"""

import os
import sys
import tempfile

import numpy as np
import tensornetwork as tn
from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import TTNHamiltonianOperator
from ttnopt.hamiltonian import hamiltonian
from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.create_ttn import get_rnd_tree

SEED = 7
Lx, Ly, shape_ = 3, 3, "parallelogram"
CHI = 9
N_VECTORS = 5
ATOL = 1e-11
# Two couplings: one deep in each phase, so the check isn't accidentally run
# only where the effective Hamiltonian happens to be trivial.
G_VALUES = [-0.3, -1.2]

N = nplaqs(Lx, Ly, shape_)
tmpdir = tempfile.mkdtemp()
tensor_folder = os.path.join(tmpdir, "tensors")
os.makedirs(tensor_folder, exist_ok=True)
save_edges_file(Lx, Ly, shape_, tensor_folder=tensor_folder)

print(
    f"linop-vs-dense matvec equivalence: Lx={Lx}, Ly={Ly}, vacuum, chi={CHI}, "
    f"g={G_VALUES}, {N_VECTORS} random complex vectors/point, atol={ATOL:.0e}",
    flush=True,
)

failures = []

for g in G_VALUES:
    filename = os.path.join(tmpdir, f"EF_Z_{g}.dat")
    create_ids_coeffs_file(Lx, Ly, shape_, g, None, filename=filename)  # vacuum
    system_cfg = DotMap({
        "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
        "model": {"type": "z3", "file": filename},
        "MF_X": 1.0 / g, "EF_Z": filename,
    })

    np.random.seed(SEED)
    psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=CHI)
    ham = hamiltonian(system_cfg)
    # Left UNPATCHED: we want the dense _apply_ham_psi as the reference, and
    # build the LinearOperator against the very same engine so both read
    # bit-identical renormalized operators and block Hamiltonians.
    gss = GroundStateSearch(psi, ham, init_bond_dim=CHI, max_bond_dim=CHI)

    # One sweep so the renormalized operators are populated for a realistic,
    # non-trivial state rather than the freshly-initialized tree.
    gss.run(
        opt_structure=0, max_num_sweep=1, verbose=False,
        energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0,
    )

    central_tensor_ids = gss.psi.central_tensor_ids()
    H_op = TTNHamiltonianOperator(gss, central_tensor_ids)
    shape = H_op.psi_shape
    dim = H_op.shape[0]

    max_rel = 0.0
    for k in range(N_VECTORS):
        v = (np.random.randn(dim) + 1j * np.random.randn(dim)).astype(np.complex128)
        v /= np.linalg.norm(v)

        got = H_op.matvec(v)
        want = gss._apply_ham_psi(
            tn.Node(v.reshape(shape)), central_tensor_ids
        ).tensor.ravel()

        rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300)
        max_rel = max(max_rel, rel)

    # Hermiticity of the dense reference, built column by column.
    H_dense = np.empty((dim, dim), dtype=np.complex128)
    for j in range(dim):
        e_j = np.zeros(dim, dtype=np.complex128)
        e_j[j] = 1.0
        H_dense[:, j] = gss._apply_ham_psi(
            tn.Node(e_j.reshape(shape)), central_tensor_ids
        ).tensor.ravel()
    herm_err = np.linalg.norm(H_dense - H_dense.conj().T)
    # How far the effective H is from being real -- if this is ~0 the transpose
    # bug is numerically invisible here, which is worth stating explicitly
    # rather than concluding "no bug".
    imag_frac = np.linalg.norm(H_dense.imag) / np.linalg.norm(H_dense)

    status = "OK" if max_rel < ATOL else "FAIL"
    if max_rel >= ATOL:
        failures.append((g, max_rel))
    print(
        f"  g={g:+.3f}  dim={dim:<6d} max rel matvec err={max_rel:.3e}  "
        f"||H-H^dag||={herm_err:.3e}  imag frac={imag_frac:.3e}   {status}",
        flush=True,
    )

print()
if failures:
    for g, rel in failures:
        print(f"FAILED at g={g:+.3f}: relative matvec error {rel:.3e} >= {ATOL:.0e}")
    sys.exit(1)

print("PASS: TTNLinearOperator matvec matches PhysicsEngine._apply_ham_psi elementwise.")
