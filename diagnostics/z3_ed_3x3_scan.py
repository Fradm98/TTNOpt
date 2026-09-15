"""
z3_ed_3x3_scan.py
------------------
Exact diagonalization of the Z3 gauge model on a 3x3 parallelogram
(N = nplaqs(3,3,'parallelogram') = 8 dual-lattice spins, local dim 3,
Hilbert space dim 3^8 = 6561), vacuum sector (bound_state=None, no charges).

Mirrors sweep.py's model setup exactly -- same create_ids_coeffs_file()
call to build each g's EF_Z.dat link file, same ttnopt.hamiltonian.hamiltonian()
builder -- but instead of handing the Hamiltonian to the TTN ground-state
search (gss), it builds the full Hamiltonian matrix directly via Kronecker
products of the bare per-site operators and diagonalizes it with
scipy.sparse.linalg.eigsh. This is a genuine exact-diagonalization
cross-check, independent of any TTN truncation -- only tractable at this
lattice size (3^8 = 6561), which is why sweep.py's default 5x5 lattice
(3^32) needs the TTN in the first place.

Run from the repo root:
    python diagnostics/z3_ed_3x3_scan.py
"""
import os
import tempfile

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from dotmap import DotMap
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs
from Z3_funcs.lattice_plaquettes import label_links
from ttnopt.hamiltonian import hamiltonian as build_hamiltonian
from ttnopt.src.Observable import bare_spin_operator

## System (mirrors sweep.py) ##
model_name = "z3"
Lx, Ly = 3, 3
shape = "parallelogram"
bound_state = None
chargesx, chargesy = None, None
R = None
spin_size = "1"
DIM = 3
N = nplaqs(Lx, Ly, shape)
precision = 3

## Numerics ##
g_values = np.linspace(-0.1, -1.5, 141)

OUT_PREFIX = os.path.join(os.path.dirname(__file__), "z3_ed_3x3_scan")

print(f"model: {model_name}, shape: {shape}, Lx:{Lx}, Ly:{Ly}, N: {N}, dim(H) = {DIM**N}")
print(f"bound state: {bound_state}, g: {g_values[0]:.{precision}f} -> {g_values[-1]:.{precision}f}, {len(g_values)} points")


def build_sparse_hamiltonian(ham, N, dim=DIM):
    """Full 3^N x 3^N sparse Hamiltonian from Hamiltonian.observables, built
    by Kronecker-producting each observable's bare per-site operator(s) into
    identities on every other site -- a direct, TTN-independent construction."""
    identity = sp.identity(dim, format="csr", dtype=complex)
    size = dim ** N
    H = sp.csr_matrix((size, size), dtype=complex)

    for ob in ham.observables:
        for ops, coef in zip(ob.operators_list, ob.coef_list):
            site_ops = {}
            for idx, opname in zip(ob.indices, ops):
                m = bare_spin_operator(opname, ham.spin_size[idx])
                site_ops[idx] = m if idx not in site_ops else site_ops[idx] @ m

            full = None
            for site in range(N):
                factor = sp.csr_matrix(site_ops[site]) if site in site_ops else identity
                full = factor if full is None else sp.kron(full, factor, format="csr")

            H = H + coef * full

    return H


results = []
with tempfile.TemporaryDirectory() as tmpdir:
    for g in tqdm(g_values, dynamic_ncols=True):
        run_folder = os.path.join(tmpdir, f"g_{g:.{precision}f}")
        os.makedirs(run_folder, exist_ok=True)
        filename = os.path.join(run_folder, "EF_Z.dat")

        create_ids_coeffs_file(
            Lx, Ly, shape, float(g), bound_state,
            R=R, chargesx=chargesx, chargesy=chargesy, filename=filename,
        )

        config_system = DotMap({
            "N": N,
            "spin_size": spin_size,
            "Lx": Lx,
            "Ly": Ly,
            "shape": shape,
            "bound_state": bound_state,
            "chargesx": chargesx,
            "chargesy": chargesy,
            "R": R,
            "model": {"type": model_name, "file": filename},
            "g": float(g),
            "MF_X": 1.0 / g,
            "EF_Z": filename,
            "precision": precision,
        })

        ham = build_hamiltonian(config_system)
        H = build_sparse_hamiltonian(ham, N)

        # Hermiticity sanity check (cheap relative to the diagonalization itself)
        herm_err = abs(H - H.getH()).max()
        if herm_err > 1e-8:
            print(f"  [warn] g={g:.{precision}f}: max |H - H^dagger| = {herm_err:.2e}")

        eigvals = np.sort(eigsh(H, k=2, which="SA", return_eigenvectors=False))
        results.append((g, eigvals[0], eigvals[1]))

results = np.array(results)
out_csv = f"{OUT_PREFIX}_results.csv"
np.savetxt(out_csv, results, delimiter=",", header="g,E0,E1", comments="")
print(f"saved: {out_csv}")

n_links = len(label_links(Lx, Ly, shape)[0])
g_col, e0_col, e1_col = results[:, 0], results[:, 1], results[:, 2]
g_abs = np.abs(g_col)

fig, (ax_norm, ax_raw) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

ax_norm.plot(g_abs, e0_col / (g_col * n_links), label="$E_0/(gL)$ (ground state)")
ax_norm.plot(g_abs, e1_col / (g_col * n_links), label="$E_1/(gL)$ (first excited)")
ax_norm.set_ylabel("energy / (g·L)")
ax_norm.set_title(f"Z3 gauge theory, {Lx}x{Ly} parallelogram, N={N}, L={n_links} links, vacuum sector (exact diag.)")
ax_norm.legend()

ax_raw.plot(g_abs, e0_col, label="$E_0$ (ground state)")
ax_raw.plot(g_abs, e1_col, label="$E_1$ (first excited)")
ax_raw.set_xlabel("|g|")
ax_raw.set_ylabel("energy (raw)")
ax_raw.legend()

fig.tight_layout()
out_png = f"{OUT_PREFIX}_results.png"
fig.savefig(out_png, dpi=200)
plt.close(fig)
print(f"saved: {out_png}")
