"""
z3_3x3_ttn_vs_ed.py
---------------------
Precision check: how close does the TTN ground-state search get to the
EXACT ground-state energy (diagnostics/z3_ed_3x3_scan_results.csv, from
z3_ed_3x3_scan.py) on the same 3x3 parallelogram / vacuum-sector / g-grid
setup? N=8 sites is small enough that this is a genuine accuracy
cross-check, not just a convergence-diagnostic proxy -- we HAVE the true
answer here, unlike the 5x5 case.

Defaults to the ORIGINAL (unpatched) dense GroundStateSearch; pass
PATCH3X3=1 to run patch_physics_engine instead, to check whether the
tracked-vs-true energy discrepancy found at 5x5 (see
diagnostics/z3_5x5_engine_diagnostics.py) shows up even at this trivial
system size, where ED gives an exact reference for free. At this system
size (N=8, chi<=20) the memory savings the patch exists for are
irrelevant either way -- this is purely about whether the patched
eigensolver reproduces the right answer.

Per point: cold-start a random tree at chi=CHI, run SWEEPS individual
sweeps (so the energy is recorded after every one, not just at the end),
and use the spread across the last TAIL sweeps as an error bar -- a
data-driven indicator of how settled the energy still is, not a
statistical uncertainty. The true error |E_ttn - E_ed| is plotted
alongside it as its own panel, since we have the exact answer to compare
against directly.

Run from the repo root:
    python diagnostics/z3_3x3_ttn_vs_ed.py
    PATCH3X3=1 python diagnostics/z3_3x3_ttn_vs_ed.py
CHI3X3 and SWEEPS3X3 can also be overridden via environment variable.
"""
import os
import csv
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine
from ttnopt.hamiltonian import hamiltonian as build_hamiltonian
from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs, save_edges_file
from Z3_funcs.create_ttn import get_rnd_tree

Lx, Ly, shape_ = 3, 3, "parallelogram"
bound_state = None
chargesx, chargesy = None, None
R = None
precision = 3
CHI = int(os.environ.get("CHI3X3", 9))
SWEEPS = int(os.environ.get("SWEEPS3X3", 10))
PATCH = bool(int(os.environ.get("PATCH3X3", 0)))
TAIL = 3  # last-N-sweeps spread used as the error bar

g_values = np.linspace(-0.1, -1.5, 141)
N = nplaqs(Lx, Ly, shape_)

ED_CSV = os.path.join(os.path.dirname(__file__), "z3_ed_3x3_scan_results.csv")
ed_lookup = {}
with open(ED_CSV) as f:
    for row in csv.DictReader(f):
        ed_lookup[round(float(row["g"]), 4)] = float(row["E0"])

patch_label = "patched" if PATCH else "unpatched-dense"
print(f"z3 3x3 TTN vs ED: chi={CHI}, {SWEEPS} sweeps/point (last {TAIL} for error bar), "
      f"{len(g_values)} g-points, N={N} sites, {patch_label} engine", flush=True)

tmpdir = tempfile.mkdtemp()
tensor_folder = os.path.join(tmpdir, "tensors")
os.makedirs(tensor_folder, exist_ok=True)
save_edges_file(Lx, Ly, shape_, tensor_folder=tensor_folder)

results = []
for i, g in enumerate(g_values):
    g = float(g)
    filename = os.path.join(tmpdir, f"EF_Z_{i}.dat")
    create_ids_coeffs_file(
        Lx, Ly, shape_, g, bound_state,
        R=R, chargesx=chargesx, chargesy=chargesy, filename=filename,
    )
    system_cfg = DotMap({
        "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
        "bound_state": bound_state, "chargesx": chargesx, "chargesy": chargesy, "R": R,
        "model": {"type": "z3", "file": filename},
        "g": g, "MF_X": 1.0 / g, "EF_Z": filename, "precision": precision,
    })
    ham = build_hamiltonian(system_cfg)

    psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=CHI)
    gss = GroundStateSearch(psi, ham, init_bond_dim=CHI, max_bond_dim=CHI)
    if PATCH:
        patch_physics_engine(gss)
    ref_edge = gss.psi.top_edge_id

    sweep_energies = []
    for s in range(SWEEPS):
        gss.run(
            opt_structure=0, max_num_sweep=1, verbose=False,
            energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0,
        )
        sweep_energies.append(gss.energy.get(ref_edge))

    final_e = sweep_energies[-1]
    tail = sweep_energies[-TAIL:]
    err = (max(tail) - min(tail)) / 2.0

    e_ed = ed_lookup.get(round(g, 4))
    true_err = abs(final_e - e_ed) if e_ed is not None else float("nan")

    results.append((g, final_e, err, e_ed, true_err))
    if i % 20 == 0 or i == len(g_values) - 1:
        print(f"  [{i+1}/{len(g_values)}] g={g:.3f}  E_ttn={final_e:.6f}  "
              f"E_ed={e_ed:.6f}  |diff|={true_err:.2e}  spread={err:.2e}", flush=True)

results = np.array(results)
out_csv = os.path.join(os.path.dirname(__file__), f"z3_3x3_ttn_vs_ed_chi{CHI}_{patch_label}.csv")
np.savetxt(out_csv, results, delimiter=",",
           header="g,E_ttn,spread_err,E_ed,abs_diff", comments="")
print(f"\nsaved: {out_csv}")

g_col, e_ttn, spread, e_ed, true_err = results.T
g_abs = np.abs(g_col)

fig, (ax_e, ax_diff) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

ax_e.plot(g_abs, e_ed, "-", color="#2a78d6", linewidth=2, label="exact (ED)", zorder=2)
ax_e.errorbar(g_abs, e_ttn, yerr=spread, fmt=".", color="#eb6834", ecolor="#eb6834",
              elinewidth=1, capsize=2, markersize=4, label=f"TTN (chi={CHI}, {SWEEPS} sweeps)", zorder=3)
ax_e.set_ylabel("ground state energy")
ax_e.set_title(f"3x3 parallelogram, vacuum: TTN ({patch_label}) vs exact diagonalization (chi={CHI})")
ax_e.legend()

ax_diff.semilogy(g_abs, np.abs(true_err), ".-", color="#822727", markersize=4, label="|E_ttn - E_ed| (true error)")
ax_diff.semilogy(g_abs, spread, ".-", color="#c05621", markersize=4, alpha=0.6,
                  label=f"last-{TAIL}-sweep spread (convergence proxy)")
ax_diff.set_xlabel("|g|")
ax_diff.set_ylabel("energy error")
ax_diff.legend()
ax_diff.grid(alpha=0.3, which="both")

fig.tight_layout()
out_png = os.path.join(os.path.dirname(__file__), f"z3_3x3_ttn_vs_ed_chi{CHI}_{patch_label}.png")
fig.savefig(out_png, dpi=200)
plt.close(fig)
print(f"saved: {out_png}")

print(f"\nmax |E_ttn - E_ed| over all g: {np.nanmax(np.abs(true_err)):.3e}")
print(f"mean |E_ttn - E_ed| over all g: {np.nanmean(np.abs(true_err)):.3e}")
print("\nDONE")
