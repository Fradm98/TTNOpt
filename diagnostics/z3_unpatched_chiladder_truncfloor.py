"""
z3_unpatched_chiladder_truncfloor.py
--------------------------------------
Professor's truncation-floor test, extended: does the persistent
degeneracy-driven truncation gap seen at chi=20 (edges 57/58, ~3e-6 with
the now-fixed tree topology) get resolved by simply using a large enough
bond dimension -- staying fully UNPATCHED (trusted dense engine)
throughout, unlike the earlier chi=20->chi=50 test which switched to the
patched engine for the chi=50 stage?

Runs the SAME chi ladder as the production g-scan (ascend 9->18->27
cheaply, converge hard at 50), entirely unpatched, at g=-1.0, with
E_lanczos vs E_after_truncation instrumented on every edge at every stage.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from z3_5x5_engine_diagnostics import make_hamiltonian, Lx, Ly, SHAPE, G, PRECISION
from ttnopt.src import GroundStateSearch
from Z3_funcs.create_ttn import get_rnd_tree

WARMSTART_DIR = os.path.join(os.path.dirname(__file__), "z3_5x5_warmstart_results")
os.makedirs(WARMSTART_DIR, exist_ok=True)

CHIS = [9, 18, 27, 50]
SWEEPS = [3, 3, 20, 20]

ham, tensor_folder = make_hamiltonian()
psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=SHAPE, path=tensor_folder, chi=CHIS[0])
gss = GroundStateSearch(psi, ham, init_bond_dim=CHIS[0], max_bond_dim=CHIS[0])
ref_edge = gss.psi.top_edge_id

print(f"[unpatched chi-ladder trunc-floor test] g={G}, chis={CHIS}, sweeps={SWEEPS}, "
      f"fully unpatched, fixed tree topology", flush=True)

all_diag = []
stage_offsets = []
for stage_idx, (chi, n_sweeps) in enumerate(zip(CHIS, SWEEPS)):
    if stage_idx > 0:
        gss.max_bond_dim = chi
        gss.move_canonical_center(ref_edge)
        gss._prime_renormalized_operators()
    diag_edges = list(set(e[2] for e in gss.psi.edges))
    print(f"\n[chi={chi}] {n_sweeps} sweeps, instrumenting {len(diag_edges)} edges", flush=True)
    stage_offsets.append((chi, len(all_diag)))
    gss.run(opt_structure=0, max_num_sweep=n_sweeps, verbose=True, diagnostic_edges=diag_edges,
            energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
    all_diag.extend([dict(r, chi=chi) for r in gss.local_update_diagnostics])
    final_e = gss.local_update_diagnostics[-1]["E_lanczos"]
    print(f"[chi={chi}] final E_lanczos on last instrumented update = {final_e:.10f}", flush=True)

tag = "unpatched_chiladder_9-18-27-50_truncfloor_g-1.000"
csv_path = os.path.join(WARMSTART_DIR, f"{tag}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["chi", "update_index", "sweep", "edge_id", "E_before", "E_lanczos", "E_after_truncation"])
    for i, r in enumerate(all_diag):
        writer.writerow([r["chi"], i, r["sweep"], r["edge_id"], r["E_before"], r["E_lanczos"], r["E_after_truncation"]])
print(f"\nsaved: {csv_path}")

print("\n=== per-chi summary: worst edges by max |E_after_truncation - E_lanczos| ===")
for chi, offset in stage_offsets:
    stage_records = [r for r in all_diag if r["chi"] == chi]
    diffs_by_edge = {}
    for r in stage_records:
        d = abs(r["E_after_truncation"] - r["E_lanczos"])
        diffs_by_edge.setdefault(r["edge_id"], []).append(d)
    worst = sorted(diffs_by_edge.items(), key=lambda kv: -max(kv[1]))[:5]
    print(f"\n[chi={chi}]")
    for edge_id, ds in worst:
        print(f"  edge {edge_id}: max diff = {max(ds):.3e}  (n updates: {len(ds)})")

print("\nDONE")
