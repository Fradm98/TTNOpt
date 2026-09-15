"""
z3_patched_chiladder_truncfloor.py
--------------------------------------
Patched-engine counterpart of z3_unpatched_chiladder_truncfloor.py -- the
SAME chi ladder (9->18->27->50), SAME sweep budget (3,3,20,20), SAME g=-1.0,
SAME fixed tree topology, SAME per-edge E_lanczos vs E_after_truncation
instrumentation, but with patch_physics_engine() applied throughout
instead of the trusted dense engine. Direct, apples-to-apples comparison
against the unpatched run.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from z3_5x5_engine_diagnostics import make_hamiltonian, Lx, Ly, SHAPE, G, PRECISION
from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine
from Z3_funcs.create_ttn import get_rnd_tree

WARMSTART_DIR = os.path.join(os.path.dirname(__file__), "z3_5x5_warmstart_results")
os.makedirs(WARMSTART_DIR, exist_ok=True)

CHIS = [9, 18, 27, 50]
SWEEPS = [3, 3, 20, 20]

ham, tensor_folder = make_hamiltonian()
psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=SHAPE, path=tensor_folder, chi=CHIS[0])
gss = GroundStateSearch(psi, ham, init_bond_dim=CHIS[0], max_bond_dim=CHIS[0])
patch_physics_engine(gss)
ref_edge = gss.psi.top_edge_id

print(f"[PATCHED chi-ladder trunc-floor test] g={G}, chis={CHIS}, sweeps={SWEEPS}, "
      f"patched throughout, fixed tree topology", flush=True)

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

tag = "patched_chiladder_9-18-27-50_truncfloor_g-1.000"
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
