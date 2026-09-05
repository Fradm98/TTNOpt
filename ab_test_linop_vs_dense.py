"""
ab_test_linop_vs_dense.py
--------------------------
Deterministic, matched A/B comparison between the original dense
PhysicsEngine.lanczos() and the TTNLinearOperator-patched ttn_eigensolver().

Design (per advisor's original suggestion, revisited because the chi-scan
diagnostics showed trunc_loss/lanczos_gain NOT improving from chi=50->150,
which is suspicious enough to re-check the modified eigensolver directly):

  - Build ONE random initial tree (fixed seed), deep-copy it into two
    independent GroundStateSearch instances -- one left as the original
    dense implementation, one patched with TTNLinearOperator.
  - Run exactly ONE sweep on each, with diagnostic_edges covering every
    internal edge (not just the top edge), so every single local update in
    that sweep is recorded on both sides.
  - Compare E_before / E_lanczos / E_after_truncation edge-by-edge.

Because both runs start from bit-identical tensors and the sweep traversal
order is deterministic, this checks mathematical equivalence of the two
eigensolvers on matched inputs -- it does NOT depend on, and is not
confounded by, the run-to-run seed-sensitivity already observed in
chi_scan_diagnostic.py (no seed is set there).

Run from the repo root:
    python ab_test_linop_vs_dense.py
"""

import copy
import os
import tempfile

import numpy as np
from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine
from ttnopt.hamiltonian import hamiltonian
from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.create_ttn import get_rnd_tree

SEED = 42
Lx, Ly, shape_ = 5, 5, "parallelogram"
g = -1.0
CHI = 50

N = nplaqs(Lx, Ly, shape_)
tmpdir = tempfile.mkdtemp()
tensor_folder = os.path.join(tmpdir, "tensors")
os.makedirs(tensor_folder, exist_ok=True)
save_edges_file(Lx, Ly, shape_, tensor_folder=tensor_folder)
filename = os.path.join(tmpdir, "EF_Z.dat")
create_ids_coeffs_file(Lx, Ly, shape_, g, None, filename=filename)  # vacuum
mf = 1 / g

system_cfg = DotMap({
    "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
    "model": {"type": "z3", "file": filename},
    "MF_X": mf, "EF_Z": filename,
})

print(f"A/B test: dense lanczos() vs TTNLinearOperator, Lx={Lx}, Ly={Ly}, "
      f"vacuum, g={g}, chi={CHI}, seed={SEED}, 1 sweep, all internal edges "
      f"instrumented", flush=True)

# One random initial tree, built once, then deep-copied so both engines
# start from bit-identical tensors.
np.random.seed(SEED)
psi_template = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=CHI)

ham_dense = hamiltonian(system_cfg)
gss_dense = GroundStateSearch(
    copy.deepcopy(psi_template), ham_dense, init_bond_dim=CHI, max_bond_dim=CHI
)
# gss_dense left unpatched -- uses the original PhysicsEngine.lanczos().

ham_linop = hamiltonian(system_cfg)
gss_linop = GroundStateSearch(
    copy.deepcopy(psi_template), ham_linop, init_bond_dim=CHI, max_bond_dim=CHI
)
patch_physics_engine(gss_linop)

# Every internal edge (every tensor's own parent-edge id, which includes the
# top edge) -- covers every possible two-site canonical-center position, so
# diagnostic_edges instruments the WHOLE sweep, not just one edge.
all_internal_edges = set(edge[2] for edge in gss_dense.psi.edges)
print(f"instrumenting {len(all_internal_edges)} internal edges", flush=True)

# Reseed immediately before each run -- guards against any np.random
# consumption elsewhere (e.g. degeneracy-breaking in decompose_two_tensors)
# making the two runs diverge even though eigsh/lanczos themselves are
# given deterministic v0 vectors built straight from the tensors.
np.random.seed(SEED)
gss_dense.run(
    opt_structure=0, max_num_sweep=1, verbose=True,
    diagnostic_edges=all_internal_edges,
    energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0,
)

np.random.seed(SEED)
gss_linop.run(
    opt_structure=0, max_num_sweep=1, verbose=True,
    diagnostic_edges=all_internal_edges,
    energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0,
)

dense_recs = gss_dense.local_update_diagnostics
linop_recs = gss_linop.local_update_diagnostics
print(f"\ndense: {len(dense_recs)} local-update records, "
      f"linop: {len(linop_recs)} local-update records", flush=True)

n = min(len(dense_recs), len(linop_recs))
if len(dense_recs) != len(linop_recs):
    print("!! record counts differ -- the two sweeps did not traverse the "
          "same edges in the same order, comparison below is only over the "
          "first common records !!", flush=True)

print(f"\n{'edge_id':>8} {'match':>6} {'dE_before':>14} {'dE_lanczos':>14} {'dE_after_trunc':>16}")
max_diffs = {"E_before": 0.0, "E_lanczos": 0.0, "E_after_truncation": 0.0}
mismatched_edges = []
for i in range(n):
    d, l = dense_recs[i], linop_recs[i]
    edge_match = d["edge_id"] == l["edge_id"]
    if not edge_match:
        mismatched_edges.append(i)
    diffs = {k: abs(d[k] - l[k]) for k in max_diffs}
    for k in max_diffs:
        max_diffs[k] = max(max_diffs[k], diffs[k])
    print(f"{d['edge_id']:>8} {str(edge_match):>6} {diffs['E_before']:>14.3e} "
          f"{diffs['E_lanczos']:>14.3e} {diffs['E_after_truncation']:>16.3e}")

print(f"\n--- summary over {n} matched local updates ---")
print(f"max |dE_before|          = {max_diffs['E_before']:.6e}")
print(f"max |dE_lanczos|         = {max_diffs['E_lanczos']:.6e}")
print(f"max |dE_after_truncation|= {max_diffs['E_after_truncation']:.6e}")
if mismatched_edges:
    print(f"!! edge_id mismatched at record indices: {mismatched_edges} -- "
          f"sweep traversal order diverged, investigate before trusting the "
          f"diffs above !!")
else:
    print("edge_id sequence matched exactly between both runs.")

print(f"\nfinal energy (ref edge, dense): {gss_dense.energy.get(gss_dense.psi.top_edge_id)}")
print(f"final energy (ref edge, linop): {gss_linop.energy.get(gss_linop.psi.top_edge_id)}")
print("\nDONE")
