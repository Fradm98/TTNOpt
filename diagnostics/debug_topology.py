import os, tempfile

from ttnopt.src import GroundStateSearch  # noqa: F401 -- import order avoids a circular import
from Z3_funcs.create_graph_file import save_edges_file, nplaqs
from Z3_funcs.create_ttn import get_rnd_tree
from Z3_funcs.independent_energy import _edge_occurrences

Lx, Ly, shape_ = 5, 5, "parallelogram"
N = nplaqs(Lx, Ly, shape_)
tmpdir = tempfile.mkdtemp()
tensor_folder = os.path.join(tmpdir, "tensors")
os.makedirs(tensor_folder, exist_ok=True)
save_edges_file(Lx, Ly, shape_, tensor_folder=tensor_folder)

psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=10)

print(f"N (physical edges expected) = {N}")
print(f"len(psi.tensors) = {len(psi.tensors)}")
print(f"len(psi.physical_edges) = {len(psi.physical_edges)}")
print(f"psi.top_edge_id = {psi.top_edge_id}")
print(f"psi.canonical_center_edge_id = {psi.canonical_center_edge_id}")
print(f"psi.gauge_tensor shape = {None if psi.gauge_tensor is None else psi.gauge_tensor.shape}")
print(f"psi.physical_edges (sorted) = {sorted(psi.physical_edges)}")

cc = psi.canonical_center_edge_id
positions_of_cc = [(t, l) for t, edge in enumerate(psi.edges) for l, e in enumerate(edge) if e == cc]
print(f"positions referencing canonical_center_edge_id ({cc}): {positions_of_cc}")
positions_of_top = [(t, l) for t, edge in enumerate(psi.edges) for l, e in enumerate(edge) if e == psi.top_edge_id]
print(f"positions referencing top_edge_id ({psi.top_edge_id}): {positions_of_top}")

lengths = set(len(e) for e in psi.edges)
print(f"distinct edge-list lengths across tensors = {lengths}")

occ = _edge_occurrences(psi.edges)
print(f"total distinct edge ids referenced = {len(occ)}")

counts = {}
for eid, positions in occ.items():
    counts.setdefault(len(positions), []).append(eid)

for n_occ, eids in sorted(counts.items()):
    print(f"\n{n_occ} occurrence(s): {len(eids)} edge ids")
    if n_occ == 1:
        physical = [e for e in eids if e in psi.physical_edges]
        top = [e for e in eids if e == psi.top_edge_id]
        other = [e for e in eids if e not in psi.physical_edges and e != psi.top_edge_id]
        print(f"  physical: {len(physical)}")
        print(f"  top_edge: {len(top)}")
        print(f"  OTHER (unexpected!): {len(other)} -> {other}")
    elif n_occ != 2:
        print(f"  ids: {eids}")

print("\nDONE")
