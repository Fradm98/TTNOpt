"""
test_independent_energy.py
----------------------------
Quick local sanity check: does the from-scratch, fully independent
<psi|H|psi> (Z3_funcs/independent_energy.py) agree with the ref_energy
GroundStateSearch reports from its renormalized-operator machinery?

5x5, vacuum, g=-2.0, chi=50 -- small and cheap enough to run locally.
"""

import os
import tempfile

from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine
from ttnopt.hamiltonian import hamiltonian
from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.create_ttn import get_rnd_tree
from Z3_funcs.independent_energy import independent_energy

Lx, Ly, shape_ = 5, 5, "parallelogram"
g = -2.0
CHI = 50
N_SWEEPS = 5

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

print(f"Independent-energy check: Lx={Lx}, Ly={Ly}, vacuum, g={g}, chi={CHI}, "
      f"{N_SWEEPS} sweeps", flush=True)

psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=CHI)
ham = hamiltonian(system_cfg)
gss = GroundStateSearch(psi, ham, init_bond_dim=CHI, max_bond_dim=CHI)
patch_physics_engine(gss)
ref_edge = gss.psi.top_edge_id

gss.run(
    opt_structure=0,
    max_num_sweep=N_SWEEPS,
    verbose=True,
    diagnostic_edges=[ref_edge],
    energy_convergence_threshold=0.0,
    entanglement_convergence_threshold=0.0,
)

ref_energy = gss.energy[ref_edge]
indep_energy = independent_energy(gss.psi, gss.hamiltonian)
abs_diff = abs(ref_energy - indep_energy)
rel_diff = abs_diff / abs(ref_energy)

print(f"\nref_energy (renormalized, from sweep)   = {ref_energy:.12f}")
print(f"independent_energy (full from-scratch)  = {indep_energy:.12f}")
print(f"abs diff = {abs_diff:.6e}")
print(f"rel diff = {rel_diff:.6e}")
print("\nDONE")
