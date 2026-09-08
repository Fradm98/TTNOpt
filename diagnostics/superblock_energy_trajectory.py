"""
superblock_energy_trajectory.py
---------------------------------
Per advisor's request: record the TOTAL energy from the Lanczos
diagonalization of EVERY superblock Hamiltonian across an entire run (not
just once per sweep, not just at one reference edge), plotted as one
continuous trajectory:
    x axis = cumulative superblock-update index (sweep 1's updates,
             immediately followed by sweep 2's, then sweep 3's, ...)
    y axis = corresponding superblock Lanczos energy
with sweep boundaries marked. This distinguishes "decreasing but noisy",
"occasional large jumps", "systematic per-sweep jump", and "oscillatory/
stationary" behavior in a way a single per-sweep summary point cannot.

Tree topology is fixed throughout (opt_structure=0, as in every other
diagnostic script in this project) -- confirmed directly from
TwoSiteUpdater.decompose_two_tensors: opt_structure==0 always uses the
fixed bipartition [psi[0],psi[1]]|[psi[2],psi[3]], never the alternative-
reconnection candidates opt_structure==1/2 try. So this run already
satisfies "optimize only the tensors, not the structure" -- nothing extra
needed for that part.

Run from the repo root, e.g.:
    OMP_NUM_THREADS=<n> MKL_NUM_THREADS=<n> OPENBLAS_NUM_THREADS=<n> python diagnostics/superblock_energy_trajectory.py
"""

import os, tempfile, csv

import matplotlib
matplotlib.use("Agg")  # headless-safe (this runs on presto, not just locally)
import matplotlib.pyplot as plt
from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine
from ttnopt.hamiltonian import hamiltonian
from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.create_ttn import get_rnd_tree

Lx, Ly, shape_ = 5, 5, "parallelogram"
g = -1.0
CHI = 20
N_SWEEPS = 5

OUT_DIR = "../5_Z3/results/energy_data"
FIG_DIR = "../5_Z3/figures"

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

print(f"Superblock energy trajectory: Lx={Lx}, Ly={Ly}, vacuum, g={g}, chi={CHI}, "
      f"{N_SWEEPS} sweeps, opt_structure=0 (tree topology fixed throughout)", flush=True)

psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=CHI)
ham = hamiltonian(system_cfg)
gss = GroundStateSearch(psi, ham, init_bond_dim=CHI, max_bond_dim=CHI)
patch_physics_engine(gss)

gss.run(
    opt_structure=0,
    max_num_sweep=N_SWEEPS,
    verbose=True,
    energy_convergence_threshold=0.0,
    entanglement_convergence_threshold=0.0,
)

traj = gss.superblock_energy_trajectory
print(f"\n{len(traj)} total superblock updates recorded across {N_SWEEPS} sweeps", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
tag = f"g{g:.3f}_chi{CHI}"

# Raw data alongside the plot, so it can be resliced/replotted without rerunning.
csv_path = os.path.join(OUT_DIR, f"superblock_energy_trajectory_{tag}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["update_index", "sweep", "edge_id", "energy"])
    for i, rec in enumerate(traj):
        writer.writerow([i, rec["sweep"], rec["edge_id"], rec["energy"]])
print(f"raw data written to {csv_path}", flush=True)

# Sweep boundaries for vertical markers: the midpoint between the last
# update of one sweep and the first of the next.
sweep_boundaries = []
current_sweep = traj[0]["sweep"] if traj else None
for i, rec in enumerate(traj):
    if rec["sweep"] != current_sweep:
        sweep_boundaries.append(i - 0.5)
        current_sweep = rec["sweep"]

energies = [rec["energy"] for rec in traj]
indices = list(range(len(traj)))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(indices, abs(energies), "-", color="#2b6cb0", linewidth=0.8, marker=".", markersize=2)
for b in sweep_boundaries:
    ax.axvline(b, color="red", linestyle="--", linewidth=0.6, alpha=1)
ax.set_xlabel("cumulative superblock update index")
ax.set_ylabel("superblock Lanczos energy")
ax.set_title(
    f"Superblock energy trajectory -- Lx={Lx}, Ly={Ly}, g={g}, chi={CHI}, "
    f"{N_SWEEPS} sweeps (tree topology fixed, opt_structure=0)"
)
ax.grid(alpha=0.3)

ax.set_yscale('log')

fig.tight_layout()

png_path = os.path.join(FIG_DIR, f"superblock_energy_trajectory_{tag}.png")
fig.savefig(png_path, dpi=200)
print(f"plot written to {png_path}", flush=True)

print("\nDONE")
