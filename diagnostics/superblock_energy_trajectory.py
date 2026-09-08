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

Resume support: TOTAL_SWEEPS is a CUMULATIVE target across however many
times you run this script, not "sweeps to run this invocation". The CSV
at csv_path is append-only and is the source of truth for how many sweeps
have already been recorded (sweep numbers keep counting up across
invocations rather than restarting at 1); a dedicated tensor checkpoint
(separate from chi_scan_diagnostic.py/g_chi_scan_diagnostic.py's own
checkpoints at the same (g, chi) -- merging them would let unrelated runs
clobber each other) lets the optimization itself continue rather than
cold-starting from a fresh random tree every time. The plot is always
regenerated from the full persisted CSV, not just this invocation's data,
so it reflects the complete history so far regardless of how many restarts
it took to get there.

Run from the repo root, e.g.:
    OMP_NUM_THREADS=<n> MKL_NUM_THREADS=<n> OPENBLAS_NUM_THREADS=<n> python diagnostics/superblock_energy_trajectory.py

CHI and TOTAL_SWEEPS can be overridden via environment variables (so the
same script runs chi=20/50/100 without duplicating it -- see the
TRAJ_CHI_* .pbs files):
    TRAJ_CHI=100 TRAJ_TOTAL_SWEEPS=15 python diagnostics/superblock_energy_trajectory.py
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
from Z3_funcs.hdf5_manager import save_tensor, load_tensor, tensor_exists

Lx, Ly, shape_ = 5, 5, "parallelogram"
g = -1.0
CHI = int(os.environ.get("TRAJ_CHI", 20))
TOTAL_SWEEPS = int(os.environ.get("TRAJ_TOTAL_SWEEPS", 15))  # cumulative target across all invocations -- see resume support above
PRECISION = 3

OUT_DIR = "../5_Z3/results/energy_data"
FIG_DIR = "../5_Z3/figures"

# Dedicated to this script -- NOT shared with chi_scan_diagnostic.py/
# g_chi_scan_diagnostic.py's own checkpoints at the same (g, chi), which
# track different things and would clobber this one (or vice versa).
CHECKPOINT_FILE = "../5_Z3/logs/superblock_trajectory_checkpoint.hdf5"
os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

tag = f"g{g:.3f}_chi{CHI}"
csv_path = os.path.join(OUT_DIR, f"superblock_energy_trajectory_{tag}.csv")

# How far a previous invocation already got, read from the CSV itself (the
# source of truth for "how many sweeps are recorded so far").
completed_sweeps = 0
existing_row_count = 0
if os.path.exists(csv_path):
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            completed_sweeps = max(completed_sweeps, int(row["sweep"]))
            existing_row_count += 1

remaining_sweeps = TOTAL_SWEEPS - completed_sweeps

print(f"Superblock energy trajectory: Lx={Lx}, Ly={Ly}, vacuum, g={g}, chi={CHI}, "
      f"target={TOTAL_SWEEPS} total sweeps, opt_structure=0 (tree topology fixed throughout)",
      flush=True)
if completed_sweeps > 0:
    print(f"Resuming: {completed_sweeps} sweep(s) already recorded in {csv_path}, "
          f"running {max(remaining_sweeps, 0)} more.", flush=True)

if remaining_sweeps > 0:
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
    ham = hamiltonian(system_cfg)

    if completed_sweeps > 0 and tensor_exists(
        CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, CHI
    ):
        psi, _ = load_tensor(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, CHI)
        print(f"loaded tensor checkpoint from {CHECKPOINT_FILE}", flush=True)
    else:
        psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=CHI)

    gss = GroundStateSearch(psi, ham, init_bond_dim=CHI, max_bond_dim=CHI)
    patch_physics_engine(gss)

    gss.run(
        opt_structure=0,
        max_num_sweep=remaining_sweeps,
        verbose=True,
        energy_convergence_threshold=0.0,
        entanglement_convergence_threshold=0.0,
    )

    new_records = gss.superblock_energy_trajectory
    print(f"\n{len(new_records)} new superblock updates recorded this invocation", flush=True)

    save_tensor(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, CHI, gss.psi)
    print(f"checkpoint saved to {CHECKPOINT_FILE}", flush=True)

    # Append, never overwrite -- sweep numbers offset by completed_sweeps
    # so they keep counting up rather than restarting at 1 (run() always
    # labels its own sweeps from 1 regardless of prior invocations).
    write_header = existing_row_count == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["update_index", "sweep", "edge_id", "energy"])
        for i, rec in enumerate(new_records):
            writer.writerow(
                [existing_row_count + i, rec["sweep"] + completed_sweeps, rec["edge_id"], rec["energy"]]
            )
    print(f"appended to {csv_path}", flush=True)
else:
    print("target sweep count already reached -- nothing to run, just replotting.", flush=True)

# ── (re)plot from the FULL persisted history, not just this invocation's
#    in-memory data, so the figure always reflects everything recorded so
#    far regardless of how many times this script has been (re)run. ──────
indices, sweeps, energies = [], [], []
with open(csv_path, "r", newline="") as f:
    for row in csv.DictReader(f):
        indices.append(int(row["update_index"]))
        sweeps.append(int(row["sweep"]))
        energies.append(float(row["energy"]))

sweep_boundaries = [i - 0.5 for i in range(1, len(sweeps)) if sweeps[i] != sweeps[i - 1]]


def _make_plot(idx, en, title_suffix, out_path, boundaries):
    # Linear scale on the actual signed energy -- NOT abs()+log. All these
    # energies cluster tightly near a single value (e.g. ~-60), so abs()+log
    # compresses exactly the fine oscillatory/plateau structure your
    # advisor is asking about into a flat line at the top, while the
    # initial cold-start transient (which starts near 0 before any
    # information has propagated through the tree) dominates the whole
    # visible range -- hiding the very thing this plot is meant to reveal.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, en, "-", color="#2b6cb0", linewidth=0.8, marker=".", markersize=2)
    for b in boundaries:
        ax.axvline(b, color="red", linestyle="--", linewidth=0.6, alpha=1)
    ax.set_xlabel("cumulative superblock update index")
    ax.set_ylabel("superblock Lanczos energy")
    ax.set_title(
        f"Superblock energy trajectory{title_suffix} -- Lx={Lx}, Ly={Ly}, g={g}, chi={CHI}, "
        f"{max(sweeps) if sweeps else 0} sweeps so far (tree topology fixed, opt_structure=0)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


png_path = os.path.join(FIG_DIR, f"superblock_energy_trajectory_{tag}.png")
_make_plot(indices, energies, "", png_path, sweep_boundaries)
print(f"plot written to {png_path}", flush=True)

# Zoomed companion: drop the first couple of sweeps (the cold-start
# transient, where energy is still far from the plateau) so the y-axis
# autoscales to the fine late-sweep fluctuations instead of being swamped
# by the initial ramp-up -- this is the view that actually answers
# "decreasing with small fluctuations vs big jumps vs oscillatory/
# stationary" for the sweeps that matter.
ZOOM_FROM_SWEEP = 3
zoom = [(i, s, e) for i, s, e in zip(indices, sweeps, energies) if s >= ZOOM_FROM_SWEEP]
if zoom:
    z_idx, _z_sw, z_en = zip(*zoom)
    z_boundaries = [b for b in sweep_boundaries if b >= min(z_idx)]
    zoom_png_path = os.path.join(FIG_DIR, f"superblock_energy_trajectory_{tag}_zoom.png")
    _make_plot(
        z_idx, z_en, f" (sweep >= {ZOOM_FROM_SWEEP}, zoomed)", zoom_png_path, z_boundaries
    )
    print(f"zoomed plot written to {zoom_png_path}", flush=True)

print("\nDONE")
