"""
compare_superblock_trajectories.py
-------------------------------------
Overlays superblock_energy_trajectory.py's recorded trajectories for
several chi values on the same axes, to see whether the oscillatory/
stationary plateau behavior found at chi=20 persists, shrinks, or
disappears as chi grows.

The x axis (cumulative superblock-update index) is directly comparable
across different chi: the number of superblock updates per sweep is a
property of the tree TOPOLOGY (fixed here, opt_structure=0), not of chi --
confirmed empirically (42 updates/sweep at both chi=20 and chi=100 for
this 5x5 system) -- so sweep boundaries land at the same index regardless
of chi, and are drawn once rather than once per chi.

Reads whatever CSVs are already present under OUT_DIR (produced by
superblock_energy_trajectory.py, possibly on a different machine -- copy
them into the same OUT_DIR here first, e.g. via scp from presto) and skips
any chi missing its file with a warning, rather than failing outright.

Run from the repo root:
    python diagnostics/compare_superblock_trajectories.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

g = -1.0
CHIS = [20, 50, 100]
PRECISION = 3
ZOOM_FROM_SWEEP = 3

OUT_DIR = "../5_Z3/results/energy_data"
FIG_DIR = "../5_Z3/figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {20: "#2b6cb0", 50: "#c05621", 100: "#2f855a"}


def _load(chi):
    path = os.path.join(OUT_DIR, f"superblock_energy_trajectory_g{g:.{PRECISION}f}_chi{chi}.csv")
    if not os.path.exists(path):
        print(f"!! no data for chi={chi} at {path} -- skipping "
              f"(copy it here first if it was generated on another machine) !!")
        return None
    indices, sweeps, energies = [], [], []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            indices.append(int(row["update_index"]))
            sweeps.append(int(row["sweep"]))
            energies.append(float(row["energy"]))
    print(f"chi={chi}: {len(indices)} updates, {max(sweeps) if sweeps else 0} sweeps, from {path}")
    return {"indices": indices, "sweeps": sweeps, "energies": energies}


data = {chi: _load(chi) for chi in CHIS}
data = {chi: d for chi, d in data.items() if d is not None}

if not data:
    raise RuntimeError(
        f"No trajectory CSVs found for any of chi={CHIS} under {OUT_DIR} -- "
        f"run superblock_energy_trajectory.py (locally and/or on presto) first."
    )

# Sweep boundaries computed once, from whichever series has the most
# sweeps recorded -- chi-independent, so any one series would do, but the
# longest one covers the boundaries the others might be missing at the end.
longest_chi = max(data, key=lambda c: len(data[c]["indices"]))
longest = data[longest_chi]
sweep_boundaries_full = [
    i - 0.5 for i in range(1, len(longest["sweeps"])) if longest["sweeps"][i] != longest["sweeps"][i - 1]
]


def _make_comparison_plot(title_suffix, out_path, min_sweep=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    max_index_plotted = 0
    for chi, d in data.items():
        if min_sweep is None:
            idx, en = d["indices"], d["energies"]
        else:
            idx, en = zip(*[(i, e) for i, s, e in zip(d["indices"], d["sweeps"], d["energies"]) if s >= min_sweep])
        ax.plot(idx, en, "-", color=COLORS.get(chi, None), linewidth=0.8,
                marker=".", markersize=2, label=f"chi={chi}")
        max_index_plotted = max(max_index_plotted, max(idx))
    for b in sweep_boundaries_full:
        if b <= max_index_plotted:
            ax.axvline(b, color="red", linestyle="--", linewidth=0.6, alpha=1)
    ax.set_xlabel("cumulative superblock update index")
    ax.set_ylabel("superblock Lanczos energy")
    ax.set_title(f"Superblock energy trajectory comparison{title_suffix} -- g={g}, "
                 f"chis={sorted(data.keys())} (tree topology fixed, opt_structure=0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"plot written to {out_path}")


_make_comparison_plot("", os.path.join(FIG_DIR, f"superblock_trajectory_compare_g{g:.{PRECISION}f}.png"))
_make_comparison_plot(
    f" (sweep >= {ZOOM_FROM_SWEEP}, zoomed)",
    os.path.join(FIG_DIR, f"superblock_trajectory_compare_g{g:.{PRECISION}f}_zoom.png"),
    min_sweep=ZOOM_FROM_SWEEP,
)

print("\nDONE")
