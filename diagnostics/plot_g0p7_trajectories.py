"""
plot_g0p7_trajectories.py
----------------------------
E-vs-sweep trajectory plot for g=-0.700, chi=27 (5x5 parallelogram),
analogous to z3_3x3_ttn_vs_ed.py's convergence panel -- except there is no
exact-diagonalization reference at this system size, and (per the run-to-run
non-determinism finding: a fresh independent dense rerun of this exact g
landed on a DIFFERENT value than an earlier dense run did) neither is any
single prior run a "ground truth" to plot against.

Instead this overlays every branch-test trajectory logged in
diagnostics/eigsh_full_ladder_test.py / dense_from_ascend_test.py's runs
(patched at various tol/num_eigvals, dense, from both a patched-ascended and
a dense-ascended chi=18 checkpoint), parsed directly from their sweep logs,
against ONE fixed sweep-by-sweep x-axis. The only "reference" drawn is the
LOWEST energy actually observed across all trajectories -- the variationally
best answer found so far, not a value assumed correct in advance.

Usage:
  python diagnostics/plot_g0p7_trajectories.py
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "/Users/fradm/Desktop/projects/5_Z3_eigsh_test/logs"
OUT_DIR = "/Users/fradm/Desktop/projects/5_Z3_eigsh_test/figures"

# (log file, label, color, linestyle). Colors assigned by hand so
# patched/dense pairs sharing a starting checkpoint share a hue.
SOURCES = [
    ("eigsh_grid_test.log", None, None, None),  # multi-config, handled specially below
    ("dense_from_ascend_test_FIXED.log", "dense (patched-ascend start)", "#2a78d6", "-"),
    ("dense_ascend_then_dense.log", "dense (dense-ascend start)", "#1f9e5c", "-"),
    ("dense_ascend_then_patched.log", "patched tol=1e-10 k=1 (dense-ascend start)", "#1f9e5c", "--"),
]

SWEEP_RE = re.compile(r"sweep (\d+)/\d+ done in [\d.]+s.*?E=(-?[\d.]+)")
CONFIG_RE = re.compile(r"--- running tol=([\d.e+-]+) k=(\d+) ---")


def parse_single_trajectory(path):
    """[(sweep_num, energy), ...] from a log with one continuous run."""
    pts = []
    with open(path) as f:
        for line in f:
            m = SWEEP_RE.search(line)
            if m:
                pts.append((int(m.group(1)), float(m.group(2))))
    return pts


def parse_multi_config(path):
    """{(tol, k): [(sweep_num, energy), ...]} from eigsh_grid_test.log's
    concatenated per-config runs (each preceded by a '--- running ... ---'
    marker)."""
    trajectories = {}
    current_key = None
    with open(path) as f:
        for line in f:
            cm = CONFIG_RE.search(line)
            if cm:
                current_key = (float(cm.group(1)), int(cm.group(2)))
                trajectories[current_key] = []
                continue
            sm = SWEEP_RE.search(line)
            if sm and current_key is not None:
                trajectories[current_key].append((int(sm.group(1)), float(sm.group(2))))
    return trajectories


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    all_energies = []

    # Patched grid (6 configs) -- greyscale-ish, thin lines, since the point
    # is they all overlap, not that any one is individually important.
    multi_path = os.path.join(LOG_DIR, "eigsh_grid_test.log")
    if os.path.exists(multi_path):
        trajectories = parse_multi_config(multi_path)
        patched_colors = plt.get_cmap("Oranges")(
            [0.9 - 0.12 * i for i in range(len(trajectories))]
        )
        for i, ((tol, k), pts) in enumerate(sorted(trajectories.items())):
            if not pts:
                continue
            sweeps, energies = zip(*pts)
            all_energies.extend(energies)
            ax.plot(sweeps, energies, "-o", color=patched_colors[i], markersize=3,
                    linewidth=1.2, alpha=0.85,
                    label=f"patched tol={tol:.0e} k={k} (patched-ascend start)")

    for fname, label, color, ls in SOURCES:
        if label is None:
            continue
        path = os.path.join(LOG_DIR, fname)
        if not os.path.exists(path):
            continue
        pts = parse_single_trajectory(path)
        if not pts:
            continue
        sweeps, energies = zip(*pts)
        all_energies.extend(energies)
        ax.plot(sweeps, energies, ls, color=color, marker="s", markersize=4,
                linewidth=1.8, label=label)

    if all_energies:
        best = min(all_energies)
        ax.axhline(best, color="black", linestyle=":", linewidth=1.2,
                   label=f"lowest energy observed ({best:.6f})")

    ax.set_xlabel("sweep number (within the chi=27 branch)")
    ax.set_ylabel("energy E")
    ax.set_title(
        "g=-0.700, chi=27: all branch-test trajectories overlaid\n"
        "(no exact reference at this system size -- lowest observed E is the "
        "best known answer, not an assumed target)"
    )
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = f"{OUT_DIR}/g0.7_chi27_trajectories.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
