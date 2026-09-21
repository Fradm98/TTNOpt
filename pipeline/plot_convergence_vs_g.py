"""
pipeline/plot_convergence_vs_g.py
------------------------------------
Stage 3 of the pipeline: reads the LAST row of each (g, chi)'s
convergence.csv (written by pipeline/g_sweep.py / sweep.py / gss) --
max_energy_reldiff and max_ee_diff, the actual sweep-to-sweep stopping
criteria (see GroundStateSearch.run(), NOT the single reference-edge-only
trace) -- and plots both vs g, one line per chi. This is the "how
reliable is this scan" report: a single, isolated peak at some g means
one genuinely slow (usually near-critical) point, not a broken scan; a
peak that grows with chi instead of shrinking would be the real red flag.

Usage:
  python pipeline/plot_convergence_vs_g.py --g-min 0.1 --g-max 1.5 --n-g 15 --chis 9 18 27
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.g_sweep import get_folder, DEFAULT_DRIVE_PATH, DEVICE_DRIVE_PATHS, DEFAULT_DEVICE, drive_path_for_device


def _last_convergence_row(folder, g, chi, precision):
    path = f"{folder}/g_{g:.{precision}f}/run_chi-{chi}/convergence.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def plot_convergence_vs_g(
    g_raw_values, chis, Lx=5, Ly=5, shape="parallelogram",
    bound_state=None, chargesx=None, chargesy=None, R=None,
    precision=3, drive_path=DEFAULT_DRIVE_PATH, out_dir=None,
    energy_convergence_threshold=1e-10, entanglement_convergence_threshold=1e-10,
):
    """energy_convergence_threshold/entanglement_convergence_threshold: the
    numerics.*_convergence_threshold values the run itself was checked
    against (see pipeline.g_sweep.run_g_sweep -- both default to 1e-10
    there). Drawn as a horizontal reference line so it's visible at a
    glance how far above/below the actual convergence criterion each point
    landed, not just its raw magnitude."""
    folder, _, _ = get_folder(drive_path, Lx, Ly, shape, bound_state, chargesx, chargesy, R)
    g_raw_values = np.asarray(sorted(g_raw_values))
    g_actual = -g_raw_values

    energy_conv = {chi: [] for chi in chis}
    ee_conv = {chi: [] for chi in chis}
    chi_max = chis[-1]
    chi_max_not_converged = []
    for chi in chis:
        for g in g_actual:
            last = _last_convergence_row(folder, g, chi, precision)
            energy_conv[chi].append(float(last["max_energy_reldiff"]))
            ee_conv[chi].append(float(last["max_ee_diff"]))
            if chi == chi_max:
                chi_max_not_converged.append(last["converged_this_sweep"].strip() != "True")
    energy_conv = {chi: np.asarray(v) for chi, v in energy_conv.items()}
    ee_conv = {chi: np.asarray(v) for chi, v in ee_conv.items()}
    chi_max_not_converged = np.asarray(chi_max_not_converged)

    n_not_conv = int(chi_max_not_converged.sum())
    if n_not_conv:
        bad_g = ", ".join(f"{g:.{precision}f}" for g in g_actual[chi_max_not_converged])
        print(f"chi_max={chi_max}: {n_not_conv}/{len(g_actual)} g-point(s) NOT converged: {bad_g}")
    else:
        print(f"chi_max={chi_max}: all {len(g_actual)} g-point(s) converged.")

    out_dir = out_dir or f"{drive_path}/figures"
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.get_cmap("tab10")
    conv_note = (
        f"{n_not_conv}/{len(g_actual)} chi_max={chi_max} pts NOT converged"
        if n_not_conv else f"chi_max={chi_max}: all converged"
    )

    def _mark_not_converged(ax, values):
        if n_not_conv:
            ax.plot(
                g_actual[chi_max_not_converged], values[chi_max][chi_max_not_converged],
                "o", markersize=13, markerfacecolor="none", markeredgecolor="red",
                markeredgewidth=2, linestyle="none",
                label=f"chi_max={chi_max} not converged", zorder=5,
            )

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, chi in enumerate(chis):
        ax.plot(g_actual, energy_conv[chi], "-o", color=colors(i), markersize=5, label=f"chi={chi}")
    _mark_not_converged(ax, energy_conv)
    ax.axhline(
        energy_convergence_threshold, color="black", linestyle="--", linewidth=1.2,
        label=f"energy_convergence_threshold={energy_convergence_threshold:.0e}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("g")
    ax.set_ylabel("max_energy_reldiff (last sweep)")
    ax.set_title(f"Energy convergence reached vs g -- {Lx}x{Ly} {shape}\n{conv_note}")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_e = f"{out_dir}/energy_convergence_vs_g.png"
    fig.savefig(out_e, dpi=200)
    plt.close(fig)
    print("saved:", out_e)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, chi in enumerate(chis):
        ax.plot(g_actual, ee_conv[chi], "-o", color=colors(i), markersize=5, label=f"chi={chi}")
    _mark_not_converged(ax, ee_conv)
    ax.axhline(
        entanglement_convergence_threshold, color="black", linestyle="--", linewidth=1.2,
        label=f"entanglement_convergence_threshold={entanglement_convergence_threshold:.0e}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("g")
    ax.set_ylabel("max_ee_diff (last sweep)")
    ax.set_title(f"Entropy convergence reached vs g -- {Lx}x{Ly} {shape}\n{conv_note}")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_s = f"{out_dir}/entropy_convergence_vs_g.png"
    fig.savefig(out_s, dpi=200)
    plt.close(fig)
    print("saved:", out_s)

    return out_e, out_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g-min", type=float, required=True)
    p.add_argument("--g-max", type=float, required=True)
    p.add_argument("--n-g", type=int, required=True)
    p.add_argument("--chis", type=int, nargs="+", default=[9, 18, 27])
    p.add_argument("--lx", type=int, default=5)
    p.add_argument("--ly", type=int, default=5)
    p.add_argument("--shape", default="parallelogram")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--device", choices=list(DEVICE_DRIVE_PATHS), default=DEFAULT_DEVICE)
    p.add_argument("--drive-path", default=None, help="override the path derived from --device")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--energy-convergence-threshold", type=float, default=1e-10,
                    help="reference line -- must match the run's numerics.energy_convergence_threshold")
    p.add_argument("--entanglement-convergence-threshold", type=float, default=1e-10,
                    help="reference line -- must match the run's numerics.entanglement_convergence_threshold")
    args = p.parse_args()

    drive_path = args.drive_path or drive_path_for_device(args.device)
    g_raw_values = np.linspace(args.g_min, args.g_max, args.n_g)
    plot_convergence_vs_g(
        g_raw_values, args.chis, Lx=args.lx, Ly=args.ly, shape=args.shape,
        precision=args.precision, drive_path=drive_path, out_dir=args.out_dir,
        energy_convergence_threshold=args.energy_convergence_threshold,
        entanglement_convergence_threshold=args.entanglement_convergence_threshold,
    )


if __name__ == "__main__":
    main()
