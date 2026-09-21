"""
pipeline/plot_final_vs_g.py
------------------------------
Stage 2 of the pipeline: reads the FINAL (last row = last-visited stage,
i.e. the descend-phase value for chi < chis[-1]) energy and entanglement
entropy from each (g, chi)'s basic.csv -- written by pipeline/g_sweep.py
(or the legacy sweep.py) -- and plots E(g) and S(g), one line per chi.

Usage:
  python pipeline/plot_final_vs_g.py --g-min 0.1 --g-max 1.5 --n-g 15 --chis 9 18 27
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.g_sweep import get_folder, DEFAULT_DRIVE_PATH, DEVICE_DRIVE_PATHS, DEFAULT_DEVICE, drive_path_for_device
from pipeline.plot_convergence_vs_g import _last_convergence_row
from Z3_funcs.utils import get_chi_energies_from_couplings, get_chi_entropies_from_couplings


def plot_final_vs_g(
    g_raw_values, chis, Lx=5, Ly=5, shape="parallelogram",
    bound_state=None, chargesx=None, chargesy=None, R=None,
    precision=3, drive_path=DEFAULT_DRIVE_PATH, n_links=None,
    out_dir=None,
):
    """g_raw_values: positive |g| values (actual g = -g_raw), matching
    Z3_funcs.utils' get_chi_energies_from_couplings convention."""
    folder, _, _ = get_folder(drive_path, Lx, Ly, shape, bound_state, chargesx, chargesy, R)
    g_raw_values = np.asarray(sorted(g_raw_values))
    g_actual = -g_raw_values

    energies = {chi: np.asarray(get_chi_energies_from_couplings(folder, chi, g_raw_values, precision=precision)) for chi in chis}
    entropies = {chi: np.asarray(get_chi_entropies_from_couplings(folder, chi, g_raw_values, precision=precision)) for chi in chis}

    # Flag g-points where chi_max (chis[-1], the hard-converged stage) never
    # actually hit converged_this_sweep=True within its sweep budget -- those
    # final energy/entropy VALUES are just whatever the last sweep landed on,
    # not a value the run itself vouches for as converged.
    chi_max = chis[-1]
    chi_max_not_converged = np.array([
        _last_convergence_row(folder, g, chi_max, precision)["converged_this_sweep"].strip() != "True"
        for g in g_actual
    ])
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
        ax.plot(g_actual, energies[chi], "-o", color=colors(i), markersize=5, label=f"chi={chi}")
    _mark_not_converged(ax, energies)
    ax.set_xlabel("g")
    ax.set_ylabel("final ground-state energy E")
    ax.set_title(f"E(g) vs chi -- {Lx}x{Ly} {shape}, bound_state={bound_state}\n{conv_note}")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_e = f"{out_dir}/final_energy_vs_g.png"
    fig.savefig(out_e, dpi=200)
    plt.close(fig)
    print("saved:", out_e)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, chi in enumerate(chis):
        ax.plot(g_actual, entropies[chi], "-o", color=colors(i), markersize=5, label=f"chi={chi}")
    _mark_not_converged(ax, entropies)
    ax.set_xlabel("g")
    ax.set_ylabel("final entanglement entropy S (reference bond)")
    ax.set_title(f"S(g) vs chi -- {Lx}x{Ly} {shape}, bound_state={bound_state}\n{conv_note}")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_s = f"{out_dir}/final_entropy_vs_g.png"
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
    args = p.parse_args()

    drive_path = args.drive_path or drive_path_for_device(args.device)
    g_raw_values = np.linspace(args.g_min, args.g_max, args.n_g)
    plot_final_vs_g(
        g_raw_values, args.chis, Lx=args.lx, Ly=args.ly, shape=args.shape,
        precision=args.precision, drive_path=drive_path, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
