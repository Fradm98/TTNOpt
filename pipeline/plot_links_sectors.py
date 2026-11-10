"""
pipeline/plot_links_sectors.py
---------------------------------
Link-field plots (color_by real, arg, flux and net) for the 5x9 hexagon vacuum, on-axis
R=2 baryon and off-axis custom1 baryon, each at the largest chi with a saved
tensor. Points without a tensor are skipped; a failing point doesn't stop the
others.

Results are cached per (config, g, chi) in tensors.hdf5: pass
force_recompute=True to replot a g-point whose checkpoint has since changed.

Each point's link values are saved as they are computed, so a run cut by
the walltime resumes where it stopped.

Usage: qsub pipeline/plot_links_sectors.pbs (see its header)
"""
import argparse

import matplotlib.pyplot as plt

from pipeline.g_sweep import drive_path_for_device
from Z3_funcs.hdf5_manager import tensor_exists
from Z3_funcs.visualize_plaquettes import plot_lattice_links

DEVICE = "presto"
LX, LY, SHAPE, PRECISION = 5, 9, "hexagon", 3
G_VALUES = (-0.1, -0.7, -1.5)  # g_sweep stores g negated
CHIS = (81, 50, 27, 9)  # largest first
COLOR_BY = ("real", "arg", "flux", "net")
CONFIGS = [
    # name, bound_state, R, chargesx, chargesy
    ("vacuum", None, None, None, None),
    ("on-axis R2", "baryon", 2, [-2, 0, 2], [4, 6, 2]),
    ("off-axis custom1", "baryon", 1, [-1, -2, 3], [2, 7, 3]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chis", type=int, nargs="+", default=list(CHIS),
                   help="chis to try, largest first (default: %(default)s)")
    p.add_argument("--g", type=float, nargs="+", default=None,
                   help="g-points to plot, sign ignored (default: 0.1 0.7 1.5)")
    p.add_argument("--color-by", nargs="+", default=list(COLOR_BY), choices=["abs", *COLOR_BY],
                   help="colour modes to draw (default: %(default)s)")
    args = p.parse_args()
    chis = sorted(args.chis, reverse=True)
    g_values = G_VALUES if args.g is None else tuple(-abs(g) for g in args.g)
    ten_file = f"{drive_path_for_device(DEVICE)}/tensors.hdf5"
    for g in g_values:
        for name, bound_state, R, xs, ys in CONFIGS:
            try:
                chi = next((c for c in chis if tensor_exists(
                    ten_file, SHAPE, LX, LY, bound_state, R, g, PRECISION, c, chargesx=xs, chargesy=ys,
                )), None)
                if chi is None:
                    print(f"SKIP {name} g={g}: no tensor", flush=True)
                    continue
                for color_by in args.color_by:
                    print(f"START {name} g={g} chi={chi} {color_by}", flush=True)
                    plot_lattice_links(
                        LX, LY, SHAPE, bound_state, R, g, chi, DEVICE,
                        xs=xs, ys=ys, precision=PRECISION, color_by=color_by,
                    )
                    plt.close("all")
            except Exception as e:
                print(f"FAILED {name} g={g}: {e!r}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
