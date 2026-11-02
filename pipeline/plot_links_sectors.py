"""
pipeline/plot_links_sectors.py
---------------------------------
Link-field plots (color_by real and arg) for the 5x9 hexagon vacuum, on-axis
R=2 baryon and off-axis custom1 baryon, each at the largest chi with a saved
tensor. Points without a tensor are skipped; a failing point doesn't stop the
others.

Results are cached per (config, g, chi) in tensors.hdf5: pass
force_recompute=True to replot a g-point whose checkpoint has since changed.

Usage: qsub pipeline/plot_links_sectors.pbs (see its header)
"""
import matplotlib.pyplot as plt

from pipeline.g_sweep import drive_path_for_device
from Z3_funcs.hdf5_manager import tensor_exists
from Z3_funcs.visualize_plaquettes import plot_lattice_links

DEVICE = "presto"
LX, LY, SHAPE, PRECISION = 5, 9, "hexagon", 3
G_VALUES = (-0.1, -0.7, -1.5)
CHIS = (81, 50, 27, 9)  # largest first
COLOR_BY = ("real", "arg")
CONFIGS = [
    # name, bound_state, R, chargesx, chargesy
    ("vacuum", None, None, None, None),
    ("on-axis R2", "baryon", 2, [-2, 0, 2], [4, 6, 2]),
    ("off-axis custom1", "baryon", 1, [-1, -2, 3], [2, 7, 3]),
]


def main():
    ten_file = f"{drive_path_for_device(DEVICE)}/tensors.hdf5"
    for g in G_VALUES:
        for name, bound_state, R, xs, ys in CONFIGS:
            try:
                chi = next((c for c in CHIS if tensor_exists(
                    ten_file, SHAPE, LX, LY, bound_state, R, g, PRECISION, c, chargesx=xs, chargesy=ys,
                )), None)
                if chi is None:
                    print(f"SKIP {name} g={g}: no tensor", flush=True)
                    continue
                for color_by in COLOR_BY:
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
