"""
pipeline/make_field_entropy_visuals.py
------------------------------------------
Stage 4 of the pipeline: at a FIXED chi, renders electric-field and/or
entanglement-entropy visuals across a g-scan -- either animated GIFs (one
frame per g) or individual single-frame plots at specific g values.
Thin wrapper around Z3_funcs.visualize_plaquettes' existing make_gif /
make_links_gif / plot_plaquettes / plot_lattice_links.

Note: make_links_gif / plot_lattice_links need each g's tensors.hdf5
checkpoint (gauge_tensor) already saved at the requested chi -- run
pipeline/g_sweep.py first. Entropy-only views (make_gif / plot_plaquettes)
only need basic.csv, which every g_sweep run already writes.

Usage (GIFs across the whole scan):
  python pipeline/make_field_entropy_visuals.py gif --g-min 0.1 --g-max 1.5 --n-g 15 \
      --chi 9 --fields entropy electric --color-by abs

Usage (single frames at specific g values):
  python pipeline/make_field_entropy_visuals.py frames --g-values 0.1 0.7 1.5 \
      --chi 9 --fields entropy electric --color-by abs real arg
"""
import argparse

from pipeline.g_sweep import DEVICE_DRIVE_PATHS, DEFAULT_DEVICE
from Z3_funcs.visualize_plaquettes import make_gif, make_links_gif, plot_plaquettes, plot_lattice_links


def make_gifs(g_actual_values, chi, Lx=5, Ly=5, shape="parallelogram",
              bound_state=None, R=None, precision=3, device=DEFAULT_DEVICE,
              fields=("entropy", "electric"), color_by=("abs",)):
    out_paths = []
    if "entropy" in fields:
        out_paths.append(make_gif(
            g_values=list(g_actual_values), Lx=Lx, Ly=Ly, shape=shape, bound_state=bound_state,
            R=R, chi=chi, device=device, precision=precision,
        ))
    if "electric" in fields:
        for cb in color_by:
            out_paths.append(make_links_gif(
                g_values=list(g_actual_values), Lx=Lx, Ly=Ly, shape=shape, bound_state=bound_state,
                R=R, chi=chi, device=device, precision=precision, color_by=cb,
            ))
    return out_paths


def make_frames(g_actual_values, chi, Lx=5, Ly=5, shape="parallelogram",
                 bound_state=None, R=None, precision=3, device=DEFAULT_DEVICE,
                 fields=("entropy", "electric"), color_by=("abs",)):
    for g in g_actual_values:
        if "entropy" in fields:
            plot_plaquettes(
                Lx=Lx, Ly=Ly, shape=shape, bound_state=bound_state, R=R,
                g=g, chi=chi, device=device, precision=precision,
            )
            print(f"entropy frame done: g={g}")
        if "electric" in fields:
            for cb in color_by:
                plot_lattice_links(
                    Lx=Lx, Ly=Ly, shape=shape, bound_state=bound_state, R=R,
                    g=g, chi=chi, device=device, precision=precision, color_by=cb,
                )
                print(f"electric field frame ({cb}) done: g={g}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    p_gif = sub.add_parser("gif")
    p_gif.add_argument("--g-min", type=float, required=True)
    p_gif.add_argument("--g-max", type=float, required=True)
    p_gif.add_argument("--n-g", type=int, required=True)
    p_gif.add_argument("--chi", type=int, required=True)
    p_gif.add_argument("--fields", nargs="+", choices=["entropy", "electric"], default=["entropy", "electric"])
    p_gif.add_argument("--color-by", nargs="+", choices=["abs", "real", "arg"], default=["abs"])
    p_gif.add_argument("--lx", type=int, default=5)
    p_gif.add_argument("--ly", type=int, default=5)
    p_gif.add_argument("--shape", default="parallelogram")
    p_gif.add_argument("--precision", type=int, default=3)
    p_gif.add_argument("--device", choices=list(DEVICE_DRIVE_PATHS), default=DEFAULT_DEVICE)

    p_frames = sub.add_parser("frames")
    p_frames.add_argument("--g-values", type=float, nargs="+", required=True,
                           help="actual g values (e.g. -0.1 -0.7 -1.5)")
    p_frames.add_argument("--chi", type=int, required=True)
    p_frames.add_argument("--fields", nargs="+", choices=["entropy", "electric"], default=["entropy", "electric"])
    p_frames.add_argument("--color-by", nargs="+", choices=["abs", "real", "arg"], default=["abs"])
    p_frames.add_argument("--lx", type=int, default=5)
    p_frames.add_argument("--ly", type=int, default=5)
    p_frames.add_argument("--shape", default="parallelogram")
    p_frames.add_argument("--precision", type=int, default=3)
    p_frames.add_argument("--device", choices=list(DEVICE_DRIVE_PATHS), default=DEFAULT_DEVICE)

    args = p.parse_args()

    if args.mode == "gif":
        import numpy as np
        g_raw = np.linspace(args.g_min, args.g_max, args.n_g)
        g_actual = [-float(g) for g in g_raw]
        out = make_gifs(g_actual, args.chi, Lx=args.lx, Ly=args.ly, shape=args.shape,
                         precision=args.precision, device=args.device,
                         fields=args.fields, color_by=args.color_by)
        for o in out:
            print("saved:", o)
    else:
        make_frames(args.g_values, args.chi, Lx=args.lx, Ly=args.ly, shape=args.shape,
                     precision=args.precision, device=args.device,
                     fields=args.fields, color_by=args.color_by)


if __name__ == "__main__":
    main()
