"""
compare_patched_vs_unpatched.py
----------------------------------
Direct energy/entropy comparison between two already-completed g-sweeps
built with pipeline.g_sweep (one --unpatched, one --patched), reading their
basic.csv files via the same Z3_funcs.utils helpers plot_final_vs_g.py uses.

Three panels:
  1. S(g): unpatched (solid) vs patched (dashed), one color per chi.
  2. E(g): same style.
  3. |dS(g)| and |dE(g)| (patched - unpatched), log scale, one color per
     chi -- shows WHERE the two engines disagree and whether the gap
     shrinks with chi (a convergence-quality signature) or not.

Usage:
  python diagnostics/compare_patched_vs_unpatched.py \
      --drive-path-unpatched /Users/fradm/Desktop/projects/5_Z3 \
      --drive-path-patched /Users/fradm/Desktop/projects/5_Z3_patched_5x5 \
      --g-min 0.1 --g-max 1.5 --n-g 15 --chis 9 18 27 \
      --out-dir /Users/fradm/Desktop/projects/5_Z3_patched_5x5/figures
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.g_sweep import get_folder
from Z3_funcs.utils import get_chi_energies_from_couplings, get_chi_entropies_from_couplings


def compare(
    g_raw_values, chis, drive_path_unpatched, drive_path_patched,
    Lx=5, Ly=5, shape="parallelogram", precision=3, out_dir=None,
):
    g_raw_values = np.asarray(sorted(g_raw_values))
    g_actual = -g_raw_values

    folder_u, _, _ = get_folder(drive_path_unpatched, Lx, Ly, shape, None, None, None, None)
    folder_p, _, _ = get_folder(drive_path_patched, Lx, Ly, shape, None, None, None, None)

    E_u, E_p, S_u, S_p = {}, {}, {}, {}
    for chi in chis:
        E_u[chi] = get_chi_energies_from_couplings(folder_u, chi, g_raw_values, precision=precision)
        E_p[chi] = get_chi_energies_from_couplings(folder_p, chi, g_raw_values, precision=precision)
        S_u[chi] = get_chi_entropies_from_couplings(folder_u, chi, g_raw_values, precision=precision)
        S_p[chi] = get_chi_entropies_from_couplings(folder_p, chi, g_raw_values, precision=precision)

    out_dir = out_dir or f"{drive_path_patched}/figures"
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.get_cmap("tab10")

    # ---- Panel 1: entropy overlay ----
    # Unpatched/patched now agree almost everywhere (see the discrepancy
    # panel below), so a same-size solid-vs-dashed pair at the same color
    # just reads as one line. Instead: a big hollow marker for unpatched
    # with a small filled marker of a DIFFERENT shape for patched, both
    # opaque -- when they coincide you see the small shape sitting inside
    # the big ring, proving the overlap rather than hiding it.
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, chi in enumerate(chis):
        ax.plot(g_actual, S_u[chi], "-", color=colors(i), linewidth=1.2,
                 marker="o", markersize=11, markerfacecolor="none", markeredgewidth=1.6,
                 label=f"chi={chi} unpatched")
        ax.plot(g_actual, S_p[chi], "-", color=colors(i), linewidth=0,
                 marker="x", markersize=6, markeredgewidth=2,
                 label=f"chi={chi} patched")
    ax.set_xlabel("g")
    ax.set_ylabel("final entanglement entropy S (reference bond)")
    ax.set_title(f"S(g): unpatched (large hollow O) vs patched (small X) -- {Lx}x{Ly} {shape}")
    ax.invert_xaxis()
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out1 = f"{out_dir}/compare_entropy_vs_g.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    print("saved:", out1)

    # ---- Panel 2: energy overlay ----
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, chi in enumerate(chis):
        ax.plot(g_actual, E_u[chi], "-", color=colors(i), linewidth=1.2,
                 marker="o", markersize=11, markerfacecolor="none", markeredgewidth=1.6,
                 label=f"chi={chi} unpatched")
        ax.plot(g_actual, E_p[chi], "-", color=colors(i), linewidth=0,
                 marker="x", markersize=6, markeredgewidth=2,
                 label=f"chi={chi} patched")
    ax.set_xlabel("g")
    ax.set_ylabel("final ground-state energy E")
    ax.set_title(f"E(g): unpatched (large hollow O) vs patched (small X) -- {Lx}x{Ly} {shape}")
    ax.invert_xaxis()
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = f"{out_dir}/compare_energy_vs_g.png"
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    print("saved:", out2)

    # ---- Panel 3: |dS| and |dE| vs g, log scale, one color per chi ----
    fig, (ax_s, ax_e) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for i, chi in enumerate(chis):
        dS = np.abs(S_p[chi] - S_u[chi])
        dE = np.abs(E_p[chi] - E_u[chi])
        ax_s.semilogy(g_actual, np.maximum(dS, 1e-16), "-o", color=colors(i), markersize=4, label=f"chi={chi}")
        ax_e.semilogy(g_actual, np.maximum(dE, 1e-16), "-o", color=colors(i), markersize=4, label=f"chi={chi}")

    ax_s.set_ylabel("|S_patched - S_unpatched|")
    ax_s.set_title(f"Patched vs unpatched discrepancy -- {Lx}x{Ly} {shape}\n"
                    f"(energy is ALWAYS higher/worse for patched -- see sign note below)")
    ax_s.grid(alpha=0.3, which="both")
    ax_s.legend(fontsize=9)

    ax_e.set_xlabel("g")
    ax_e.set_ylabel("|E_patched - E_unpatched|")
    ax_e.grid(alpha=0.3, which="both")
    ax_e.invert_xaxis()
    ax_e.legend(fontsize=9)

    # Annotate the one-directional energy bias directly on the plot -- this
    # is the key evidence that the gap is a convergence/eigensolver effect,
    # not two genuinely different (possibly degenerate) physical states:
    # a variational method can only ever OVER-estimate the true ground
    # energy, so a bias that never changes sign points at one engine
    # systematically settling for a slightly less-optimized state.
    all_positive = all(np.all(E_p[chi] - E_u[chi] >= 0) for chi in chis)
    sign_note = (
        "E_patched - E_unpatched >= 0 at every single (g, chi) point -- "
        "consistent with the patched eigensolver landing on a slightly "
        "less-converged state, not a different physical state"
        if all_positive else
        "sign of E_patched - E_unpatched is NOT uniform -- re-examine"
    )
    fig.text(0.5, 0.005, sign_note, ha="center", fontsize=8, style="italic", wrap=True)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out3 = f"{out_dir}/compare_discrepancy_vs_g.png"
    fig.savefig(out3, dpi=200)
    plt.close(fig)
    print("saved:", out3)

    return out1, out2, out3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-path-unpatched", required=True)
    p.add_argument("--drive-path-patched", required=True)
    p.add_argument("--g-min", type=float, required=True)
    p.add_argument("--g-max", type=float, required=True)
    p.add_argument("--n-g", type=int, required=True)
    p.add_argument("--chis", type=int, nargs="+", default=[9, 18, 27])
    p.add_argument("--lx", type=int, default=5)
    p.add_argument("--ly", type=int, default=5)
    p.add_argument("--shape", default="parallelogram")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    g_raw_values = np.linspace(args.g_min, args.g_max, args.n_g)
    compare(
        g_raw_values, args.chis, args.drive_path_unpatched, args.drive_path_patched,
        Lx=args.lx, Ly=args.ly, shape=args.shape, precision=args.precision, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
