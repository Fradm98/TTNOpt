"""
z3_plot_chiladder_truncfloor.py
----------------------------------
Plots max|E_after_truncation - E_lanczos| vs chi, for the chi=9,18,27,50
ladder trunc-floor test(s) (unpatched and/or patched) -- both the
dominant problem edges (57/58, the bond joining the last two plaquettes
at the final coarse-graining level) and the top/reference edge (60,
essentially float-noise-only) for contrast.

Usage: python diagnostics/z3_plot_chiladder_truncfloor.py [unpatched] [patched]
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WARMSTART_DIR = os.path.join(os.path.dirname(__file__), "z3_5x5_warmstart_results")
CHIS = [9, 18, 27, 50]
PROBLEM_EDGES = [57, 58]
TOP_EDGE = 60


def load_max_diffs(csv_path):
    per_chi = {chi: {} for chi in CHIS}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            chi = int(row["chi"])
            edge = int(row["edge_id"])
            d = abs(float(row["E_after_truncation"]) - float(row["E_lanczos"]))
            per_chi[chi][edge] = max(per_chi[chi].get(edge, 0.0), d)
    return per_chi


def series_for(per_chi, edges):
    ys = []
    for chi in CHIS:
        vals = [per_chi[chi][e] for e in edges if e in per_chi[chi]]
        ys.append(max(vals) if vals else float("nan"))
    return ys


def main():
    variants = sys.argv[1:] or ["unpatched"]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"unpatched": "#2b6cb0", "patched": "#c05621"}
    markers = {"unpatched": "o", "patched": "s"}

    for variant in variants:
        tag = f"{variant}_chiladder_9-18-27-50_truncfloor_g-1.000"
        csv_path = os.path.join(WARMSTART_DIR, f"{tag}.csv")
        if not os.path.exists(csv_path):
            print(f"skip {variant}: {csv_path} not found")
            continue
        per_chi = load_max_diffs(csv_path)
        problem_ys = series_for(per_chi, PROBLEM_EDGES)
        top_ys = series_for(per_chi, [TOP_EDGE])
        c = colors.get(variant, "#333333")
        ax.plot(CHIS, problem_ys, "-", marker=markers.get(variant, "o"), color=c, markersize=7,
                label=f"{variant}: edges 57/58 (problem bond)")
        ax.plot(CHIS, top_ys, "--", marker=markers.get(variant, "o"), color=c, markersize=7, alpha=0.6,
                label=f"{variant}: edge 60 (top/reference, float noise)")
        for chi, y in zip(CHIS, problem_ys):
            print(f"  [{variant}] chi={chi}: edges57/58 max diff = {y:.3e}")
        for chi, y in zip(CHIS, top_ys):
            print(f"  [{variant}] chi={chi}: edge60 max diff = {y:.3e}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(CHIS)
    ax.set_xticklabels([str(c) for c in CHIS])
    ax.set_xlabel("chi")
    ax.set_ylabel("max |E_after_truncation - E_lanczos|")
    ax.set_title("Truncation cost vs chi -- g=-1.0, 5x5, fixed tree topology\n"
                  "problem bond (57/58) vs top/reference edge (60)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(WARMSTART_DIR, f"chiladder_truncfloor_vs_chi_{'_'.join(variants)}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
