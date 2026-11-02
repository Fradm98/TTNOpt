"""
pipeline/plot_entropy_sectors_vs_g.py
-------------------------------------
Preliminary comparison of the reference-bond entanglement entropy S(g)
across charge sectors (vacuum + every 3-q configuration found), using
whatever points exist on disk -- missing g-points are skipped, not errors.

basic.csv is only written at the END of a stage, so a g-point whose
chi_max stage is still running only has its ascend-stage chis (9/27/50).
For each (config, g) this takes the LARGEST chi that has a basic.csv, and
the marker shows which: circle = chi_max, square/triangle/diamond = a
smaller chi. Hollow markers = the g-point has no ALL_STAGES_DONE yet
(still running or cut off), so its value is provisional.

Usage (on the cluster, from the TTNOpt root):
  python -m pipeline.plot_entropy_sectors_vs_g --lx 5 --ly 9 --shape hexagon \
      --drive-path /home/fradm/projects/5_Z3
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MARKERS = {81: "o", 50: "s", 27: "^", 9: "D"}


def _best_point(g_folder, chis):
    """(chi, S) from the largest chi with a basic.csv, or None."""
    for chi in sorted(chis, reverse=True):
        f = f"{g_folder}/run_chi-{chi}/basic.csv"
        if os.path.exists(f):
            return chi, float(pd.read_csv(f)["entanglement"].iloc[-1])
    return None


def collect(config_folder, chis):
    rows = []
    for g_folder in glob.glob(f"{config_folder}/g_*"):
        m = re.search(r"g_(-?[\d.]+)$", g_folder)
        if not m:
            continue  # e.g. g_-1.500_stuck
        best = _best_point(g_folder, chis)
        if best is None:
            continue
        rows.append({
            "g": float(m.group(1)), "chi": best[0], "S": best[1],
            "done": os.path.exists(f"{g_folder}/ALL_STAGES_DONE"),
        })
    return pd.DataFrame(rows, columns=["g", "chi", "S", "done"]).sort_values("g", ascending=False)


def plot(Lx, Ly, shape, drive_path, chis, out_dir=None):
    root = f"{drive_path}/shape_{shape}/_Lx{Lx}_Ly{Ly}"
    configs = [("vacuum", f"{root}/runs_vacuum")] + [
        (os.path.basename(c), c) for c in sorted(glob.glob(f"{root}/runs_3-q/*"))
    ]
    data = {name: collect(folder, chis) for name, folder in configs}
    data = {k: v for k, v in data.items() if len(v)}
    chi_max = max(chis)

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.get_cmap("tab10")
    vac = data.get("vacuum")
    vac_S = dict(zip(vac["g"], vac["S"])) if vac is not None else {}

    for i, (name, df) in enumerate(data.items()):
        c = colors(i)
        ax_s.plot(df["g"], df["S"], "-", color=c, alpha=0.6, label=name)
        for _, r in df.iterrows():
            ax_s.plot(r["g"], r["S"], MARKERS.get(r["chi"], "x"), color=c, markersize=8,
                      markerfacecolor=c if r["done"] else "none", markeredgewidth=1.8)
        if name != "vacuum" and vac_S:
            d = df[df["g"].isin(vac_S.keys())]
            if len(d):
                dS = d["S"].values - np.array([vac_S[g] for g in d["g"]])
                ax_d.plot(d["g"], dS, "-", color=c, alpha=0.6, label=name)
                for (_, r), y in zip(d.iterrows(), dS):
                    ax_d.plot(r["g"], y, MARKERS.get(r["chi"], "x"), color=c, markersize=8,
                              markerfacecolor=c if r["done"] else "none", markeredgewidth=1.8)

    # legend entries for the marker conventions
    for chi in sorted(chis, reverse=True):
        ax_s.plot([], [], MARKERS.get(chi, "x"), color="gray", label=f"best chi available = {chi}")
    ax_s.plot([], [], "o", color="gray", markerfacecolor="none", label="not finished (provisional)")

    ax_s.set_ylabel("final entanglement entropy S (reference bond)")
    ax_s.set_title(f"S(g) by sector -- {Lx}x{Ly} {shape} (preliminary)")
    ax_d.axhline(0, color="k", lw=0.8)
    ax_d.set_ylabel("S(config) - S(vacuum)  [same g]")
    ax_d.set_title("entropy excess over vacuum")
    for ax in (ax_s, ax_d):
        ax.set_xlabel("g")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()

    out_dir = out_dir or f"{drive_path}/figures"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/entropy_sectors_vs_g_chi{chi_max}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved:", out)

    for name, df in data.items():
        print(f"\n{name}")
        print(df.to_string(index=False))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lx", type=int, default=5)
    p.add_argument("--ly", type=int, default=9)
    p.add_argument("--shape", default="hexagon")
    p.add_argument("--chis", type=int, nargs="+", default=[9, 27, 50, 81])
    p.add_argument("--drive-path", default="/home/fradm/projects/5_Z3")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    plot(args.lx, args.ly, args.shape, args.drive_path, args.chis, args.out_dir)


if __name__ == "__main__":
    main()
