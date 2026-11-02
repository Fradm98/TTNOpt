"""
pipeline/plot_entropy_chis_vs_g.py
----------------------------------
Reference-bond entropy S(g) at every chi, for the vacuum and the on-axis and
off-axis baryons: one color per sector, one shade per chi (darkest = largest).
Right panel: excess over the vacuum at the same g and chi.

Also prints what exists for each point and what is still missing:
  ok    = last convergence.csv row has converged_this_sweep=True
  prov  = basic.csv, but not converged (ALL_STAGES_DONE can't tell: the
          provisional same-chi reload touches it too)
  old   = chi<chi_max only: basic.csv older than chi_max's basic.csv and
          input_descend.yml, so a cheap ascend-stage value (or chi_max
          changed after the descent)
  --    = no basic.csv
On the plot, filled = ok; hollow = prov or old. Pass --list-only to
print the table without plotting.

Usage (on the cluster, from the TTNOpt root):
  python -m pipeline.plot_entropy_chis_vs_g --lx 5 --ly 9 --shape hexagon \
      --drive-path /home/fradm/projects/5_Z3
"""
import argparse
import glob
import os
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# name, folder relative to the shape root, colormap
SECTORS = [
    ("vacuum", "runs_vacuum", "Greys"),
    ("on-axis R2", "runs_3-q/x-2-0-2_y4-6-2", "Greens"),
    ("off-axis custom1", "runs_3-q/x-1--2-3_y2-7-3", "Oranges"),
]


def _converged(run_folder):
    f = f"{run_folder}/convergence.csv"
    if not os.path.exists(f):
        return False
    rows = pd.read_csv(f)
    return len(rows) > 0 and str(rows["converged_this_sweep"].iloc[-1]).strip() == "True"


def collect(config_folder, chis):
    """One row per (g, chi) with a basic.csv, plus the status of each chi."""
    chi_max = max(chis)
    rows = []
    for g_folder in glob.glob(f"{config_folder}/g_*"):
        m = re.search(r"g_(-?[\d.]+)$", g_folder)
        if not m:
            continue  # e.g. g_-1.500_stuck
        g = float(m.group(1))
        top = f"{g_folder}/run_chi-{chi_max}/basic.csv"
        # A descent writes input_descend.yml first, then the lower chis, so
        # this also works when chi_max has no basic.csv of its own.
        marks = [top, f"{g_folder}/input_descend.yml"]
        times = [os.path.getmtime(f) for f in marks if os.path.exists(f)]
        top_time = max(times) if times else None
        for chi in chis:
            f = f"{g_folder}/run_chi-{chi}/basic.csv"
            if not os.path.exists(f):
                rows.append({"g": g, "chi": chi, "S": float("nan"), "status": "--"})
                continue
            if chi != chi_max and (top_time is None or os.path.getmtime(f) <= top_time):
                status = "old"
            else:
                status = "ok" if _converged(f"{g_folder}/run_chi-{chi}") else "prov"
            S = float(pd.read_csv(f)["entanglement"].iloc[-1])
            rows.append({"g": g, "chi": chi, "S": S, "status": status})
    return pd.DataFrame(rows, columns=["g", "chi", "S", "status"])


def print_status(name, df, chis):
    print(f"\n{name}")
    if df.empty:
        print("  no g-points")
        return
    table = df.pivot(index="g", columns="chi", values="status").sort_index(ascending=False)
    print(table[sorted(chis, reverse=True)].to_string())


def plot(data, chis, title, out):
    chi_max = max(chis)
    shades = {chi: 0.35 + 0.6 * k / max(len(chis) - 1, 1) for k, chi in enumerate(sorted(chis))}
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(15, 6))
    vacuum = data.get("vacuum")

    for name, (df, cmap) in data.items():
        df = df.dropna(subset=["S"])
        for chi in sorted(chis, reverse=True):
            d = df[df["chi"] == chi].sort_values("g", ascending=False)
            if d.empty:
                continue
            color = plt.get_cmap(cmap)(shades[chi])
            _draw(ax_s, d["g"], d["S"], d["status"], color, f"{name}, chi={chi}")
            if vacuum is None or name == "vacuum":
                continue
            v = vacuum[0]
            v = v[v["chi"] == chi].dropna(subset=["S"]).set_index("g")["S"]
            d = d[d["g"].isin(v.index)]
            if len(d):
                _draw(ax_d, d["g"], d["S"].values - v[d["g"]].values, d["status"], color, f"{name}, chi={chi}")

    ax_s.plot([], [], "o", color="gray", markerfacecolor="none", label="not converged or not descended")
    ax_s.set_ylabel("final entanglement entropy S (reference bond)")
    ax_s.set_title(title)
    ax_d.axhline(0, color="k", lw=0.8)
    ax_d.set_ylabel("S(config) - S(vacuum)  [same g, same chi]")
    ax_d.set_title("entropy excess over vacuum")
    for ax in (ax_s, ax_d):
        ax.set_xlabel("g")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=len(data), loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("\nsaved:", out)


def _draw(ax, g, y, status, color, label):
    ax.plot(g, y, "-", color=color, label=label)
    ok = (status == "ok").values
    ax.plot(g[ok], y[ok], "o", color=color, markersize=6)
    ax.plot(g[~ok], y[~ok], "o", color=color, markersize=6, markerfacecolor="none", markeredgewidth=1.5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lx", type=int, default=5)
    p.add_argument("--ly", type=int, default=9)
    p.add_argument("--shape", default="hexagon")
    p.add_argument("--chis", type=int, nargs="+", default=[9, 27, 50, 81])
    p.add_argument("--drive-path", default="/home/fradm/projects/5_Z3")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--list-only", action="store_true", help="print the status table, no plot")
    args = p.parse_args()

    root = f"{args.drive_path}/shape_{args.shape}/_Lx{args.lx}_Ly{args.ly}"
    data = {}
    for name, rel, cmap in SECTORS:
        folder = f"{root}/{rel}"
        if not os.path.isdir(folder):
            print(f"WARNING: {name}: no folder {folder}")
            continue
        df = collect(folder, args.chis)
        print_status(name, df, args.chis)
        data[name] = (df, cmap)

    if args.list_only:
        return
    out_dir = args.out_dir or f"{args.drive_path}/figures"
    plot(data, args.chis, f"S(g) by sector and chi -- {args.lx}x{args.ly} {args.shape}",
         f"{out_dir}/entropy_chis_vs_g.png")


if __name__ == "__main__":
    main()
