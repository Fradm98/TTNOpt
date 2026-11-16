"""
pipeline/export_link_fields.py
------------------------------
Copy every cached whole-lattice link field <V_l> ("ef_links" next to each
tensor checkpoint in tensors.hdf5, written by get_lattice_electric_fields)
into plain CSV files, so they survive when save_tensor later deletes and
rewrites that checkpoint group.

One file per (config, g, chi), mirroring the hdf5 path:
  {out_dir}/{shape}/_Lx{Lx}_Ly{Ly}/{sector}/{config}/g_{g}/chi-{chi}.csv
with columns i, j, d, p_left, p_right, kind, re, im, abs, arg
(p_left/p_right = -1 for a boundary link). Read-only on tensors.hdf5.
Existing CSVs are overwritten. Pass --partial to also export unfinished
"ef_links_partial" caches (as chi-{chi}_partial.csv).

Usage (on the cluster, from the TTNOpt root):
  python -m pipeline.export_link_fields --drive-path /home/fradm/projects/5_Z3
"""
import argparse
import csv
import os
import re

import numpy as np

from Z3_funcs.hdf5_manager import _open_h5


def find_caches(f, names):
    found = []

    def visit(path, obj):
        if os.path.basename(path) in names and re.search(r"/g_[^/]+/run_chi-\d+/[^/]+$", path):
            found.append(path)
    f.visititems(visit)
    return sorted(found)


def write_csv(sub, out):
    ijd = sub["ijd"][:]
    p_left = sub["p_left"][:]
    p_right = sub["p_right"][:]
    values = sub["value"][:].view(np.complex128).reshape(-1)
    kinds = [k.decode() if isinstance(k, bytes) else k for k in sub["kind"][:]]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["i", "j", "d", "p_left", "p_right", "kind", "re", "im", "abs", "arg"])
        for (i, j, d), pl, pr, kind, v in zip(ijd, p_left, p_right, kinds, values):
            w.writerow([int(i), int(j), int(d), int(pl), int(pr), kind]
                      + [f"{x:.17g}" for x in (v.real, v.imag, abs(v), np.angle(v))])
    return len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive-path", default="/home/fradm/projects/5_Z3")
    p.add_argument("--out-dir", default=None, help="default: {drive-path}/links_csv")
    p.add_argument("--partial", action="store_true", help="also export ef_links_partial caches")
    args = p.parse_args()

    out_dir = args.out_dir or f"{args.drive_path}/links_csv"
    names = {"ef_links", "ef_links_partial"} if args.partial else {"ef_links"}
    with _open_h5(f"{args.drive_path}/tensors.hdf5", "r") as f:
        paths = find_caches(f, names)
        for path in paths:
            ckpt, name = path.rsplit("/", 1)
            base, chi = ckpt.rsplit("/run_chi-", 1)
            suffix = "_partial" if name == "ef_links_partial" else ""
            out = f"{out_dir}/{base}/chi-{int(chi)}{suffix}.csv"
            n = write_csv(f[path], out)
            print(f"{n:4d} links -> {out}", flush=True)
    print(f"DONE: {len(paths)} file(s)")


if __name__ == "__main__":
    main()
