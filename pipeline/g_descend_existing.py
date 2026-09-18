"""
pipeline/g_descend_existing.py
---------------------------------
For g-points that already have a converged chi=50 checkpoint (e.g. from
the warm-start fine scan), truncate DOWN through chi=27,18,9 at each g
(same g, not a neighboring one) via the same lossless
move_canonical_center-based descend mechanism sweep.py/g_sweep.py use --
without redoing the expensive chi=50 convergence itself. Writes into the
SAME drive_path/folder so basic.csv/convergence.csv for chi=27/18/9
appear right alongside the existing chi=50 ones, ready for
plot_final_vs_g.py / plot_convergence_vs_g.py with --chis 9 18 27 50.

sweeps_descend defaults to 10 (not 3): with only 3 resettling sweeps per
stage, descending 50->27->18->9 can land badly under-converged (chi=27
max_ee_diff stuck ~5e-5 vs. ~1e-9 for a proper 10-sweep hard-converge at
that chi) despite looking "stable" sweep-to-sweep. With 10 sweeps per
stage instead, the descended values come out MORE converged than
independently hard-converging each chi from scratch (chi=27 ~1e-12,
chi=18 ~1e-9 -- both beating an ascend/converge/descend-built reference
by 2-6 orders of magnitude), confirming sweep.py's own design philosophy
(truncate down from the best available state) -- as long as the descend
budget is actually large enough to re-equilibrate after the cut.

Usage:
  python pipeline/g_descend_existing.py --g-values -0.60 -0.61 ... -0.80 \
      --source-chi 50 --descend-chis 27 18 9 --sweeps-descend 10 --device mac
"""
import argparse
import os
import subprocess

import yaml

from pipeline.g_sweep import get_folder, DEVICE_DRIVE_PATHS, DEFAULT_DEVICE, drive_path_for_device
from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs


def descend_existing(
    g_values, source_chi, descend_chis, sweeps_descend=10,
    Lx=5, Ly=5, shape="parallelogram", bound_state=None, chargesx=None, chargesy=None, R=None,
    precision=3, device=DEFAULT_DEVICE, drive_path=None, unpatched=True,
):
    drive_path = drive_path or drive_path_for_device(device)
    N = nplaqs(Lx, Ly, shape)
    folder, chargesx, chargesy = get_folder(drive_path, Lx, Ly, shape, bound_state, chargesx, chargesy, R)
    init_tensor_file = f"{drive_path}/tensors.hdf5"

    print(f"descending {list(descend_chis)} from chi={source_chi} for {len(g_values)} g-points, "
          f"drive_path={drive_path}", flush=True)

    failed = []
    for g in g_values:
        run_folder = f"{folder}/g_{g:.{precision}f}"
        filename = os.path.join(run_folder, "EF_Z.dat")
        create_ids_coeffs_file(Lx, Ly, shape, g, bound_state, chargesx=chargesx, chargesy=chargesy, filename=filename)

        numerics_dict = {
            "opt_structure": {"type": 0},
            "initial_bond_dimension": source_chi,
            "energy_convergence_threshold": 1e-5,
            "entanglement_convergence_threshold": 1e-10,
            "energy_degeneracy_threshold": 1e-13,
            "entanglement_degeneracy_threshold": 1e-8,
            "verbose_sweeps": True,
            "unpatched": unpatched,
            "init_tree": 3,
            "init_tensor_file": init_tensor_file,
            "init_g": g,
            "init_chi": source_chi,
            "max_bond_dimensions": list(descend_chis),
            "max_num_sweeps": [sweeps_descend] * len(descend_chis),
            "lanczos_tol": [None] * len(descend_chis),
            "lanczos_maxiter": [None] * len(descend_chis),
        }

        input_dict = {
            "system": {
                "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape,
                "bound_state": bound_state, "chargesx": chargesx, "chargesy": chargesy, "R": R,
                "model": {"type": "z3", "file": filename},
                "g": g, "MF_X": 1.0 / g, "EF_Z": filename, "precision": precision,
            },
            "numerics": numerics_dict,
            "output": {
                "dir": run_folder, "tensors": folder, "save_tensors": drive_path,
                "single_site": 0, "two_site": 0,
            },
        }

        inputfile = os.path.join(run_folder, "input_descend.yml")
        with open(inputfile, "w") as f:
            yaml.dump(input_dict, f, sort_keys=False)

        print(f"\n[g={g:.{precision}f}] descending {source_chi} -> {list(descend_chis)}", flush=True)
        try:
            subprocess.run(["gss", inputfile], check=True)
        except subprocess.CalledProcessError as e:
            # One point's numerical hiccup (e.g. a rare LAPACK SVD
            # non-convergence) shouldn't kill the other, unrelated
            # g-points -- log it and keep going instead of dying here.
            failed.append(g)
            print(f"[g={g:.{precision}f}] FAILED ({e}) -- continuing with the remaining g-values", flush=True)

    if failed:
        print(f"\n{len(failed)} g-value(s) failed: {[f'{g:.{precision}f}' for g in failed]}")
    print("DONE")
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g-values", type=float, nargs="+", required=True)
    p.add_argument("--source-chi", type=int, required=True)
    p.add_argument("--descend-chis", type=int, nargs="+", required=True)
    p.add_argument("--sweeps-descend", type=int, default=10)
    p.add_argument("--lx", type=int, default=5)
    p.add_argument("--ly", type=int, default=5)
    p.add_argument("--shape", default="parallelogram")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--unpatched", action="store_true", default=True)
    p.add_argument("--patched", dest="unpatched", action="store_false")
    p.add_argument("--device", choices=list(DEVICE_DRIVE_PATHS), default=DEFAULT_DEVICE)
    p.add_argument("--drive-path", default=None)
    args = p.parse_args()

    descend_existing(
        args.g_values, args.source_chi, args.descend_chis, sweeps_descend=args.sweeps_descend,
        Lx=args.lx, Ly=args.ly, shape=args.shape, precision=args.precision,
        device=args.device, drive_path=args.drive_path, unpatched=args.unpatched,
    )


if __name__ == "__main__":
    main()
