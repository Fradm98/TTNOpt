"""
pipeline/rerun_nonconverged.py
---------------------------------
For g-points where chi_max's LAST convergence.csv row shows
converged_this_sweep=False, resume from that same (g, chi_max) checkpoint
and run MORE sweeps at the SAME chi -- no chi change, just topping up the
converge stage's sweep budget for the specific points that needed it,
instead of blanket-raising sweeps_converge for the whole g-scan.

Only touches chi_max = chis[-1]: lower chis are meant to be descended AFTER
chi_max is trustworthy (see pipeline.g_descend_existing) -- if chi_max's
value moves after a top-up, re-run g_descend_existing for the affected
g-points afterwards to refresh chi<chi_max too.

Sweep budget scales with how far the last sweep's residual sits above the
convergence threshold, in orders of magnitude:
    distance = max(0, log10(max_energy_reldiff / energy_threshold),
                       log10(max_ee_diff / entanglement_threshold))
    extra_sweeps = max(sweeps_per_order, ceil(sweeps_per_order * distance))
i.e. sweeps_per_order (default 10) extra sweeps per decade past whichever
threshold (energy or entropy) is worse, with a floor of one full batch for
any point flagged not-converged at all -- a point 0.3 decades over the line
still gets a real chance to cross it, not a token 3-4 sweeps.

Usage:
  python pipeline/rerun_nonconverged.py --g-min 0.1 --g-max 1.5 --n-g 15 \
      --chis 9 18 27 --sweeps-per-order 10 --device mac --dry-run
"""
import argparse
import math
import os
import subprocess

import numpy as np
import yaml

from pipeline.g_sweep import (
    get_folder, DEVICE_DRIVE_PATHS, DEFAULT_DEVICE, DEFAULT_DRIVE_PATH,
    drive_path_for_device, GSS_EXECUTABLE,
)
from pipeline.plot_convergence_vs_g import _last_convergence_row
from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs


def _extra_sweeps_for(last_row, energy_threshold, entanglement_threshold, sweeps_per_order):
    energy_resid = float(last_row["max_energy_reldiff"])
    ee_resid = float(last_row["max_ee_diff"])
    dist_e = math.log10(energy_resid / energy_threshold) if energy_resid > energy_threshold else 0.0
    dist_s = math.log10(ee_resid / entanglement_threshold) if ee_resid > entanglement_threshold else 0.0
    distance = max(dist_e, dist_s, 0.0)
    extra = max(sweeps_per_order, math.ceil(sweeps_per_order * distance))
    return extra, distance


def rerun_nonconverged(
    g_raw_values, chis, Lx=5, Ly=5, shape="parallelogram",
    bound_state=None, chargesx=None, chargesy=None, R=None,
    precision=3, drive_path=DEFAULT_DRIVE_PATH,
    energy_convergence_threshold=1e-10, entanglement_convergence_threshold=1e-10,
    sweeps_per_order=10, max_extra_sweeps=None, unpatched=True, dry_run=False,
):
    """g_raw_values: positive |g| values (actual g = -g_raw), matching
    plot_convergence_vs_g's convention (so the same _last_convergence_row
    lookup applies unchanged)."""
    folder, chargesx, chargesy = get_folder(drive_path, Lx, Ly, shape, bound_state, chargesx, chargesy, R)
    init_tensor_file = f"{drive_path}/tensors.hdf5"
    chi_max = chis[-1]
    # Native floats, not numpy.float64 -- yaml.dump serializes numpy scalars
    # with a Python-specific tag (tag:yaml.org,2002:python/object/apply:...)
    # that gss's yaml.safe_load can't reconstruct, so the run.yml write
    # below would otherwise succeed but every gss invocation would fail
    # immediately with a ConstructorError before running a single sweep.
    g_actual = [float(g) for g in -np.asarray(sorted(g_raw_values))]

    print(f"Scanning {len(g_actual)} g-point(s) at chi_max={chi_max} for non-convergence "
          f"(energy_thr={energy_convergence_threshold:.0e}, "
          f"entanglement_thr={entanglement_convergence_threshold:.0e})...", flush=True)

    todo = []  # (g, extra_sweeps, distance)
    for g in g_actual:
        last = _last_convergence_row(folder, g, chi_max, precision)
        if last["converged_this_sweep"].strip() == "True":
            continue
        extra, distance = _extra_sweeps_for(
            last, energy_convergence_threshold, entanglement_convergence_threshold, sweeps_per_order
        )
        if max_extra_sweeps is not None:
            extra = min(extra, max_extra_sweeps)
        todo.append((g, extra, distance))

    if not todo:
        print("Nothing to do -- every g-point is already converged at chi_max.")
        return []

    print(f"{len(todo)}/{len(g_actual)} point(s) need extra sweeps:")
    for g, extra, distance in todo:
        print(f"  g={g:.{precision}f}  distance={distance:.2f} decades past threshold  -> +{extra} sweeps")

    if dry_run:
        print("\n--dry-run: not launching gss. Re-run without --dry-run to actually top these up.")
        return []

    failed = []
    for g, extra, distance in todo:
        run_folder = f"{folder}/g_{g:.{precision}f}"
        filename = os.path.join(run_folder, "EF_Z.dat")
        create_ids_coeffs_file(Lx, Ly, shape, g, bound_state, chargesx=chargesx, chargesy=chargesy, filename=filename)

        numerics_dict = {
            "opt_structure": {"type": 0},
            "initial_bond_dimension": chi_max,
            "energy_convergence_threshold": energy_convergence_threshold,
            "entanglement_convergence_threshold": entanglement_convergence_threshold,
            "energy_degeneracy_threshold": 1e-13,
            "entanglement_degeneracy_threshold": 1e-10,
            "verbose_sweeps": True,
            "unpatched": unpatched,
            # Resume from THIS SAME g/chi's existing checkpoint -- not a chi
            # change (that's g_descend_existing's job), just more sweeps at
            # the same bond dimension.
            "init_tree": 3,
            "init_tensor_file": init_tensor_file,
            "init_g": g,
            "init_chi": chi_max,
            "max_bond_dimensions": [chi_max],
            "max_num_sweeps": [extra],
            "lanczos_tol": [None],
            "lanczos_maxiter": [None],
        }

        input_dict = {
            "system": {
                "N": nplaqs(Lx, Ly, shape), "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape,
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

        inputfile = os.path.join(run_folder, "input_rerun_nonconverged.yml")
        with open(inputfile, "w") as f:
            yaml.dump(input_dict, f, sort_keys=False)

        print(f"\n[g={g:.{precision}f}] +{extra} sweeps at chi_max={chi_max} "
              f"(was {distance:.2f} decades past threshold)", flush=True)
        try:
            subprocess.run([GSS_EXECUTABLE, inputfile], check=True)
        except subprocess.CalledProcessError as e:
            failed.append(g)
            print(f"[g={g:.{precision}f}] FAILED ({e}) -- continuing with the remaining g-values", flush=True)

    if failed:
        print(f"\n{len(failed)} g-value(s) failed: {[f'{g:.{precision}f}' for g in failed]}")
    print("\nDONE -- re-run pipeline.plot_final_vs_g / pipeline.plot_convergence_vs_g to see updated results.")
    return failed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g-min", type=float, required=True)
    p.add_argument("--g-max", type=float, required=True)
    p.add_argument("--n-g", type=int, required=True)
    p.add_argument("--chis", type=int, nargs="+", default=[9, 18, 27])
    p.add_argument("--sweeps-per-order", type=int, default=10,
                    help="extra sweeps added per decade the last residual sits above threshold")
    p.add_argument("--max-extra-sweeps", type=int, default=None,
                    help="cap on extra sweeps for any single point (default: uncapped)")
    p.add_argument("--lx", type=int, default=5)
    p.add_argument("--ly", type=int, default=5)
    p.add_argument("--shape", default="parallelogram")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--unpatched", action="store_true", default=True)
    p.add_argument("--patched", dest="unpatched", action="store_false")
    p.add_argument("--energy-convergence-threshold", type=float, default=1e-10)
    p.add_argument("--entanglement-convergence-threshold", type=float, default=1e-10)
    p.add_argument("--device", choices=list(DEVICE_DRIVE_PATHS), default=DEFAULT_DEVICE)
    p.add_argument("--drive-path", default=None, help="override the path derived from --device")
    p.add_argument("--dry-run", action="store_true", help="list what would be done without launching gss")
    args = p.parse_args()

    drive_path = args.drive_path or drive_path_for_device(args.device)
    g_raw_values = np.linspace(args.g_min, args.g_max, args.n_g)
    rerun_nonconverged(
        g_raw_values, args.chis, Lx=args.lx, Ly=args.ly, shape=args.shape,
        precision=args.precision, drive_path=drive_path,
        energy_convergence_threshold=args.energy_convergence_threshold,
        entanglement_convergence_threshold=args.entanglement_convergence_threshold,
        sweeps_per_order=args.sweeps_per_order, max_extra_sweeps=args.max_extra_sweeps,
        unpatched=args.unpatched, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
