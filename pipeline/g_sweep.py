"""
pipeline/g_sweep.py
--------------------
Reusable, parameterized g-sweep runner -- generalizes sweep.py's hardcoded
module-level script into an importable function plus a CLI, so the other
pipeline stages (plot_final_vs_g.py, plot_convergence_vs_g.py,
make_field_entropy_visuals.py) can all point at whatever sweep this
produces without duplicating the ascend/converge/descend chi-ladder logic.

Behavior is identical to sweep.py: per g (in confined-phase-first order,
i.e. largest |g| first), ascend cheaply through chis[:-1], converge hard
at chis[-1], then descend back down through the same chis for
consistently-converged finite-chi data. Resume-safe (skips a leading
prefix of g_order that already has a chis[0] checkpoint). Writes
tensors.hdf5 (via output.save_tensors) plus one basic.csv/convergence.csv
per (g, chi) under drive_path/shape_.../runs_.../g_.../run_chi-N/ --
exactly what plot_final_vs_g.py, plot_convergence_vs_g.py and
make_field_entropy_visuals.py all read.

CLI example (matches this investigation's 15-point scan):
  python pipeline/g_sweep.py --g-min 0.1 --g-max 1.5 --n-g 15 \
      --chis 9 18 27 --sweeps-ascend 3 --sweeps-converge 10 --sweeps-descend 10 \
      --unpatched

Importable example:
  from pipeline.g_sweep import run_g_sweep
  run_g_sweep(g_values=[0.1, 0.5, 1.0], chis=[9, 18, 27], ...)
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml
from tqdm import tqdm

from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs, get_coord_charges
from Z3_funcs.hdf5_manager import tensor_exists

# Single source of truth for device -> drive_path, matching sweep.py and
# Z3_funcs.visualize_plaquettes' own per-function copies of this mapping.
DEVICE_DRIVE_PATHS = {
    "pc": "D:work/projects/5_Z3",
    "ngt": "/eos/user/f/fdimarca/projects/5_Z3",
    "presto": "/home/fradm/projects/5_Z3",
    "mac": "/Users/fradm/Desktop/projects/5_Z3",
}
DEFAULT_DEVICE = "mac"
DEFAULT_DRIVE_PATH = DEVICE_DRIVE_PATHS[DEFAULT_DEVICE]


def _gss_executable():
    """Absolute path to the `gss` console script installed alongside the
    running interpreter (sys.executable), e.g.
    /opt/homebrew/.../envs/ttn/bin/gss locally or
    $HOME/.conda/envs/ttn/bin/gss on presto.

    subprocess.run(["gss", ...]) only works if the conda env's bin/ is on
    PATH, which a non-interactive shell (a PBS job, a nohup'd background
    run, an IDE's "Run" button) is not guaranteed to have even though the
    right python is being used -- `conda activate` inside such a shell is
    unreliable (see g_sweep_presto.pbs's own comment on this). Resolving
    next to sys.executable sidesteps PATH entirely: `gss` is installed into
    the same env's bin/ as `python` is, always.
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "gss")
    if os.path.isfile(candidate):
        return candidate
    # Fall back to PATH lookup (e.g. sys.executable is a symlink/venv shim
    # in some other layout) so this doesn't regress a setup where the bare
    # "gss" already worked.
    found = shutil.which("gss")
    return found if found else "gss"


GSS_EXECUTABLE = _gss_executable()


def drive_path_for_device(device):
    try:
        return DEVICE_DRIVE_PATHS[device]
    except KeyError:
        raise ValueError(f"Unknown device {device!r} -- expected one of {list(DEVICE_DRIVE_PATHS)}")


def get_folder(drive_path, Lx, Ly, shape, bound_state, chargesx, chargesy, R):
    folder = f"{drive_path}/shape_{shape}/_Lx{Lx}_Ly{Ly}"
    if bound_state is None:
        return f"{folder}/runs_vacuum", chargesx, chargesy
    if chargesx is None:
        chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state=bound_state, shape=shape, R=R)
    xs = "-".join(str(x) for x in chargesx)
    ys = "-".join(str(y) for y in chargesy)
    return f"{folder}/runs_{len(chargesx)}-q/x{xs}_y{ys}", chargesx, chargesy


def run_g_sweep(
    g_values,
    chis=(9, 18, 27),
    Lx=5, Ly=5, shape="parallelogram",
    bound_state=None, chargesx=None, chargesy=None, R=None,
    precision=3,
    sweeps_ascend=3, sweeps_converge=30, sweeps_descend=10,
    energy_convergence_threshold=1e-10,
    entanglement_convergence_threshold=1e-10,
    unpatched=True,
    warm_start_g=False,
    lanczos_tol_ascend=None,
    lanczos_maxiter_ascend=None,
    drive_path=DEFAULT_DRIVE_PATH,
    model_name="z3",
    verbose_sweeps=True,
):
    """Run the ascend/converge/descend chi-ladder g-sweep. g_values are the
    actual physical couplings (e.g. [-0.1, -0.2, ...] -- NOT the g_raw/
    positive convention some of this investigation's older scripts used).
    Visits g_values in confined-phase-first order (largest |g| first).
    Resume-safe: re-running with the same drive_path/g_values/chis skips
    whatever's already checkpointed.

    sweeps_descend defaults to 10, not a token 3: a direct experiment
    (descending chi=50 -> 27 -> 18 -> 9 with only 3 resettling sweeps per
    stage vs. 10) showed the 3-sweep version can be badly under-converged
    -- e.g. chi=27's max_ee_diff stuck at ~5e-5 to 7e-5 (vs. chi=27's own
    10-sweep hard-converge reaching ~1e-9) -- while descending with 10
    sweeps instead not only fixes this but comes out MORE converged than
    independently hard-converging each chi from scratch (e.g. chi=27
    reaching ~1e-12, chi=18 ~1e-9, both beating the ascend/converge/
    descend-built coarse-scan equivalents by 2-6 orders of magnitude).
    3 sweeps just isn't enough budget to re-equilibrate the WHOLE tree
    after a large bond-dimension cut of this size; a small sweep-to-sweep
    residual after only 3 sweeps can mean "stable" without meaning
    "close to the true chi-limited optimum". Don't go back to a low
    sweeps_descend without re-checking this.

    sweeps_descend may also be a list of length len(chis)-1, one entry per
    descend stage in chis[-2::-1] order (i.e. chi_max down to chis[0]) --
    e.g. chis=[9,27,50,81] with sweeps_descend=[25,20,10] gives 81->50 25
    sweeps, 50->27 20 sweeps, 27->9 10 sweeps. A larger bond-dimension cut
    (chi_max down to the next stage right after the expensive converge
    stage) generally needs more resettling budget than a smaller one
    further down the ladder, so tapering the descend budget is a
    reasonable choice, not just an arbitrary distribution of the same
    total. A bare int still broadcasts to every stage as before.
    """
    g_values = [float(g) for g in g_values]
    chis = list(chis)
    n_descend = len(chis) - 1
    if isinstance(sweeps_descend, (list, tuple)):
        if len(sweeps_descend) != n_descend:
            raise ValueError(
                f"sweeps_descend list must have one entry per descend stage "
                f"(len(chis)-1={n_descend}), got {len(sweeps_descend)}: {sweeps_descend}"
            )
        sweeps_descend_list = list(sweeps_descend)
    else:
        sweeps_descend_list = [sweeps_descend] * n_descend
    N = nplaqs(Lx, Ly, shape)
    folder, chargesx, chargesy = get_folder(drive_path, Lx, Ly, shape, bound_state, chargesx, chargesy, R)
    os.makedirs(folder, exist_ok=True)
    save_edges_file(Lx, Ly, shape, tensor_folder=folder)
    init_tensor_file = f"{drive_path}/tensors.hdf5"

    print(f"model: {model_name}, shape: {shape}, Lx:{Lx}, Ly:{Ly}, bound_state: {bound_state}")
    print(f"parameter space: {len(g_values)} points, g ext: {min(g_values):.{precision}f}..{max(g_values):.{precision}f}")
    print(f"bond dimensions: {chis} (ascend {sweeps_ascend}/converge {sweeps_converge}/descend {sweeps_descend_list}), "
          f"unpatched: {unpatched}, warm_start_g: {warm_start_g}", flush=True)

    # confined-phase-first order: sort by |g| descending
    g_order = sorted(g_values, key=lambda g: -abs(g))
    descend_chis = chis[-2::-1]

    resume_start_idx = 0
    previous_g = None
    for g in g_order:
        if tensor_exists(init_tensor_file, shape, Lx, Ly, bound_state, R, g, precision,
                          chis[0], chargesx=chargesx, chargesy=chargesy):
            resume_start_idx += 1
            previous_g = g
        else:
            break
    if resume_start_idx > 0:
        print(f"Resuming: first {resume_start_idx} g-value(s) already checkpointed "
              f"(last: g={previous_g:.{precision}f}).")

    pbar = tqdm(g_order[resume_start_idx:], dynamic_ncols=True)
    for g in pbar:
        use_warm_start = warm_start_g and previous_g is not None
        pbar.set_description(
            f"g={g:.{precision}f} (" + (f"warm from g={previous_g:.{precision}f}" if use_warm_start else "cold start") + ")"
        )

        run_folder = f"{folder}/g_{g:.{precision}f}"
        os.makedirs(run_folder, exist_ok=True)
        filename = os.path.join(run_folder, "EF_Z.dat")
        create_ids_coeffs_file(Lx, Ly, shape, g, bound_state, chargesx=chargesx, chargesy=chargesy, filename=filename)

        numerics_dict = {
            "opt_structure": {"type": 0},
            "initial_bond_dimension": chis[0],
            "energy_convergence_threshold": energy_convergence_threshold,
            "entanglement_convergence_threshold": entanglement_convergence_threshold,
            "energy_degeneracy_threshold": 1e-13,
            "entanglement_degeneracy_threshold": 1e-10,
            "verbose_sweeps": verbose_sweeps,
            "unpatched": unpatched,
        }

        if use_warm_start:
            numerics_dict["init_tree"] = 3
            numerics_dict["init_tensor_file"] = init_tensor_file
            numerics_dict["init_g"] = previous_g
            numerics_dict["init_chi"] = chis[-1]
            numerics_dict["max_bond_dimensions"] = [chis[-1]] + descend_chis
            numerics_dict["max_num_sweeps"] = [sweeps_converge] + sweeps_descend_list
            numerics_dict["lanczos_tol"] = [None] * len(numerics_dict["max_bond_dimensions"])
            numerics_dict["lanczos_maxiter"] = [None] * len(numerics_dict["max_bond_dimensions"])
        else:
            numerics_dict["init_tree"] = 2
            numerics_dict["max_bond_dimensions"] = chis + descend_chis
            numerics_dict["max_num_sweeps"] = (
                [sweeps_ascend] * (len(chis) - 1) + [sweeps_converge] + sweeps_descend_list
            )
            n_ascend = len(chis) - 1
            n_rest = 1 + len(descend_chis)
            ascend_tol = None if unpatched else lanczos_tol_ascend
            ascend_maxiter = None if unpatched else lanczos_maxiter_ascend
            numerics_dict["lanczos_tol"] = [ascend_tol] * n_ascend + [None] * n_rest
            numerics_dict["lanczos_maxiter"] = [ascend_maxiter] * n_ascend + [None] * n_rest

        input_dict = {
            "system": {
                "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape,
                "bound_state": bound_state, "chargesx": chargesx, "chargesy": chargesy, "R": R,
                "model": {"type": model_name, "file": filename},
                "g": g, "MF_X": 1.0 / g, "EF_Z": filename, "precision": precision,
            },
            "numerics": numerics_dict,
            "output": {
                "dir": run_folder, "tensors": folder, "save_tensors": drive_path,
                "single_site": 0, "two_site": 0,
            },
        }

        inputfile = os.path.join(run_folder, "input.yml")
        with open(inputfile, "w") as f:
            yaml.dump(input_dict, f, sort_keys=False)

        subprocess.run([GSS_EXECUTABLE, inputfile], check=True)
        previous_g = g

    print("DONE")
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g-min", type=float, required=True, help="smallest |g| (g_raw), e.g. 0.1")
    p.add_argument("--g-max", type=float, required=True, help="largest |g| (g_raw), e.g. 1.5")
    p.add_argument("--n-g", type=int, required=True)
    p.add_argument("--chis", type=int, nargs="+", default=[9, 18, 27])
    p.add_argument("--sweeps-ascend", type=int, default=3)
    p.add_argument("--sweeps-converge", type=int, default=40)
    p.add_argument("--sweeps-descend", type=int, nargs="+", default=[10],
                    help="one value (broadcast to every descend stage) or one value "
                         "per descend stage, in chi_max-down-to-chis[0] order "
                         "(len(chis)-1 values)")
    p.add_argument("--lx", type=int, default=5)
    p.add_argument("--ly", type=int, default=9)
    p.add_argument("--shape", default="hexagon")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--unpatched", action="store_true", default=True)
    p.add_argument("--patched", dest="unpatched", action="store_false")
    p.add_argument("--warm-start-g", action="store_true")
    # Only meaningful with --patched: the unpatched dense lanczos() has no
    # tolerance/iteration parameters at all, so run_g_sweep drops these.
    p.add_argument("--lanczos-tol-ascend", type=float, default=None,
                    help="loosen the eigensolver tolerance during the cheap ascend "
                         "stages only (patched runs only; e.g. 1e-6)")
    p.add_argument("--lanczos-maxiter-ascend", type=int, default=None,
                    help="cap eigensolver iterations during the ascend stages only "
                         "(patched runs only)")
    p.add_argument("--device", choices=list(DEVICE_DRIVE_PATHS), default=DEFAULT_DEVICE)
    p.add_argument("--drive-path", default=None,
                    help="override the path derived from --device")
    args = p.parse_args()

    if args.unpatched and (
        args.lanczos_tol_ascend is not None or args.lanczos_maxiter_ascend is not None
    ):
        p.error(
            "--lanczos-tol-ascend/--lanczos-maxiter-ascend require --patched; "
            "the original dense lanczos() has no such parameters."
        )

    g_raw = np.linspace(args.g_min, args.g_max, args.n_g)
    g_values = [-float(g) for g in g_raw]
    drive_path = args.drive_path or drive_path_for_device(args.device)

    run_g_sweep(
        g_values=g_values, chis=args.chis, Lx=args.lx, Ly=args.ly, shape=args.shape,
        precision=args.precision, sweeps_ascend=args.sweeps_ascend,
        sweeps_converge=args.sweeps_converge,
        # A single CLI value still means "broadcast to every descend stage"
        # (nargs="+" always returns a list, even for one value) -- only an
        # explicit multi-value list is treated as per-stage.
        sweeps_descend=args.sweeps_descend[0] if len(args.sweeps_descend) == 1 else args.sweeps_descend,
        unpatched=args.unpatched, warm_start_g=args.warm_start_g, drive_path=drive_path,
        lanczos_tol_ascend=args.lanczos_tol_ascend,
        lanczos_maxiter_ascend=args.lanczos_maxiter_ascend,
    )


if __name__ == "__main__":
    main()