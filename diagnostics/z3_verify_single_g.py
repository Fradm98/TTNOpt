"""
z3_verify_single_g.py
----------------------
Re-run sweep.py's exact cold-start, unpatched ascend/converge/descend chi
ladder for ONE g value, with a configurable (larger) converge-stage sweep
budget, writing to an ISOLATED output directory -- never touches the
production tensors.hdf5 or shape_parallelogram/.../run_chi-* folders that
the real 15-point g-scan (sweep.py) already wrote.

Why this exists: g=-0.700 (right at the entropy peak / suspected vacuum
transition) converged its outer-sweep criteria to only ~1e-8 (ref_energy_reldiff)
after using its full SWEEP_CONVERGE=10 budget at chi=27, while every other
point in the 15-point scan reached ~1e-13..1e-15 in the same budget. Its
convergence.csv shows reldiff still shrinking smoothly and monotonically
(no oscillation) sweep over sweep -- consistent with a harder, more slowly
convergent point near criticality rather than a stuck/bugged state, but
that needs to actually be confirmed with a larger sweep budget before
trusting it enough to build a finer-resolution scan on top of it.

This mirrors sweep.py's per-g block (ascend chi[:-1] cheaply, converge hard
at chi[-1], descend back down through chi[:-1] reversed) exactly, just for
one g and with --sweeps-converge overridable.

Usage:
  python diagnostics/z3_verify_single_g.py --g -0.7 --sweeps-converge 25
"""
import argparse
import os
import subprocess

import yaml

from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.utils import get_chi_energy_from_coupling, get_chi_entropies_from_coupling

Lx, Ly, SHAPE = 5, 5, "parallelogram"
PRECISION = 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--g", type=float, required=True, help="actual physical coupling, e.g. -0.7")
    p.add_argument("--chis", type=int, nargs="+", default=[9, 18, 27])
    p.add_argument("--sweeps-ascend", type=int, default=3)
    p.add_argument("--sweeps-converge", type=int, default=25)
    p.add_argument("--sweeps-descend", type=int, default=3)
    p.add_argument("--out-dir", type=str,
                    default="/Users/fradm/Desktop/projects/5_Z3_verify")
    args = p.parse_args()

    g = args.g
    chis = args.chis
    descend_chis = chis[-2::-1]

    N = nplaqs(Lx, Ly, SHAPE)
    folder = f"{args.out_dir}/shape_{SHAPE}/_Lx{Lx}_Ly{Ly}/runs_vacuum"
    os.makedirs(folder, exist_ok=True)
    save_edges_file(Lx, Ly, SHAPE, tensor_folder=folder)

    run_folder = f"{folder}/g_{g:.{PRECISION}f}"
    os.makedirs(run_folder, exist_ok=True)
    filename = os.path.join(run_folder, "EF_Z.dat")
    create_ids_coeffs_file(Lx, Ly, SHAPE, g, None, filename=filename)  # vacuum

    numerics_dict = {
        "opt_structure": {"type": 0},
        "initial_bond_dimension": chis[0],
        "energy_convergence_threshold": 1e-5,
        "entanglement_convergence_threshold": 1e-10,
        "energy_degeneracy_threshold": 1e-13,
        "entanglement_degeneracy_threshold": 1e-8,
        "verbose_sweeps": True,
        "unpatched": True,
        "init_tree": 2,  # cold start, matching sweep.py's WARM_START_G=False
        "max_bond_dimensions": chis + descend_chis,
        "max_num_sweeps": (
            [args.sweeps_ascend] * (len(chis) - 1)
            + [args.sweeps_converge]
            + [args.sweeps_descend] * len(descend_chis)
        ),
        "lanczos_tol": [None] * (len(chis) + len(descend_chis)),
        "lanczos_maxiter": [None] * (len(chis) + len(descend_chis)),
    }

    input_dict = {
        "system": {
            "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": SHAPE,
            "bound_state": None, "chargesx": None, "chargesy": None, "R": None,
            "model": {"type": "z3", "file": filename},
            "g": g, "MF_X": 1.0 / g, "EF_Z": filename, "precision": PRECISION,
        },
        "numerics": numerics_dict,
        "output": {
            "dir": run_folder, "tensors": folder, "save_tensors": args.out_dir,
            "single_site": 0, "two_site": 0,
        },
    }

    inputfile = os.path.join(run_folder, "input.yml")
    with open(inputfile, "w") as f:
        yaml.dump(input_dict, f, sort_keys=False)

    print(f"g={g}, chis={chis + descend_chis}, "
          f"sweeps={[args.sweeps_ascend]*(len(chis)-1) + [args.sweeps_converge] + [args.sweeps_descend]*len(descend_chis)}, "
          f"unpatched=True, cold start, output isolated under {args.out_dir}", flush=True)
    subprocess.run(["gss", inputfile], check=True)

    top_chi = chis[0]  # final descend stage always ends back at chis[0]
    e = get_chi_energy_from_coupling(folder, top_chi, -g, precision=PRECISION)
    s = get_chi_entropies_from_coupling(folder, top_chi, -g, precision=PRECISION)
    print(f"\nfinal (after descend to chi={top_chi}): E={e:.12f}  S={s:.8f}")
    for chi in chis:
        conv_path = f"{run_folder}/run_chi-{chi}/convergence.csv"
        if os.path.exists(conv_path):
            with open(conv_path) as f:
                last = f.readlines()[-1].strip()
            print(f"  [chi={chi}] last convergence.csv row: {last}")
    print("DONE")


if __name__ == "__main__":
    main()
