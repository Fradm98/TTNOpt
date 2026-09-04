import os
import pandas as pd
import numpy as np
from Z3_funcs.lattice_plaquettes import get_coord_charges


def get_folder(Lx, Ly, shape, bound_state=None, chargesx=None, chargesy=None, R=1, device="pc"):
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"

    folder = f"{drive_path}/shape_{shape}/_Lx{Lx}_Ly{Ly}"
    if bound_state is None:
        folder = f"{folder}/runs_vacuum"
    else:
        if chargesx is None:
            chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state=bound_state, R=R)
        xs = "-".join(str(x) for x in chargesx)
        ys = "-".join(str(y) for y in chargesy)
        config = f"x{xs}_y{ys}"
        folder = f"{folder}/runs_{len(chargesx)}-q"
        folder = f"{folder}/{config}"
    return folder


def get_chi_energies_from_results(folder, chi):
    df = pd.read_csv(f"{folder}/results.csv")
    return df[df["chi"] == chi]["energy"].values, df[df["chi"] == chi]["g"].values

def get_chi_energy_from_coupling(folder, chi, g, precision=3):
    folder = folder + f"/g_-{g:.{precision}f}" + f"/run_chi-{chi}"
    df = pd.read_csv(f"{folder}/basic.csv")
    return df["energy"].iloc[-1]

def get_chi_energies_from_couplings(folder, chi, g_values, precision=3):
    energies = []
    for g in g_values:
        energy = get_chi_energy_from_coupling(folder, chi, g, precision)
        energies.append(energy)
    return np.asarray(energies)

def get_chi_entropies_from_coupling(folder, chi, g, precision=3):
    folder = folder + f"/g_-{g:.{precision}f}" + f"/run_chi-{chi}"
    df = pd.read_csv(f"{folder}/basic.csv")
    return df["entanglement"].iloc[-1]

def get_chi_entropies_from_couplings(folder, chi, g_values, precision=3):
    entropies = []
    for g in g_values:
        entropy = get_chi_entropies_from_coupling(folder, chi, g, precision)
        entropies.append(entropy)
    return np.asarray(entropies)

def get_chi_error_from_coupling(folder, chi, g, precision=3):
    folder = folder + f"/g_-{g:.{precision}f}" + f"/run_chi-{chi}"
    df = pd.read_csv(f"{folder}/basic.csv")
    return df["error"].iloc[-1]

def get_chi_errors_from_couplings(folder, chi, g_values, precision=3):
    errors = []
    for g in g_values:
        error = get_chi_error_from_coupling(folder, chi, g, precision)
        errors.append(error)
    return np.asarray(errors)

def get_chi_energy_conv_from_coupling(folder, chi, g, precision=3):
    folder = folder + f"/g_-{g:.{precision}f}" + f"/run_chi-{chi}"
    df = pd.read_csv(f"{folder}/convergence.csv")
    return df["max_energy_reldiff"].iloc[-1]

def get_chi_energies_conv_from_couplings(folder, chi, g_values, precision=3):
    energies = []
    for g in g_values:
        energy = get_chi_energy_conv_from_coupling(folder, chi, g, precision)
        energies.append(energy)
    return np.asarray(energies)

def get_chi_entropy_conv_from_coupling(folder, chi, g, precision=3):
    folder = folder + f"/g_-{g:.{precision}f}" + f"/run_chi-{chi}"
    df = pd.read_csv(f"{folder}/convergence.csv")
    return df["max_ee_diff"].iloc[-1]

def get_chi_entropies_conv_from_couplings(folder, chi, g_values, precision=3):
    energies = []
    for g in g_values:
        energy = get_chi_entropy_conv_from_coupling(folder, chi, g, precision)
        energies.append(energy)
    return np.asarray(energies)

def define_R(chargesx, chargesy, bound_state):
    if bound_state == "meson":
        return int(abs(chargesx[-1] - chargesx[0]))
    elif bound_state == "baryon":
        return np.sqrt((chargesx[0]-chargesx[1])**2 + ((chargesy[0]-chargesy[1])/2)**2)
    elif bound_state is None:
        return None
    else:
        raise ValueError(f"Unknown bound state {bound_state}, select among 'meson', 'baryon', or None")