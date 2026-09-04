from Z3_funcs.observables import (static_potential_chi,
                         static_potential,
                         static_potential_R,
                         discrete_string_tension,
                         discrete_luscher_term,
                         discrete_string_tension_chi,
                         discrete_string_tension_R,
                         discrete_luscher_term_chi,
                         discrete_luscher_term_R,
                         half_cut_entropy,
                         half_cut_entropy_chi,
                         error,
                         error_chi,
                         energy,
                         energy_chi,
                         energy_R,
                        )
import matplotlib.pyplot as plt
import numpy as np

# All error bars come from convergence.csv's sweep-to-sweep diagnostics and
# are only as good as that diagnostic (see observables.py); points without a
# convergence.csv (e.g. ascend-leg-only stages) come back as nan, which
# np.nan_to_num turns into a zero-length (i.e. invisible) error bar rather
# than crashing plt.errorbar.

def plot_observable(observable_func, Lx, Ly, shape, fixed, bound_state=None, device="pc", g_values=None):
    observable_values, errors, gs = observable_func(Lx, Ly, shape, fixed, bound_state=bound_state, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    plt.errorbar(np.abs(gs), observable_values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3)
    plt.xlabel("g")
    plt.ylabel(observable_func.__name__.replace('_', ' ').title())
    plt.title(f"{observable_func.__name__.replace('_', ' ').title()} vs g for shape {shape}, Lx={Lx}, Ly={Ly}")
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/{observable_func.__name__}_{shape}_Lx{Lx}_Ly{Ly}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_observable_param(observable_func, Lx, Ly, shape, params, fixed, bound_state=None, device="pc", g_values=None):
    observable_values = observable_func(Lx, Ly, shape, params, fixed, bound_state=bound_state, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for param in params:
        values, errors, gs = observable_values[param]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"param={param}")
    plt.xlabel("g")
    plt.ylabel(observable_func.__name__.replace('_', ' ').title())
    plt.title(f"{observable_func.__name__.replace('_', ' ').title()} vs g for shape {shape}, Lx={Lx}, Ly={Ly}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/{observable_func.__name__}_{shape}_Lx{Lx}_Ly{Ly}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_energy(Lx, Ly, shape, chi, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    energies, errors, gs = energy(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    plt.errorbar(np.abs(gs), energies, yerr=np.nan_to_num(errors), fmt='-o', capsize=3)
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.title(f"Energy vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}, R={R}")
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/energy_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_energy_chi(Lx, Ly, shape, chis, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    energies = energy_chi(Lx, Ly, shape, chis, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for chi in chis:
        values, errors, gs = energies[chi]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"chi={chi}")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.title(f"Energy vs g for shape {shape}, Lx={Lx}, Ly={Ly}, R={R}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/energy_{shape}_Lx{Lx}_Ly{Ly}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_energy_R(Lx, Ly, shape, chi, Rs, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    energies = energy_R(Lx, Ly, shape, chi, Rs, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for R in Rs:
        values, errors, gs = energies[R]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"R={R}")
    plt.xlabel("g")
    plt.ylabel("Energy")
    plt.title(f"Energy vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/energy_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_static_potential(Lx, Ly, shape, chi, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    static_potentials, errors, gs = static_potential(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    plt.errorbar(np.abs(gs), static_potentials, yerr=np.nan_to_num(errors), fmt='-o', capsize=3)
    plt.xlabel("g")
    plt.ylabel("Static Potential")
    plt.title(f"Static Potential vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}, R={R}")
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/static_potential_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_static_potential_chi(Lx, Ly, shape, chis, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    static_potentials = static_potential_chi(Lx, Ly, shape, chis, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for chi in chis:
        values, errors, gs = static_potentials[chi]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"chi={chi}")
    plt.xlabel("g")
    plt.ylabel("Static Potential")
    plt.title(f"Static Potential vs g for shape {shape}, Lx={Lx}, Ly={Ly}, R={R}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/static_potential_{shape}_Lx{Lx}_Ly{Ly}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_static_potential_R(Lx, Ly, shape, chi, Rs, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    static_potentials = static_potential_R(Lx, Ly, shape, chi, Rs, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for R in Rs:
        values, errors, gs = static_potentials[R]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"R={R}")
    plt.xlabel("g")
    plt.ylabel("Static Potential")
    plt.title(f"Static Potential vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/static_potential_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_discrete_string_tension(Lx, Ly, shape, chi, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    string_tensions, errors, gs = discrete_string_tension(Lx, Ly, shape, chi, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    plt.errorbar(np.abs(gs), string_tensions, yerr=np.nan_to_num(errors), fmt='-o', capsize=3)
    plt.xlabel("g")
    plt.ylabel("Discrete String Tension")
    plt.title(f"Discrete String Tension vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}, R={R}")
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/discrete_string_tension_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_discrete_string_tension_chi(Lx, Ly, shape, chis, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    string_tensions = discrete_string_tension_chi(Lx, Ly, shape, chis, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for chi in chis:
        values, errors, gs = string_tensions[chi]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"chi={chi}")
    plt.xlabel("g")
    plt.ylabel("Discrete String Tension")
    plt.title(f"Discrete String Tension vs g for shape {shape}, Lx={Lx}, Ly={Ly}, R={R}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/discrete_string_tension_{shape}_Lx{Lx}_Ly{Ly}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_discrete_string_tension_R(Lx, Ly, shape, chi, Rs, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    string_tensions = discrete_string_tension_R(Lx, Ly, shape, chi, Rs, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for R in Rs:
        values, errors, gs = string_tensions[R]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"R={R}")
    plt.xlabel("g")
    plt.ylabel("Discrete String Tension")
    plt.title(f"Discrete String Tension vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/discrete_string_tension_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_discrete_luscher_term(Lx, Ly, shape, chi, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    luscher_terms, errors, gs = discrete_luscher_term(Lx, Ly, shape, chi, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    plt.errorbar(np.abs(gs), luscher_terms, yerr=np.nan_to_num(errors), fmt='-o', capsize=3)
    plt.xlabel("g")
    plt.ylabel("Discrete Lüscher Term")
    plt.title(f"Discrete Lüscher Term vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}, R={R}")
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/discrete_luscher_term_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_discrete_luscher_term_chi(Lx, Ly, shape, chis, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    luscher_terms = discrete_luscher_term_chi(Lx, Ly, shape, chis, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for chi in chis:
        values, errors, gs = luscher_terms[chi]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"chi={chi}")
    plt.xlabel("g")
    plt.ylabel("Discrete Lüscher Term")
    plt.title(f"Discrete Lüscher Term vs g for shape {shape}, Lx={Lx}, Ly={Ly}, R={R}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/discrete_luscher_term_{shape}_Lx{Lx}_Ly{Ly}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_discrete_luscher_term_R(Lx, Ly, shape, chi, Rs, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    luscher_terms = discrete_luscher_term_R(Lx, Ly, shape, chi, Rs, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for R in Rs:
        values, errors, gs = luscher_terms[R]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"R={R}")
    plt.xlabel("g")
    plt.ylabel("Discrete Lüscher Term")
    plt.title(f"Discrete Lüscher Term vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/discrete_luscher_term_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_half_cut_entropy(Lx, Ly, shape, chi, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    entropies, errors, gs = half_cut_entropy(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    plt.errorbar(np.abs(gs), entropies, yerr=np.nan_to_num(errors), fmt='-o', capsize=3)
    plt.xlabel("g")
    plt.ylabel("Half-Cut Entropy")
    plt.title(f"Half-Cut Entropy vs g for shape {shape}, Lx={Lx}, Ly={Ly}, chi={chi}, R={R}")
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/half_cut_entropy_{shape}_Lx{Lx}_Ly{Ly}_chi{chi}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_half_cut_entropy_chi(Lx, Ly, shape, chis, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    entropies = half_cut_entropy_chi(Lx, Ly, shape, chis, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fradm/projects/5_Z3"
    elif device == "mac":
        drive_path = "/Users/fradm/Desktop/projects/5_Z3"
    figure_path = f"{drive_path}/figures"

    for chi in chis:
        values, errors, gs = entropies[chi]
        plt.errorbar(np.abs(gs), values, yerr=np.nan_to_num(errors), fmt='-o', capsize=3, label=f"chi={chi}")
    plt.xlabel("g")
    plt.ylabel("Half-Cut Entropy")
    plt.title(f"Half-Cut Entropy vs g for shape {shape}, Lx={Lx}, Ly={Ly}, R={R}")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f"{figure_path}/half_cut_entropy_{shape}_Lx{Lx}_Ly{Ly}_R{R}.png", dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    # Example usage
    Lx = 3
    Ly = 3
    shape = "hexagon"
    shape = "parallelogram"

    chi = 40
    chis = [9,18,27,36,45,54,63,81]
    R = 1
    bound_state = "baryon"
    bound_state = "meson"
    chargesx, chargesy = None, None
    device = "pc"
    g_values = None
    g_values = [0.1,0.14,0.18,0.21,0.25,0.29,0.33,0.37,0.4,0.44,0.48,0.56,0.59,0.63,0.67,0.71,0.75,0.78,0.82,0.86,0.90]
    g_values = np.linspace(0.5,1,6)

    x = np.linspace(0,5,100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x,y1,label='sine function')
    plt.plot(x,y2,label='cosine function')
    plt.vlines(x=np.pi, ymin=min(min(y1),min(y2)), ymax=max(max(y1),max(y2)), linestyles='--', colors='black', label="$\\pi$")
    plt.legend()
    plt.text(0,0,"ciao campione!")
    plt.show()
    # plot_energy_chi(Lx, Ly, shape, chis, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    # plot_half_cut_entropy_chi(Lx, Ly, shape, chis, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    # plot_static_potential(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    # plot_static_potential_chi(Lx, Ly, shape, chis, R=R, bound_state=bound_state,chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)

    # # plot_discrete_string_tension(Lx, Ly, shape, chi, R, a=1, bound_state=bound_state, device=device)
    # # plot_discrete_string_tension_chi(Lx, Ly, shape, chis, R, a=1, bound_state=bound_state, device=device)
    # # plot_discrete_luscher_term(Lx, Ly, shape, chi, R, a=1, bound_state=bound_state, device=device)
    # # plot_discrete_luscher_term_chi(Lx, Ly, shape, chis, R, a=1, bound_state=bound_state, device=device)

    # plot_half_cut_entropy(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    # plot_half_cut_entropy_chi(Lx, Ly, shape, chis, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
