from typing import List, Union
import numpy as np


class Observable:
    """A class for observables."""

    def __init__(
        self,
        indices: List[int],
        operators_list: List[List[str]],
        coef_list: Union[List[float], List[complex]],
    ):
        """Initialize an Observable object.

        Args:
            indices : Indices of the observable.
            operators_list : List of operators, "S+", "S-", "Sz", "Sx" and "Sy".
            coef_list : Coefficients for each operator.
        """
        self.indices = indices
        self.operators_list = operators_list
        self.coef_list = coef_list

        self.indices_num = len(indices)
        self.operators_num = len(operators_list)


def spin_ind(spin_num):
    if isinstance(spin_num, str):
        if spin_num.startswith("S="):
            spin_value_str = spin_num[2:]
        else:
            print("=" * 50)
            print("⚠️  Error: Invalid spin string format. Expected format 'S=...'.")
            print("=" * 50)
            exit()

        if "/" in spin_value_str:
            numerator, denominator = spin_value_str.split("/")
            spin_value = int(numerator) / int(denominator)
        else:
            spin_value = float(spin_value_str)
    elif isinstance(spin_num, (int, float)):
        spin_value = spin_num
    else:
        print("=" * 50)
        print("⚠️  Error: Invalid type for spin_num. Expected a string or a number.")
        print("=" * 50)
        exit()

    return spin_value


omega = np.exp(1j*2*np.pi/3)

def bare_spin_operator(spin, spin_num):
    spin_value = spin_ind(spin_num)  # Get both string form and numeric value
    # Check if spin_value is valid
    if not (spin_value == int(spin_value) or spin_value == int(spin_value) + 0.5):
        print("=" * 50)
        print(
            "⚠️  Error: Invalid spin number. Spin number must be an integer or half-integer."
        )
        print("=" * 50)
        exit()
    dim = int(2 * spin_value + 1)  # Dimension of the matrix based on the spin value
    if spin == "S+":
        # Construct the raising operator S+
        S_plus = np.zeros((dim, dim), dtype=np.complex128)
        for m in range(dim - 1):
            S_plus[m, m + 1] = np.sqrt(
                spin_value * (spin_value + 1) - (spin_value - m) * (spin_value - m - 1)
            )
        return S_plus

    elif spin == "S-":
        # Construct the raising operator S+
        S_plus = np.zeros((dim, dim), dtype=np.complex128)
        for m in range(dim - 1):
            S_plus[m, m + 1] = np.sqrt(
                spin_value * (spin_value + 1) - (spin_value - m) * (spin_value - m - 1)
            )
        S_minus = np.transpose(S_plus)
        return S_minus

    elif spin == "Sz":
        # Construct the Sz operator
        Sz = np.zeros((dim, dim), dtype=np.complex128)
        for m in range(dim):
            Sz[m, m] = spin_value - m
        return Sz

    elif spin == "Sx":
        Sp = bare_spin_operator("S+", spin_num)
        Sm = bare_spin_operator("S-", spin_num)
        Sx = (Sp + Sm) / 2.0
        return Sx

    elif spin == "Sy":
        Sp = bare_spin_operator("S+", spin_num)
        Sm = bare_spin_operator("S-", spin_num)
        Sy = (Sp - Sm) / 2.0j
        return Sy

    elif spin == "V":
        # Construct the Sz operator
        V = np.zeros((dim, dim), dtype=np.complex128)
        for m in range(dim):
            V[m, m] = omega**(dim-m)
        return V
    
    elif spin == "v":
        # Construct the Sz operator
        V = bare_spin_operator("V", spin_num)
        return V.conjugate().T

    elif spin == "U":
        # Construct the Sz operator
        U = np.zeros((dim, dim), dtype=np.complex128)
        for m in range(dim):
            U[m, (m+1)%dim] = 1
        return U
    
    elif spin == "u":
        # Construct the Sz operator
        U = bare_spin_operator("U", spin_num)
        return U.conjugate().T


def spin_dof(spin_num):
    spin_value = spin_ind(spin_num)
    dim = int(2 * spin_value + 1)
    return dim
