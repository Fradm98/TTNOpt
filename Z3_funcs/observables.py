from Z3_funcs.utils import (get_folder,
                         get_chi_energies_from_results,
                         get_chi_energies_from_couplings,
                         get_chi_entropies_from_couplings,
                         get_chi_errors_from_couplings,
                         get_chi_energies_conv_from_couplings,
                         get_chi_entropies_conv_from_couplings)
import numpy as np


def _energy_and_error(folder, chi, g_values):
    """Energy plus its absolute error for one folder (vacuum or bound-state).

    The error is convergence.csv's max_energy_reldiff (relative, sweep N vs
    sweep N-1) turned into an absolute uncertainty via |reldiff * energy|.
    """
    energies = get_chi_energies_from_couplings(folder, chi, g_values)
    max_dE_rel = get_chi_energies_conv_from_couplings(folder, chi, g_values)
    errors = np.abs(max_dE_rel * energies)
    return energies, errors


def energy(Lx, Ly, shape, chi, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    folder = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R, device=device)
    if g_values is None:
        energies, gs = get_chi_energies_from_results(folder, chi)
        errors = np.full_like(energies, np.nan, dtype=float)
    else:
        energies, errors = _energy_and_error(folder, chi, g_values)
        gs = g_values
    return energies, errors, gs

def energy_chi(Lx, Ly, shape, chis, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    energies = {}
    for chi in chis:
        energies[chi] = energy(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return energies

def energy_R(Lx, Ly, shape, chi, Rs, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    energies = {}
    for R in Rs:
        energies[R] = energy(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return energies

def static_potential(Lx, Ly, shape, chi, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    folder_vacuum = get_folder(Lx, Ly, shape, bound_state=None, chargesx=chargesx, chargesy=chargesy, R=R, device=device)
    folder_bs = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R, device=device)
    if g_values is None:
        energies_vacuum, _ = get_chi_energies_from_results(folder_vacuum, chi)
        energies_bs, gs = get_chi_energies_from_results(folder_bs, chi)
        errors_vacuum = np.full_like(energies_vacuum, np.nan, dtype=float)
        errors_bs = np.full_like(energies_bs, np.nan, dtype=float)
    else:
        energies_vacuum, errors_vacuum = _energy_and_error(folder_vacuum, chi, g_values)
        energies_bs, errors_bs = _energy_and_error(folder_bs, chi, g_values)
        gs = g_values
    values = np.abs(energies_bs - energies_vacuum)
    errors = np.sqrt(errors_bs**2 + errors_vacuum**2)
    return values, errors, gs

def static_potential_chi(Lx, Ly, shape, chis, R=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    static_potentials = {}
    for chi in chis:
        static_potentials[chi] = static_potential(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return static_potentials

def static_potential_R(Lx, Ly, shape, chi, Rs, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    static_potentials = {}
    for R in Rs:
        static_potentials[R] = static_potential(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return static_potentials


def discrete_string_tension(Lx, Ly, shape, chi, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    static_potentials, _, _ = static_potential(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    static_potentials_minus_a, _, gs = static_potential(Lx, Ly, shape, chi, R=R-a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    values = (static_potentials - static_potentials_minus_a) / a

    if g_values is None:
        errors = np.full_like(values, np.nan, dtype=float)
    else:
        # The vacuum energy is the SAME run at every R (see get_folder: the
        # vacuum path never depends on R), so its error cancels exactly in
        # this difference. Propagate from the bound-state energies alone --
        # combining static_potential's own (vacuum-inclusive) errors here
        # would double-count the already-cancelled vacuum contribution.
        folder_bs = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R, device=device)
        folder_bs_minus_a = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R-a, device=device)
        _, err_bs = _energy_and_error(folder_bs, chi, g_values)
        _, err_bs_minus_a = _energy_and_error(folder_bs_minus_a, chi, g_values)
        errors = np.sqrt(err_bs**2 + err_bs_minus_a**2) / abs(a)
    return values, errors, gs

def discrete_string_tension_chi(Lx, Ly, shape, chis, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    string_tensions = {}
    for chi in chis:
        string_tensions[chi] = discrete_string_tension(Lx, Ly, shape, chi, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return string_tensions

def discrete_string_tension_R(Lx, Ly, shape, chi, Rs, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    string_tensions = {}
    for R in Rs:
        string_tensions[R] = discrete_string_tension(Lx, Ly, shape, chi, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return string_tensions


def discrete_luscher_term(Lx, Ly, shape, chi, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    static_potentials, _, _ = static_potential(Lx, Ly, shape, chi, R=R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    static_potentials_plus_a, _, _ = static_potential(Lx, Ly, shape, chi, R=R+a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    static_potentials_minus_a, _, gs = static_potential(Lx, Ly, shape, chi, R=R-a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    values = ((static_potentials_plus_a + static_potentials_minus_a - (2 * static_potentials)) / (2 * a**2)) * (R**3)

    if g_values is None:
        errors = np.full_like(values, np.nan, dtype=float)
    else:
        # Same vacuum-cancellation reasoning as discrete_string_tension: the
        # three static_potential calls above share the identical vacuum run,
        # and its contribution cancels exactly in this second difference
        # (coefficients +1, +1, -2 sum to zero). Propagate from the three
        # bound-state energies alone.
        folder_bs = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R, device=device)
        folder_bs_plus_a = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R+a, device=device)
        folder_bs_minus_a = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R-a, device=device)
        _, err_bs = _energy_and_error(folder_bs, chi, g_values)
        _, err_bs_plus_a = _energy_and_error(folder_bs_plus_a, chi, g_values)
        _, err_bs_minus_a = _energy_and_error(folder_bs_minus_a, chi, g_values)
        errors = np.sqrt(err_bs_plus_a**2 + err_bs_minus_a**2 + 4 * err_bs**2) / (2 * a**2) * np.abs(R)**3
    return values, errors, gs

def discrete_luscher_term_chi(Lx, Ly, shape, chis, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    luscher_terms = {}
    for chi in chis:
        luscher_terms[chi] = discrete_luscher_term(Lx, Ly, shape, chi, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return luscher_terms

def discrete_luscher_term_R(Lx, Ly, shape, chi, Rs, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    luscher_terms = {}
    for R in Rs:
        luscher_terms[R] = discrete_luscher_term(Lx, Ly, shape, chi, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return luscher_terms

def half_cut_entropy(Lx, Ly, shape, chi, R, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    folder = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R, device=device)
    entropies = get_chi_entropies_from_couplings(folder, chi, g_values)
    if g_values is None:
        errors = np.full_like(entropies, np.nan, dtype=float)
    else:
        errors = get_chi_entropies_conv_from_couplings(folder, chi, g_values)
    return entropies, errors, g_values

def half_cut_entropy_chi(Lx, Ly, shape, chis, R, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    half_cut_entropies = {}
    for chi in chis:
        half_cut_entropies[chi] = half_cut_entropy(Lx, Ly, shape, chi, R, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return half_cut_entropies

def error(Lx, Ly, shape, chi, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    return get_chi_errors_from_couplings(get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, R=R, device=device), chi, g_values)

def error_chi(Lx, Ly, shape, chis, R, a=1, bound_state=None, chargesx=None, chargesy=None, device="pc", g_values=None):
    errors = {}
    for chi in chis:
        errors[chi] = error(Lx, Ly, shape, chi, R, a=a, bound_state=bound_state, chargesx=chargesx, chargesy=chargesy, device=device, g_values=g_values)
    return errors


def load_engine_from_tensor(ten_file, Lx, Ly, shape, chi, g, bound_state=None, R=1, precision=3, chargesx=None, chargesy=None):
    """Reconstruct a queryable GroundStateSearch engine from a saved TTN
    checkpoint in ten_file (tensors.hdf5), for post-hoc expectation-value
    queries (e.g. link_electric_fields) on an already-converged state.

    Runs no sweeps: GroundStateSearch's constructor detects the loaded
    tensors are already valid and auto-primes edge_spin_operators/
    block_hamiltonians via _prime_renormalized_operators(). The rebuilt
    Hamiltonian only needs to match the checkpoint's structure (N, site
    dimension) for that priming step to succeed -- expval_general itself
    only reads edge_spin_operators, never block_hamiltonians -- but it's
    rebuilt with the same g/bound_state/R as the checkpoint for full
    consistency.
    """
    import tempfile
    from dotmap import DotMap
    from ttnopt.src import GroundStateSearch
    from ttnopt.hamiltonian import hamiltonian
    from Z3_funcs.create_graph_file import create_ids_coeffs_file, nplaqs
    from Z3_funcs.hdf5_manager import load_tensor

    psi, _ = load_tensor(
        ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi,
        chargesx=chargesx, chargesy=chargesy,
    )
    if psi.gauge_tensor is None:
        raise ValueError(
            f"Checkpoint g={g}, chi={chi}, Lx={Lx}, Ly={Ly}, shape={shape} has no "
            f"saved gauge_tensor -- it predates gauge_tensor being included in "
            f"save_tensor, so the canonical center's state isn't fully specified "
            f"and post-hoc expectation-value queries (move_canonical_center/"
            f"expval_general) aren't possible on it. Rerun to regenerate the "
            f"checkpoint with the current code."
        )

    N = nplaqs(Lx, Ly, shape)
    with tempfile.TemporaryDirectory() as tmpdir:
        ef_z_path = f"{tmpdir}/EF_Z.dat"
        create_ids_coeffs_file(
            Lx, Ly, shape, g, bound_state, R=R,
            chargesx=chargesx, chargesy=chargesy, filename=ef_z_path,
        )
        system_cfg = DotMap({
            "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape,
            "model": {"type": "z3", "file": ef_z_path},
            "MF_X": 1.0 / g, "EF_Z": ef_z_path,
        })
        ham = hamiltonian(system_cfg)
        engine = GroundStateSearch(psi, ham, init_bond_dim=chi, max_bond_dim=chi)

    return engine


class TreeExpval:
    """Expectation values of products of one-site operators on a TTN in
    canonical form, without moving the canonical center.

    Every tensor away from the center is an isometry towards it, so a
    subtree with no operator in it contracts to the identity. An operator
    is therefore carried up from its site only along its own branch, and
    the two branches meet at their common ancestor or at the center. Cost
    per value: (tree depth) x chi^3, plus chi^4 at the two central tensors.

    Same contraction convention as PhysicsEngine (_set_edge_spin,
    expval_twosite): the operator's first index goes with the state, the
    second with its conjugate, and the value is divided by <psi|psi>.

    psi is read only, never modified. Single-site operators carried up a
    branch are cached, since every link of a plaquette reuses them.
    """

    def __init__(self, psi, spin_size):
        from ttnopt.src.Observable import bare_spin_operator

        self.psi = psi
        self.center = psi.canonical_center_edge_id
        self.central = psi.central_tensor_ids()
        if len(self.central) != 2:
            raise ValueError(f"expected 2 central tensors, got {self.central}")
        self.below = {}  # internal edge -> tensor under it (center excluded)
        self.parent_tensor = {}  # edge -> tensor that has it as a child
        for t, (c0, c1, p) in enumerate(psi.edges):
            if p != self.center:
                self.below[p] = t
            self.parent_tensor[c0] = t
            self.parent_tensor[c1] = t
        self.sites_under = {}
        for e in self.parent_tensor:
            self._sites(e)
        self.bare = {}
        for site, size in spin_size.items():
            V = bare_spin_operator("V", size)
            self.bare[site] = {"V": V, "v": V.conj().T}
        self._cache = {}
        self._norm2 = None

    def _sites(self, e):
        if e not in self.sites_under:
            t = self.below.get(e)
            if t is None:
                self.sites_under[e] = frozenset([e])
            else:
                c0, c1, _ = self.psi.edges[t]
                self.sites_under[e] = self._sites(c0) | self._sites(c1)
        return self.sites_under[e]

    @staticmethod
    def _through(T, R0, R1):
        """Carry child operators R0 (leg 0) and R1 (leg 1) through T to its
        parent leg; None stands for the identity."""
        if R0 is None and R1 is None:
            return np.einsum("ika,ikb->ab", T, T.conj())
        if R1 is None:
            return np.einsum("ika,ij,jkb->ab", T, R0, T.conj(), optimize=True)
        if R0 is None:
            return np.einsum("ika,kl,ilb->ab", T, R1, T.conj(), optimize=True)
        return np.einsum("ika,ij,kl,jlb->ab", T, R0, R1, T.conj(), optimize=True)

    def _up(self, e, ops):
        """Operator on edge e from ops = {site: name} under it, or None if
        no operator is under e (identity, by isometry)."""
        here = {s: o for s, o in ops.items() if s in self.sites_under[e]}
        if not here:
            return None
        key = (e, tuple(sorted(here.items())))
        if key in self._cache:
            return self._cache[key]
        t = self.below.get(e)
        if t is None:
            R = self.bare[e][here[e]]
        else:
            c0, c1, _ = self.psi.edges[t]
            R = self._through(self.psi.tensors[t], self._up(c0, here), self._up(c1, here))
        if len(here) == 1:  # products of two are rarely reused
            self._cache[key] = R
        return R

    def _center_value(self, ops):
        sides = []
        for t in self.central:
            c0, c1, _ = self.psi.edges[t]
            sides.append(self._through(self.psi.tensors[t], self._up(c0, ops), self._up(c1, ops)))
        G = self.psi.gauge_tensor
        return np.einsum("xy,xa,yb,ab->", G, sides[0], sides[1], G.conj(), optimize=True)

    def expval(self, ops):
        """<prod_site op(site)> / <psi|psi> for ops = {site: "V" or "v"}."""
        if self._norm2 is None:
            self._norm2 = np.real(self._center_value({}))
        return self._center_value(ops) / self._norm2


def link_electric_fields(engine, plaquette_label, Lx, Ly, shape="parallelogram", done=None, on_link=None, fast=True):
    """<V.v> electric-field link values, matching Hamiltonian.py's
    z3-model convention exactly (see ttnopt/src/Hamiltonian.py's "z3"
    branch): V is applied at the link's left-slot plaquette (label_links'
    first tuple entry), v at the right-slot plaquette; on a boundary link
    (only one side exists), the lone surviving operator is "V" if that
    side is the left slot, "v" if it's the right slot.

    plaquette_label : if given, restrict to links bordering that one
        plaquette. If None, compute every link in the lattice (checks
        every (i,j,dir) entry of label_links, i.e. every site pair (i,j)
        that label_links reports as an existing direct-lattice link).

    done : dict of links already computed (same format as the return
        value); those links are kept and not recomputed.
    on_link : optional callback(key, results, n_total), called after each
        newly computed link with everything computed so far.

    engine must already hold a valid, loaded state (e.g. from
    load_engine_from_tensor).

    fast : True (default) evaluates every link with TreeExpval on
        engine.psi, leaving the state untouched. False uses the old
        engine.expval_general path, which moves the canonical center and
        rebuilds every renormalized operator for each link (kept for
        cross-checks).

    Returns
    -------
    dict (i, j, dir) -> {
        "p_left": plaquette label or None (label_links' left slot, gets V),
        "p_right": plaquette label or None (right slot, gets v),
        "neighbor": only meaningful when plaquette_label is given -- the
            *other* bordering plaquette (i.e. whichever of p_left/p_right
            isn't plaquette_label), or None on a boundary link. Always
            None when plaquette_label is None (there's no fixed "self"
            side to be relative to).
        "value": complex <V.v> (interior) or <V>/<v> (boundary),
        "kind": "Vv", "V", or "v",
    }
    for every matching link.
    """
    from Z3_funcs.lattice_plaquettes import label_links

    link_plaquettes, _ = label_links(Lx, Ly, shape)
    results = dict(done or {})
    n_total = sum(1 for p_left, p_right in link_plaquettes.values()
                  if plaquette_label is None or plaquette_label in (p_left, p_right))
    if fast:
        tree = TreeExpval(engine.psi, engine.hamiltonian.spin_size)

        def expval(p_i, op_i, p_j=None, op_j=None):
            ops = {p_i: op_i} if p_j is None else {p_i: op_i, p_j: op_j}
            return complex(tree.expval(ops))
    else:
        expval = engine.expval_general
    for key, (p_left, p_right) in link_plaquettes.items():
        if plaquette_label is not None and plaquette_label not in (p_left, p_right):
            continue
        if key in results:
            continue
        if p_left is not None and p_right is not None:
            value = expval(p_left, "V", p_right, "v")
            if plaquette_label is None:
                neighbor = None
            else:
                neighbor = p_right if plaquette_label == p_left else p_left
            kind = "Vv"
        elif p_left is not None:
            value = expval(p_left, "V")
            neighbor = None
            kind = "V"
        else:
            value = expval(p_right, "v")
            neighbor = None
            kind = "v"
        results[key] = {
            "p_left": p_left, "p_right": p_right,
            "neighbor": neighbor, "value": value, "kind": kind,
        }
        if on_link is not None:
            on_link(key, results, n_total)
    return results


def get_lattice_electric_fields(ten_file, Lx, Ly, shape, chi, g, bound_state=None, R=1, precision=3, chargesx=None, chargesy=None, force_recompute=False):
    """Whole-lattice <V.v> link electric-field values (i.e.
    link_electric_fields(engine, plaquette_label=None, ...)), cached
    inside ten_file next to the tensor checkpoint it's computed from.

    Once computed for a given (config, g, chi) the values are saved and
    reused on later calls. With the TreeExpval path each link costs about
    (tree depth) x chi^3, so loading the engine dominates; the old path
    (fast=False in link_electric_fields) moved the canonical center and
    rebuilt every renormalized operator for each link.

    Set force_recompute=True to ignore an existing cache and recompute
    (e.g. the checkpoint at this g/chi was re-run/overwritten since the
    cache was last written).

    Links are saved as they are computed ("ef_links_partial"), so a run cut
    by the walltime resumes where it stopped; each link's time is printed.
    """
    from Z3_funcs.hdf5_manager import (
        link_electric_fields_exist, load_link_electric_fields,
        save_link_electric_fields,
    )

    if not force_recompute and link_electric_fields_exist(
        ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi,
        chargesx=chargesx, chargesy=chargesy,
    ):
        return load_link_electric_fields(
            ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi,
            chargesx=chargesx, chargesy=chargesy,
        )

    import time

    cache_args = (ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi)
    done = {}
    if not force_recompute and link_electric_fields_exist(
        *cache_args, chargesx=chargesx, chargesy=chargesy, name="ef_links_partial",
    ):
        done = load_link_electric_fields(
            *cache_args, chargesx=chargesx, chargesy=chargesy, name="ef_links_partial",
        )
        print(f"    resuming link fields: {len(done)} links already computed", flush=True)

    engine = load_engine_from_tensor(
        ten_file, Lx, Ly, shape, chi, g,
        bound_state=bound_state, R=R, precision=precision,
        chargesx=chargesx, chargesy=chargesy,
    )
    t_last = [time.time()]

    def save_partial(key, results, n_total):
        save_link_electric_fields(
            *cache_args, results, chargesx=chargesx, chargesy=chargesy, name="ef_links_partial",
        )
        now = time.time()
        print(f"    link {len(results)}/{n_total} {key}: {now - t_last[0]:.0f}s", flush=True)
        t_last[0] = now

    links = link_electric_fields(engine, None, Lx, Ly, shape, done=done, on_link=save_partial)
    save_link_electric_fields(
        ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi, links,
        chargesx=chargesx, chargesy=chargesy,
    )
    return links
