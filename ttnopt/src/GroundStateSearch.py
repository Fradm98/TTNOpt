import time
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import tensornetwork as tn

from ttnopt.src.Hamiltonian import Hamiltonian
from ttnopt.src.PhysicsEngine import PhysicsEngine
from ttnopt.src.TTN import TreeTensorNetwork


class GroundStateSearch(PhysicsEngine):
    """A class for ground state search algorithm based on DMRG.
    Args:
        psi (TreeTensorNetwork): The quantum state.
        hamiltonians (Hamiltonian): Hamiltonian which is list of Observable.
        init_bond_dim (int, optional): Initial bond dimension. Defaults to 4.
        max_bond_dim (int, optional): Maximum bond dimension. Defaults to 16.
    """

    def __init__(
        self,
        psi: TreeTensorNetwork,
        hamiltonian: Hamiltonian,
        init_bond_dim: int = 4,
        max_bond_dim: int = 16,
        energy_degeneracy_threshold: float = 1e-13,
        entanglement_degeneracy_threshold: float = 1e-8,
    ):
        """Initialize a DMRG object.

        Args:
            psi : The quantum state.
            hamiltonians : The Hamiltonian.
            init_bond_dim : Initial bond dimension.
            max_bond_dim : Maximum bond dimension.
            truncation_error : Maximum truncation error.
        """
        self.energy: Dict[int, float] = {}
        self.entanglement: Dict[int, float] = {}
        self.error: Dict[int, float] = {}
        self.one_site_expval: Dict[int, Dict[str, float]] = {}
        self.two_site_expval: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.convergence_history: list = []
        self.converged: bool = False
        self.local_update_diagnostics: list = []
        self.superblock_energy_trajectory: list = []

        super().__init__(
            psi,
            hamiltonian,
            init_bond_dim,
            max_bond_dim,
            energy_degeneracy_threshold,
            entanglement_degeneracy_threshold,
        )

    def _rayleigh_quotient(self, psi_tensor, central_tensor_ids):
        """<phi|H_eff|phi> / <phi|phi> for an arbitrary two-site tensor phi
        (not necessarily normalized or an eigenvector) at central_tensor_ids.

        Uses self._apply_ham_psi, which patch_physics_engine() never
        replaces (it only patches lanczos and _set_block_hamiltonian), so
        this gives a consistent measure regardless of whether the LinOp
        patch has been applied -- useful for the E_before/E_after_truncation
        diagnostics, which want the same yardstick on both sides of a
        Lanczos+truncation step.
        """
        psi_node = tn.Node(psi_tensor)
        h_psi = self._apply_ham_psi(psi_node, central_tensor_ids)
        flat_psi = psi_tensor.ravel()
        flat_h_psi = h_psi.tensor.ravel()
        numerator = np.real(np.vdot(flat_psi, flat_h_psi))
        denominator = np.real(np.vdot(flat_psi, flat_psi))
        return float(numerator / denominator)

    def run(
        self,
        opt_structure: int = 0,
        energy_convergence_threshold: float = 1e-8,
        entanglement_convergence_threshold: float = 1e-8,
        max_num_sweep: int = 10,
        converged_count: int = 2,
        eval_onesite_expval: bool = False,
        eval_twosite_expval: bool = False,
        temperature: float = 0.0,
        tau: int = 0,
        verbose: bool = False,
        lanczos_tol: float = None,
        lanczos_maxiter: int = None,
        lanczos_ncv: int = None,
        reference_edge_id: int = None,
        diagnostic_edges: list = None,
    ):
        """Run DMRG algorithm.

        Args:
            opt_structure (bool, optional): If optimize the tree structure or not. Defaults to False.
            energy_convergence_threshold (float, optional): Energy threshold for convergence. Defaults to 1e-8.
            entanglement_convergence_threshold (float, optional): Entanglement entropy threshold for automatic optimization. Defaults to 1e-8.
            converged_count (int, optional): Converged count. Defaults to 1.
            eval_onesite_expval (bool): If evaluate one-site expectation value or not.
            eval_twosite_expval (bool): If evaluate two-site expectation value or not.
            lanczos_tol (float, optional): Overrides the per-two-site eigensolver's
                convergence tolerance for this call (only takes effect when
                patch_physics_engine() has been applied -- the unpatched
                PhysicsEngine.lanczos() uses different parameter names and
                ignores this). Leave None to use the eigensolver's own default
                (tight, 1e-10). Meant for cheaply loosening precision during
                warm-up sweeps where the local ground state doesn't need to be
                highly accurate yet.
            lanczos_maxiter (int, optional): Same caveat as lanczos_tol, caps
                the eigensolver's Lanczos/Arnoldi iteration count.
            lanczos_ncv (int, optional): Same caveat as lanczos_tol, sets the
                Krylov subspace size (number of Lanczos vectors) eigsh uses.
                None keeps eigsh's own default sizing.
            reference_edge_id (int, optional): Which edge's own energy trace
                to record every sweep in convergence_history (ref_energy,
                ref_energy_reldiff vs the same edge one sweep ago,
                ref_truncation_error, ref_lanczos_residual) -- as opposed to
                the worst-case-over-all-edges max_energy_reldiff/max_ee_diff,
                which mixes edges visited at very different points in a
                sweep's update history. Defaults to self.psi.top_edge_id,
                which every sweep visits last (both subtrees below it are
                guaranteed already updated this sweep by the time it's
                reached), making it the least-stale single position to track.
            diagnostic_edges (list, optional): Edge ids to instrument with
                E_before -> E_Lanczos -> E_after_truncation tracking (see
                self.local_update_diagnostics). Opt-in and meant to stay a
                short list ("a few representative updates") -- each
                instrumented update costs two extra H_eff matvecs via
                _rayleigh_quotient, on top of the matvecs the eigensolver
                itself already needs. Leave None (default) for zero extra
                cost. Only opt_structure=0 is supported for this (matches
                the only opt_structure value used anywhere in this project);
                edge_order is not accounted for otherwise.
        """
        lanczos_kwargs = {}
        if lanczos_tol is not None:
            lanczos_kwargs["tol"] = lanczos_tol
        if lanczos_maxiter is not None:
            lanczos_kwargs["max_iter"] = lanczos_maxiter
        if lanczos_ncv is not None:
            lanczos_kwargs["ncv"] = lanczos_ncv
        if reference_edge_id is None:
            reference_edge_id = self.psi.top_edge_id
        diagnostic_edges = set(diagnostic_edges) if diagnostic_edges else set()
        self.local_update_diagnostics = []
        self.superblock_energy_trajectory = []
        energy_at_edge: Dict[int, float] = {}
        _energy_at_edge: Dict[int, float] = {}
        ee_at_edge: Dict[int, float] = {}
        _ee_at_edge: Dict[int, float] = {}
        _error_at_edge: Dict[int, float] = {}
        onesite_expval: Dict[int, Dict[str, float]] = {}
        twosite_expval: Dict[Tuple[int, int], Dict[str, float]] = {}

        edges, _edges = deepcopy(self.psi.edges), deepcopy(self.psi.edges)

        self.convergence_history = []
        converged_num = 0

        if tau == 0:
            tau = max_num_sweep // 2 + 1
        for sweep_num in range(max_num_sweep):
            temp = temperature * (2 ** (-sweep_num / tau))
            if converged_num > converged_count:
                break

            if verbose:
                print(f"  sweep {sweep_num + 1}/{max_num_sweep} starting...", flush=True)
            t_sweep_start = time.time()

            energy_at_edge = deepcopy(_energy_at_edge)
            ee_at_edge = deepcopy(_ee_at_edge)
            edges = deepcopy(_edges)

            self.distance = self.initial_distance()
            self.flag = self.initial_flag()

            (
                _edge_id,
                _selected_tensor_id,
                _connected_tensor_id,
                _not_selected_tensor_id,
            ) = self.local_two_tensor()

            ref_lanczos_residual_this_sweep = np.nan
            ref_lanczos_retried_this_sweep = False
            num_lanczos_retries_this_sweep = 0

            # print("Sweep count: " + str(sweep_num + 1))
            while True:
                edge_id = _edge_id
                selected_tensor_id = _selected_tensor_id
                connected_tensor_id = _connected_tensor_id
                not_selected_tensor_id = _not_selected_tensor_id
                # absorb gauge tensor
                iso = tn.Node(self.psi.tensors[selected_tensor_id])
                gauge = tn.Node(self.psi.gauge_tensor)
                iso[2] ^ gauge[0]
                iso = tn.contractors.auto(
                    [iso, gauge], output_edge_order=[iso[0], iso[1], gauge[1]]
                )
                self.psi.tensors[selected_tensor_id] = iso.get_tensor()

                self.set_ttn_properties_at_one_tensor(edge_id, selected_tensor_id)

                self._set_edge_spin(not_selected_tensor_id)

                self._set_block_hamiltonian(not_selected_tensor_id)

                ground_state_order = [selected_tensor_id, connected_tensor_id]

                instrument_this_update = edge_id in diagnostic_edges
                if instrument_this_update:
                    pre_lanczos_1 = tn.Node(self.psi.tensors[selected_tensor_id])
                    pre_lanczos_2 = tn.Node(self.psi.tensors[connected_tensor_id])
                    pre_lanczos_1[2] ^ pre_lanczos_2[2]
                    pre_lanczos_state = tn.contractors.auto(
                        [pre_lanczos_1, pre_lanczos_2],
                        output_edge_order=[
                            pre_lanczos_1[0],
                            pre_lanczos_1[1],
                            pre_lanczos_2[0],
                            pre_lanczos_2[1],
                        ],
                    ).get_tensor()
                    e_before = self._rayleigh_quotient(pre_lanczos_state, ground_state_order)

                ground_state, energy = self.lanczos(ground_state_order, **lanczos_kwargs)
                # Unconditional (unlike local_update_diagnostics) -- this is
                # exactly the eigenvalue lanczos() already returns, so
                # recording it costs nothing extra (no additional matvecs),
                # for every single superblock diagonalization in the whole
                # run. Gives a continuous energy-vs-cumulative-update-index
                # trajectory across sweep boundaries, as opposed to
                # convergence_history's one-point-per-sweep summary.
                self.superblock_energy_trajectory.append(
                    {"sweep": sweep_num + 1, "edge_id": edge_id, "energy": energy}
                )
                # last_lanczos_retried is set by ttn_eigensolver (only when
                # patch_physics_engine() has been applied) and is reset to
                # False at the top of every ttn_eigensolver() call, so
                # checking it right here after each edge's own lanczos()
                # call is safe -- it can't leak a stale True from an earlier
                # edge in this same sweep.
                if getattr(self, "last_lanczos_retried", False):
                    num_lanczos_retries_this_sweep += 1
                if edge_id == reference_edge_id:
                    # last_lanczos_residual is set by ttn_eigensolver (only
                    # when patch_physics_engine() has been applied) and gets
                    # overwritten on every call, so it must be captured here,
                    # right after this edge's own lanczos() call -- reading it
                    # at the end of the sweep would just give whichever edge
                    # happened to be visited last, not this one.
                    ref_lanczos_residual_this_sweep = getattr(
                        self, "last_lanczos_residual", np.nan
                    )
                    ref_lanczos_retried_this_sweep = getattr(
                        self, "last_lanczos_retried", False
                    )
                psi_edges = (
                    self.psi.edges[selected_tensor_id][:2]
                    + self.psi.edges[connected_tensor_id][:2]
                )

                u, s, v, probability, error, edge_order = self.decompose_two_tensors(
                    ground_state,
                    self.max_bond_dim,
                    opt_structure=opt_structure,
                    temperature=temp,
                    operate_degeneracy=True,
                    epsilon=entanglement_convergence_threshold,
                    delta=self.entanglement_degeneracy_threshold,
                )

                if instrument_this_update:
                    # Reconstruct the (possibly truncated) two-site tensor
                    # from u/s/v and re-evaluate the same Rayleigh quotient,
                    # to see how much of Lanczos's improvement survives the
                    # SVD truncation back to max_bond_dim. Valid for
                    # opt_structure=0 only, where edge_order==[0,1,2,3] so
                    # u/v's leg order already matches ground_state_order.
                    u_node = tn.Node(u)
                    s_node = tn.Node(s)
                    v_node = tn.Node(v)
                    u_node[2] ^ s_node[0]
                    s_node[1] ^ v_node[2]
                    reconstructed = tn.contractors.auto(
                        [u_node, s_node, v_node],
                        output_edge_order=[u_node[0], u_node[1], v_node[0], v_node[1]],
                    ).get_tensor()
                    e_after_truncation = self._rayleigh_quotient(
                        reconstructed, ground_state_order
                    )
                    self.local_update_diagnostics.append(
                        {
                            "sweep": sweep_num + 1,
                            "edge_id": edge_id,
                            "E_before": e_before,
                            "E_lanczos": energy,
                            "E_after_truncation": e_after_truncation,
                        }
                    )

                self.psi.tensors[selected_tensor_id] = u
                self.psi.tensors[connected_tensor_id] = v
                self.psi.gauge_tensor = s
                (
                    self.psi.edges[selected_tensor_id][0],
                    self.psi.edges[selected_tensor_id][1],
                ) = (
                    psi_edges[edge_order[0]],
                    psi_edges[edge_order[1]],
                )
                (
                    self.psi.edges[connected_tensor_id][0],
                    self.psi.edges[connected_tensor_id][1],
                ) = (
                    psi_edges[edge_order[2]],
                    psi_edges[edge_order[3]],
                )

                self.distance = self.initial_distance()
                _energy_at_edge[self.psi.canonical_center_edge_id] = energy
                # print(energy)
                ee = self.entanglement_entropy(probability)
                _ee_at_edge[self.psi.canonical_center_edge_id] = ee
                ee_dict = self.entanglement_entropy_at_physical_bond(
                    ground_state, psi_edges
                )
                for key in ee_dict.keys():
                    _ee_at_edge[key] = ee_dict[key]
                _error_at_edge[self.psi.canonical_center_edge_id] = error

                if self.candidate_edge_ids() == []:
                    break

                (
                    _edge_id,
                    _selected_tensor_id,
                    _connected_tensor_id,
                    _not_selected_tensor_id,
                ) = self.local_two_tensor()
                self.set_flag(_not_selected_tensor_id)
                # eval expval
                if self.flag[_not_selected_tensor_id]:
                    if eval_onesite_expval:
                        onesite_expval_dict = self.expval_onesite(
                            _not_selected_tensor_id,
                            ground_state,
                            ground_state_order,
                        )
                        for key in onesite_expval_dict.keys():
                            onesite_expval[key] = onesite_expval_dict[key]
                    if eval_twosite_expval:
                        twosite_expval_dict = self.expval_twosite(
                            _not_selected_tensor_id,
                            ground_state,
                            ground_state_order,
                        )
                        for key in twosite_expval_dict.keys():
                            twosite_expval[key] = twosite_expval_dict[key]

            if eval_onesite_expval:
                for i in self.psi.central_tensor_ids():
                    onesite_expval_dict = self.expval_onesite(
                        i, ground_state, ground_state_order
                    )
                    for key in onesite_expval_dict.keys():
                        onesite_expval[key] = onesite_expval_dict[key]
            if eval_twosite_expval:
                for i in self.psi.central_tensor_ids():
                    twosite_expval_dict = self.expval_twosite(
                        i, ground_state, ground_state_order
                    )
                    for key in twosite_expval_dict.keys():
                        twosite_expval[key] = twosite_expval_dict[key]
                twosite_expval_dict = self.expval_twosite_origin(
                    twosite_expval.keys(), ground_state, ground_state_order
                )
                for key in twosite_expval_dict.keys():
                    twosite_expval[key] = twosite_expval_dict[key]

            _edges = deepcopy(self.psi.edges)
            sweep_elapsed = time.time() - t_sweep_start

            sweep_num += 1
            if sweep_num <= 2:
                if verbose:
                    print(f"  sweep {sweep_num}/{max_num_sweep} done in {sweep_elapsed:.1f}s "
                          f"(no diff yet, need >=3 sweeps of history)", flush=True)
            if sweep_num > 2:
                diff_energy = [
                    np.abs(1 - _energy_at_edge[key] / energy_at_edge[key])
                    for key in energy_at_edge.keys()
                ]
                diff_ee = [
                    np.abs(ee_at_edge[key] - _ee_at_edge[key])
                    for key in ee_at_edge.keys()
                ]
                structure_unchanged = all(
                    [
                        set(edge[:2]) == set(_edge[:2]) and edge[2] == _edge[2]
                        for edge, _edge in zip(edges, _edges)
                    ]
                )
                converged_this_sweep = (
                    structure_unchanged
                    and all(
                        [
                            energy < energy_convergence_threshold
                            for energy in diff_energy
                        ]
                    )
                    and all(
                        [ee < entanglement_convergence_threshold for ee in diff_ee]
                    )
                )
                # Reference-edge-only trace: the SAME edge's own value this
                # sweep vs. one sweep ago, as opposed to max_energy_reldiff/
                # max_ee_diff above which are the worst case over ALL edges
                # (a mix of edges visited at very different points in the
                # sweep's update history -- see professor's diagnostic notes).
                if (
                    reference_edge_id in _energy_at_edge
                    and reference_edge_id in energy_at_edge
                    and energy_at_edge[reference_edge_id] != 0
                ):
                    ref_energy_reldiff = float(
                        np.abs(
                            1
                            - _energy_at_edge[reference_edge_id]
                            / energy_at_edge[reference_edge_id]
                        )
                    )
                else:
                    ref_energy_reldiff = float("nan")
                self.convergence_history.append(
                    {
                        "sweep": sweep_num,
                        "max_energy_reldiff": float(np.max(diff_energy)),
                        "max_ee_diff": float(np.max(diff_ee)),
                        "converged_this_sweep": converged_this_sweep,
                        "ref_edge": reference_edge_id,
                        "ref_energy": float(
                            _energy_at_edge.get(reference_edge_id, float("nan"))
                        ),
                        "ref_energy_reldiff": ref_energy_reldiff,
                        "ref_truncation_error": float(
                            _error_at_edge.get(reference_edge_id, float("nan"))
                        ),
                        "ref_lanczos_residual": float(ref_lanczos_residual_this_sweep),
                        "ref_lanczos_retried": bool(ref_lanczos_retried_this_sweep),
                        "num_lanczos_retries": num_lanczos_retries_this_sweep,
                    }
                )
                if verbose:
                    rec = self.convergence_history[-1]
                    print(
                        f"  sweep {sweep_num}/{max_num_sweep} done in {sweep_elapsed:.1f}s: "
                        f"max|dE/E|={rec['max_energy_reldiff']:.3e}, "
                        f"max|d(ee)|={rec['max_ee_diff']:.3e}, "
                        f"converged_this_sweep={converged_this_sweep} | "
                        f"ref_edge={rec['ref_edge']}: E={rec['ref_energy']:.10f}, "
                        f"reldiff={rec['ref_energy_reldiff']:.3e}, "
                        f"trunc_err={rec['ref_truncation_error']:.3e}, "
                        f"lanczos_resid={rec['ref_lanczos_residual']:.3e}, "
                        f"ref_retried={rec['ref_lanczos_retried']}, "
                        f"sweep_retries={rec['num_lanczos_retries']}",
                        flush=True,
                    )
                if converged_this_sweep:
                    converged_num += 1

        self.energy = _energy_at_edge
        self.entanglement = _ee_at_edge
        self.error = _error_at_edge
        self.one_site_expval = onesite_expval
        self.two_site_expval = twosite_expval
        self.converged = converged_num > converged_count
        return 0
