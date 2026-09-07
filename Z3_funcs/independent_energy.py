"""
independent_energy.py
----------------------
An independently-evaluated <psi|H|psi> for a TTN state, computed from
scratch by fully contracting bra and ket copies of the entire tree against
the raw physical Hamiltonian -- deliberately NOT using PhysicsEngine's
renormalized block_hamiltonians / edge_spin_operators caches anywhere.

Why this is a genuinely independent check (per advisor's original
suggestion): the per-sweep ref_energy already tracked in
GroundStateSearch.convergence_history comes from the local two-site
Lanczos eigenvalue at one edge, which is only equal to the true global
<psi|H|psi> if the renormalized block-Hamiltonian bookkeeping elsewhere in
GroundStateSearch.run()/PhysicsEngine.py is itself correct. The A/B test
comparing dense lanczos() vs TTNLinearOperator (ab_test_linop_vs_dense.py)
doesn't cover this either -- both eigensolvers consume the SAME shared
renormalized operators built by the same sweep loop, so a bug in that
shared bookkeeping would make both agree while both being wrong. This
module recomputes the energy a completely different way: for every
Hamiltonian term, build fresh bra/ket tensor copies and contract the WHOLE
tree from the raw physical-space operators, with no cached intermediate
results reused between terms or shared with the sweep's own machinery.

Usage:
    from Z3_funcs.independent_energy import independent_energy
    E = independent_energy(gss.psi, gss.hamiltonian)
"""

import numpy as np
import tensornetwork as tn

from ttnopt.src.Observable import bare_spin_operator


def _edge_occurrences(edges):
    """edge_id -> list of (tensor_id, leg_idx) occurrences across psi.edges.

    Internal edges appear exactly twice (once as some tensor's child leg 0
    or 1, once as another tensor's parent leg 2); physical edges and the
    top edge appear exactly once.
    """
    occ = {}
    for t, edge in enumerate(edges):
        for leg_idx, e in enumerate(edge):
            occ.setdefault(e, []).append((t, leg_idx))
    return occ


def _gauge_absorbed_tensors(psi):
    """Fresh copy of psi.tensors with psi.gauge_tensor absorbed into one of
    the two current canonical-center tensors -- without this, the tensors
    alone are missing the singular values from the last SVD.

    The canonical center is a genuine TWO-SITE form: canonical_center_edge_id
    sits at LEG 2 (not a child leg 0/1) of BOTH of the two center tensors
    simultaneously (verified directly -- e.g. positions [(28, 2), (29, 2)]
    for a fresh 5x5 tree), with gauge_tensor's singular values factored out
    of that shared bond. An earlier version of this function searched for
    the edge in position 0/1 (child slots), which never matches this
    structure, silently fell through to a no-op, and left the gauge
    unabsorbed entirely -- equivalent to treating it as the identity, which
    is what produced the norm^2=9 bug this replaces.

    This is unavoidable bookkeeping about what the state IS, not a use of
    any renormalized/cached shortcut -- distinct from the
    block_hamiltonians/edge_spin_operators caches this module deliberately
    avoids. Operates on a copy; never mutates the live psi.
    """
    tensors = [np.array(t, dtype=np.complex128, copy=True) for t in psi.tensors]
    if psi.gauge_tensor is None:
        return tensors

    selected_tensor_id = None
    for t, edge in enumerate(psi.edges):
        if edge[2] == psi.canonical_center_edge_id:
            selected_tensor_id = t
            break
    if selected_tensor_id is None:
        # Shouldn't happen -- canonical_center_edge_id should always be at
        # leg 2 of (at least) one tensor -- but be defensive rather than
        # silently returning a gauge-incomplete state.
        raise RuntimeError(
            f"independent_energy: no tensor found with canonical_center_edge_id "
            f"({psi.canonical_center_edge_id}) at leg 2 -- cannot absorb gauge_tensor."
        )

    iso = tn.Node(tensors[selected_tensor_id])
    gauge = tn.Node(np.array(psi.gauge_tensor, dtype=np.complex128))
    iso[2] ^ gauge[0]
    absorbed = tn.contractors.auto([iso, gauge], output_edge_order=[iso[0], iso[1], gauge[1]])
    tensors[selected_tensor_id] = absorbed.get_tensor()
    return tensors


def _full_sandwich(tensors, edges, physical_operators):
    """<psi| (operators in physical_operators, identity elsewhere) |psi>,
    contracted fully from scratch. physical_operators: dict physical
    edge_id -> operator matrix (numpy array). Edges not present there
    (including the top edge) are traced directly between bra and ket.
    """
    ket_nodes = [tn.Node(t) for t in tensors]
    bra_nodes = [tn.Node(np.conj(t)) for t in tensors]

    occ = _edge_occurrences(edges)
    nodes = list(ket_nodes) + list(bra_nodes)

    for edge_id, positions in occ.items():
        if edge_id in physical_operators:
            (t, leg) = positions[0]
            op_node = tn.Node(physical_operators[edge_id])
            nodes.append(op_node)
            op_node[1] ^ ket_nodes[t][leg]
            op_node[0] ^ bra_nodes[t][leg]
        elif len(positions) == 1:
            # physical edge with no operator (identity), or the top edge
            (t, leg) = positions[0]
            ket_nodes[t][leg] ^ bra_nodes[t][leg]
        else:
            # internal edge: contract within ket, and separately within bra
            (t1, l1), (t2, l2) = positions
            ket_nodes[t1][l1] ^ ket_nodes[t2][l2]
            bra_nodes[t1][l1] ^ bra_nodes[t2][l2]

    result = tn.contractors.auto(nodes, output_edge_order=[])
    return complex(result.tensor)


def independent_energy(psi, hamiltonian):
    """<psi|H|psi> / <psi|psi>, evaluated completely from scratch.

    Cost: one full from-scratch tree contraction per Hamiltonian term
    (each O(N_tensors * chi^3)), so O(N_terms * N_tensors * chi^3) total --
    much more expensive than the renormalized incremental approach used
    during sweeps, which is exactly why this is meant as an occasional
    cross-check (e.g. once per finished chi stage), not a per-sweep cost.
    """
    tensors = _gauge_absorbed_tensors(psi)
    edges = psi.edges

    norm2 = np.real(_full_sandwich(tensors, edges, {}))
    print(f"    [independent_energy] <psi|psi> = {norm2:.10f} (should be close to 1.0 "
          f"for a properly normalized state -- if not, the bug is in the generic "
          f"contraction/gauge-absorption, not the Hamiltonian-term handling)", flush=True)

    total = 0.0 + 0.0j
    for ob in hamiltonian.observables:
        for n in range(ob.operators_num):
            ops = ob.operators_list[n]
            coef = ob.coef_list[n]
            physical_operators = {}
            for idx, opname in zip(ob.indices, ops):
                op_mat = bare_spin_operator(opname, hamiltonian.spin_size[idx])
                physical_operators[idx] = op_mat if idx not in physical_operators else physical_operators[idx] @ op_mat
            total += coef * _full_sandwich(tensors, edges, physical_operators)

    return np.real(total) / norm2
