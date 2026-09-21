"""
TTNLinearOperator.py
--------------------
Alternative two-site eigensolver for PhysicsEngine:

  - TTNHamiltonianOperator : scipy LinearOperator wrapping the on-the-fly
                             H|psi> application at the two-site canonical center
  - ttn_eigensolver        : replaces PhysicsEngine.lanczos() entirely
  - patch_physics_engine   : monkey-patches an existing PhysicsEngine instance

What this actually buys
-----------------------
NOT block-Hamiltonian storage: PhysicsEngine._set_block_hamiltonian already
stores each edge's block Hamiltonian as a (χ, χ) matrix, and this module reads
those same matrices. The difference is purely in the eigensolver.

The original PhysicsEngine.lanczos() is a hand-rolled two-pass Lanczos: it
tridiagonalizes without storing the Krylov basis, then REGENERATES every
Krylov vector in a second pass to reconstruct the eigenvector (so every
H|psi> is computed twice), and then runs an inverse-iteration refinement loop
until ||H|v> - e|v>|| < inverse_tol, which is unbounded in iteration count.
ttn_eigensolver replaces all of that with ARPACK's implicitly-restarted
Lanczos (scipy eigsh), which needs one matvec per iteration and a bounded
ncv-vector basis, and exposes tol/maxiter/ncv so warm-up sweeps can be run
cheaply.

Both paths share the SAME renormalized operators and block Hamiltonians built
by the sweep loop, so they must agree elementwise on H|psi>, not merely on the
resulting energy -- see diagnostics/test_linop_matches_dense_matvec.py.

Usage
-----
Call patch_physics_engine(engine) once after construction. The public API that
GroundStateSearch.run() calls is unchanged:
    ground_state, energy = engine.lanczos(ground_state_order)
"""

from typing import List, Optional

import numpy as np
import tensornetwork as tn
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

from ttnopt.src.functionTTN import get_bare_edges


# ---------------------------------------------------------------------------
# 1.  Core matvec — computes H|v> without storing any (χ,χ,χ,χ) object
# ---------------------------------------------------------------------------

def _prep_leg(v_tensor: np.ndarray, leg: int, shape: tuple) -> np.ndarray:
    """Move `leg` to the front and flatten the rest, ready for `H @ ...`.

    For leg == 0 this is a free view (already C-contiguous); for leg in
    {1,2,3} the reshape after moveaxis forces a real (χ_leg, rest) copy —
    that copy is the one part of _block_ham_matvec worth avoiding when the
    same v/leg pair is reused across multiple Hamiltonian terms (see the
    v_prepped cache in _apply_ham_psi_matvec).
    """
    chi_leg = shape[leg]
    return np.moveaxis(v_tensor, leg, 0).reshape(chi_leg, -1)


def _unprep_leg(mat: np.ndarray, leg: int, shape: tuple) -> np.ndarray:
    """Inverse of _prep_leg: restore the original (χ0,χ1,χ2,χ3) axis order."""
    chi_leg = shape[leg]
    result = mat.reshape((chi_leg,) + tuple(s for i, s in enumerate(shape) if i != leg))
    return np.moveaxis(result, 0, leg)


def _block_ham_matvec(
    v_tensor: np.ndarray,
    block_ham_matrix: np.ndarray,
    leg: int,
    shape: tuple,
) -> np.ndarray:
    """
    Apply a block Hamiltonian to one leg of a 4-leg tensor via matmul.

    Parameters
    ----------
    v_tensor      : ndarray of shape `shape`  (χ0, χ1, χ2, χ3)
    block_ham_matrix : ndarray of shape (χ_leg, χ_leg)  — the block H as a matrix
    leg           : which axis (0-3) to act on
    shape         : tuple describing v_tensor's shape

    Returns
    -------
    result of shape `shape`
    """
    v_moved = _prep_leg(v_tensor, leg, shape)
    # .T because every renormalized operator in this engine is stored with
    # index 0 on the un-conjugated-tensor side and index 1 on the conjugated
    # side (see PhysicsEngine._set_edge_spin / _set_block_hamiltonian, both
    # emitting output_edge_order=[bra[2], ket[2]]). Index 0 is therefore the
    # one the dense consumers contract with psi (_block_ham_psi does
    # `psi_[apply_id] ^ h[0]`). Dropping the .T here silently applies H^T,
    # which has the same spectrum for Hermitian H and so hides in any
    # energy-only comparison -- but is wrong for any complex Hamiltonian.
    result = block_ham_matrix.T @ v_moved
    return _unprep_leg(result, leg, shape)


def _compute_block_ham_matrix(engine, edge_id: int) -> Optional[np.ndarray]:
    """
    The block Hamiltonian for `edge_id` as a (χ, χ) matrix, or None if this
    edge carries no block-Hamiltonian contribution.

    PhysicsEngine._set_block_hamiltonian already stores these as (χ, χ) --
    it contracts the rank-4 two-leg block Hamiltonian from
    _get_block_hamiltonian down against the isometry and emits
    output_edge_order=[bra[2], ket[2]] -- so this is a plain lookup, with
    index 0 on the un-conjugated-tensor side.
    """
    return engine.block_hamiltonians.get(edge_id)


def _apply_ham_psi_matvec(
    v_flat: np.ndarray,
    engine,
    central_tensor_ids: List[int],
    psi_shape: tuple,
) -> np.ndarray:
    """
    The core matvec: given the two-site state as a flat vector, compute H|psi>
    and return it flat.  No (χ,χ,χ,χ) intermediate is ever formed or stored.

    The 4-leg tensor has legs ordered as:
        leg 0 : edges[central_tensor_ids[0]][0]   (left  child of tensor 0)
        leg 1 : edges[central_tensor_ids[0]][1]   (right child of tensor 0)
        leg 2 : edges[central_tensor_ids[1]][0]   (left  child of tensor 1)
        leg 3 : edges[central_tensor_ids[1]][1]   (right child of tensor 1)
    """
    v = v_flat.reshape(psi_shape)
    result = np.zeros(psi_shape, dtype=np.complex128)

    tid0, tid1 = central_tensor_ids
    edge_legs = [
        engine.psi.edges[tid0][0],  # leg 0
        engine.psi.edges[tid0][1],  # leg 1
        engine.psi.edges[tid1][0],  # leg 2
        engine.psi.edges[tid1][1],  # leg 3
    ]

    # v's "leg moved to front, flattened to a matrix" form, memoized per leg.
    # Multiple Hamiltonian terms below apply different operators to the SAME
    # leg of the SAME original v (Part A's one block-Ham term per leg, and
    # every Part-B term whose leg_a matches) -- v itself never changes within
    # this call, so the moveaxis+reshape (the one genuinely non-view, copying
    # step in _block_ham_matvec) only needs to happen once per leg actually
    # used, not once per term. Profiling on a real run (Lx=5,Ly=5, chi=27)
    # showed this reshape alone costing ~240s of a ~1390s matvec-bound sweep.
    v_prepped = {}

    def get_v_prepped(leg):
        prepped = v_prepped.get(leg)
        if prepped is None:
            prepped = _prep_leg(v, leg, psi_shape)
            v_prepped[leg] = prepped
        return prepped

    # ------------------------------------------------------------------
    # Part A: block Hamiltonian terms (one per leg)
    # Each is a (χ_leg × χ_leg) matrix acting on that leg only.
    # Memory: O(χ²) per term, discarded after use.
    # ------------------------------------------------------------------
    for leg, edge_id in enumerate(edge_legs):
        H_mat = _compute_block_ham_matrix(engine, edge_id)
        if H_mat is not None:
            # .T for the same index-convention reason as in _block_ham_matvec.
            result += _unprep_leg(H_mat.T @ get_v_prepped(leg), leg, psi_shape)

    # ------------------------------------------------------------------
    # Part B: two-body interaction terms connecting pairs of legs
    # Uses edge_spin_operators already stored as (χ,χ) matrices — fine.
    # This mirrors _ham_psi but operates directly on the reshaped v.
    # ------------------------------------------------------------------
    # The 6 distinct pairs of legs in a 4-leg tensor:
    leg_pairs = [
        (0, 1, engine.psi.edges[tid0][:2]),        # within tensor 0
        (2, 3, engine.psi.edges[tid1][:2]),        # within tensor 1
        (0, 2, [engine.psi.edges[tid0][0], engine.psi.edges[tid1][0]]),
        (0, 3, [engine.psi.edges[tid0][0], engine.psi.edges[tid1][1]]),
        (1, 2, [engine.psi.edges[tid0][1], engine.psi.edges[tid1][0]]),
        (1, 3, [engine.psi.edges[tid0][1], engine.psi.edges[tid1][1]]),
    ]

    for leg_a, leg_b, (edge_a, edge_b) in leg_pairs:
        l_bare = get_bare_edges(edge_a, engine.psi.edges, engine.psi.physical_edges)
        r_bare = get_bare_edges(edge_b, engine.psi.edges, engine.psi.physical_edges)

        # Paper Sec. 3.2.2 ("we adjust the order of summation of spin
        # operators"): every term sharing the same partner (site, operator)
        # is summed into ONE matrix on the larger side before being applied,
        # so the number of O(chi^5) two-operator applications is set by the
        # smaller region rather than by the raw term count. Grouping side is
        # chosen by region size, the paper's g' > g criterion -- same rule the
        # dense _ham_psi uses via `len(l_bare_edges) > len(r_bare_edges)`.
        group_left = len(l_bare) > len(r_bare)
        grouped_leg, partner_leg = (leg_a, leg_b) if group_left else (leg_b, leg_a)
        grouped_edge, partner_edge = (edge_a, edge_b) if group_left else (edge_b, edge_a)

        accum = {}
        partner_ops = {}

        for ham in engine.hamiltonian.observables:
            if len(ham.indices) != 2:
                continue

            # Check if this Hamiltonian term connects edge_a ↔ edge_b
            if ham.indices[0] in l_bare and ham.indices[1] in r_bare:
                idx_l, idx_r = ham.indices[0], ham.indices[1]
                flip = False
            elif ham.indices[1] in l_bare and ham.indices[0] in r_bare:
                idx_l, idx_r = ham.indices[1], ham.indices[0]
                flip = True
            else:
                continue

            for n in range(ham.operators_num):
                ops = ham.operators_list[n]
                coef = ham.coef_list[n]

                op_l_name = ops[1] if flip else ops[0]
                op_r_name = ops[0] if flip else ops[1]

                if group_left:
                    g_idx, g_name = idx_l, op_l_name
                    p_idx, p_name = idx_r, op_r_name
                else:
                    g_idx, g_name = idx_r, op_r_name
                    p_idx, p_name = idx_l, op_l_name

                # `coef * op` always allocates, so the engine's stored
                # operator is never mutated by the accumulation below.
                contrib = coef * engine._spin_operator_at_edge(
                    grouped_edge, g_idx, g_name
                )
                key = (p_idx, p_name)
                if key in accum:
                    accum[key] = accum[key] + contrib
                else:
                    accum[key] = contrib
                    partner_ops[key] = engine._spin_operator_at_edge(
                        partner_edge, p_idx, p_name
                    )

        for key, grouped_op in accum.items():
            # .T for the index convention (see _block_ham_matvec). Summing
            # then transposing is identical to transposing then summing, so
            # the accumulation above can stay untransposed.
            tmp = _unprep_leg(
                grouped_op.T @ get_v_prepped(grouped_leg), grouped_leg, psi_shape
            )
            # tmp is a fresh per-key intermediate (not v), so there's nothing
            # to cache on this second application.
            result += _block_ham_matvec(tmp, partner_ops[key], partner_leg, psi_shape)

    return result.ravel()


# ---------------------------------------------------------------------------
# 2.  LinearOperator subclass — mirrors your TensorMultiplierOperator exactly
# ---------------------------------------------------------------------------

class TTNHamiltonianOperator(LinearOperator):
    """
    Scipy LinearOperator representing H_eff at the two-site canonical center
    of a TTN.

    Shape: (χ⁴, χ⁴)  where χ = max bond dimension at the canonical center.
    The operator never forms or stores a (χ,χ,χ,χ) block Hamiltonian.
    All intermediate tensors are O(χ³) and are discarded after each matvec.

    Parameters
    ----------
    engine              : PhysicsEngine instance
    central_tensor_ids  : [id0, id1] as returned by local_two_tensor()
    """

    def __init__(self, engine, central_tensor_ids: List[int]):
        self.engine = engine
        self.central_tensor_ids = central_tensor_ids

        tid0, tid1 = central_tensor_ids
        t0 = engine.psi.tensors[tid0]
        t1 = engine.psi.tensors[tid1]
        # Shape of the two-site state: (d0, d1, d2, d3)
        self.psi_shape = (t0.shape[0], t0.shape[1], t1.shape[0], t1.shape[1])
        dim = int(np.prod(self.psi_shape))

        super().__init__(dtype=np.complex128, shape=(dim, dim))

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        return _apply_ham_psi_matvec(
            v, self.engine, self.central_tensor_ids, self.psi_shape
        )

    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        # rmatvec is A^H @ v, and H_eff is Hermitian, so it IS matvec.
        # (conj(matvec(conj(v))) would be A^T @ v -- equal only for real A.)
        return self._matvec(v)


# ---------------------------------------------------------------------------
# 3.  Drop-in replacement for PhysicsEngine.lanczos()
# ---------------------------------------------------------------------------

# Hard cap on the ArpackNoConvergence retry's Krylov workspace. Uncapped, the
# retry uses ncv=80 regardless of dim -- fine at chi<=80, but at chi=150
# (dim=chi**4) that's ncv=80 * dim * 16 bytes =~ 650GB for the basis alone,
# more RAM than any single node on this cluster physically has (~376GB max).
# No amount of PBS memory request can fix a request that exceeds a node's
# physical RAM. Tune this to comfortably fit under whatever you request for
# a given run.
MAX_RETRY_WORKSPACE_GB = 300

def ttn_eigensolver(
    engine,
    central_tensor_ids: List[int],
    num_eigvals: int = 1,
    max_iter: int = 300,
    tol: float = 1e-10,
    ncv: int = None,
) -> tuple:
    """
    Replaces PhysicsEngine.lanczos().

    Uses scipy.sparse.linalg.eigsh with a LinearOperator — identical interface
    to your MPS eigensolver.  No custom Lanczos loop, no accumulation of
    O(χ⁴) Krylov vectors.

    Parameters
    ----------
    engine             : PhysicsEngine instance
    central_tensor_ids : [id0, id1]
    num_eigvals        : number of eigenvalues to compute (default 1)
    max_iter           : max Lanczos iterations passed to eigsh
    tol                : convergence tolerance
    ncv                : size of the Krylov subspace (number of Lanczos
                         vectors) passed to eigsh. None (default) lets scipy
                         pick its own default (min(dim, max(2k+1, 20))) --
                         same behavior as before this parameter existed.
                         Larger ncv can resolve near-degenerate spectra more
                         reliably (this is exactly what the
                         ArpackNoConvergence retry below already does
                         automatically) at the cost of more memory
                         (ncv * dim * 16 bytes for complex128) and more work
                         per restart.

    Returns
    -------
    (ground_state_node, energy)  — same as the original lanczos()
    """
    tid0, tid1 = central_tensor_ids

    # Build initial guess: by the time lanczos() is called, the gauge has
    # already been absorbed into tensors[tid0] by the sweep loop (lines 116-122
    # of GroundStateSearch.run). The two central tensors are connected on leg 2.
    psi_1 = tn.Node(engine.psi.tensors[tid0])
    psi_2 = tn.Node(engine.psi.tensors[tid1])
    psi_1[2] ^ psi_2[2]
    psi0 = tn.contractors.auto(
        [psi_1, psi_2],
        output_edge_order=[psi_1[0], psi_1[1], psi_2[0], psi_2[1]],
    )
    v0 = psi0.tensor.ravel()
    v0 = v0 / np.linalg.norm(v0)

    # Build the LinearOperator — no (χ,χ,χ,χ) allocation happens here
    H_op = TTNHamiltonianOperator(engine, central_tensor_ids)

    # Reset every call so a stale True from a previous edge's retry can never
    # leak into this edge's diagnostics (GroundStateSearch.run() reads this
    # right after each self.lanczos() call).
    engine.last_lanczos_retried = False

    if H_op.shape[0] <= 2:
        # Tiny subspace: dense fallback (same guard as original lanczos)
        H_dense = H_op @ np.eye(H_op.shape[0], dtype=np.complex128)
        evals, evecs = np.linalg.eigh(H_dense)
        energy = float(np.real(evals[0]))
        v = evecs[:, 0]
        ground_state = tn.Node(v.reshape(H_op.psi_shape))
        engine.last_lanczos_residual = float(
            np.linalg.norm(H_dense @ v - energy * v)
        )
        return ground_state, energy

    try:
        eigenvalues, eigenvectors = eigsh(
            H_op,
            k=num_eigvals,
            which="SA",
            v0=v0,
            tol=tol,
            maxiter=max_iter,
            ncv=ncv,
        )
    except ArpackNoConvergence:
        # ARPACK's implicitly-restarted Lanczos stalled before reaching `tol`
        # within `max_iter` restarts -- typically because the local block
        # spectrum has near-degenerate low-lying states (common in this
        # gauge-theory model) and the default Krylov subspace (ncv, scipy's
        # min(dim, max(2k+1,20))) was too small to resolve them. Retry once
        # with a much larger subspace and iteration budget instead of
        # crashing the whole sweep/g-scan; this is the same local problem,
        # just given more room to converge.
        dim = H_op.shape[0]
        uncapped_ncv_retry = min(dim, max(4 * num_eigvals + 1, 80))
        # -4 is a rough accounting for ARPACK's own workd (~3*dim) and resid
        # (~dim) work arrays alongside the ncv-vector basis -- not exact, but
        # comfortably conservative rather than precise.
        bytes_per_vector = dim * 16  # complex128
        budget_vectors = int(MAX_RETRY_WORKSPACE_GB * 1e9 / bytes_per_vector) - 4
        ncv_retry = max(2 * num_eigvals + 1, min(uncapped_ncv_retry, budget_vectors))
        capped = ncv_retry < uncapped_ncv_retry
        # This used to be silent -- at chi=100 (dim=chi**4) ncv_retry=80
        # means ~80 Krylov vectors of dim floats/complex128 live at once
        # (~128GB at chi=100), which is the single biggest memory/timing
        # risk in this solver and was previously invisible in the logs.
        print(
            f"    [ttn_eigensolver] ArpackNoConvergence at central_tensor_ids="
            f"{central_tensor_ids} (dim={dim}, tol={tol:.1e}, maxiter={max_iter}) "
            f"-- retrying with ncv={ncv_retry}, maxiter={max_iter * 5}"
            + (
                f" [capped from {uncapped_ncv_retry} by MAX_RETRY_WORKSPACE_GB="
                f"{MAX_RETRY_WORKSPACE_GB}GB -- may not resolve this update]"
                if capped else ""
            ),
            flush=True,
        )
        engine.last_lanczos_retried = True
        eigenvalues, eigenvectors = eigsh(
            H_op,
            k=num_eigvals,
            which="SA",
            v0=v0,
            tol=tol,
            maxiter=max_iter * 5,
            ncv=ncv_retry,
        )

    energy = float(np.real(eigenvalues[0]))
    v = np.array(eigenvectors[:, 0], dtype=np.complex128)
    ground_state = tn.Node(v.reshape(H_op.psi_shape))
    # ||H_eff|psi> - E|psi>|| -- direct check on whether eigsh actually solved
    # the local problem accurately, independent of its own reported tol. Not
    # used for anything internally; stashed as an attribute (not a return
    # value) so the unpatched PhysicsEngine.lanczos() and all its callers are
    # untouched. GroundStateSearch.run() reads it via getattr(..., None).
    engine.last_lanczos_residual = float(np.linalg.norm(H_op.matvec(v) - energy * v))
    return ground_state, energy


# ---------------------------------------------------------------------------
# 4.  Monkey-patch helper
# ---------------------------------------------------------------------------

def patch_physics_engine(engine):
    """
    Patch an existing PhysicsEngine instance in-place:

      - engine.lanczos() → ttn_eigensolver()

    Nothing else is replaced. In particular _set_block_hamiltonian,
    _set_edge_spin, _ham_psi and _block_ham_psi stay exactly as
    PhysicsEngine defines them, so the renormalized-operator bookkeeping is
    shared verbatim between the patched and unpatched paths and the two
    differ only in how the local two-site eigenproblem is solved.

    Call this once after constructing your PhysicsEngine / GroundStateSearch:

        gss = GroundStateSearch(psi, hamiltonian, ...)
        patch_physics_engine(gss)
        gss.run(...)
    """
    engine.lanczos = lambda central_tensor_ids, **kw: ttn_eigensolver(
        engine, central_tensor_ids, **kw
    )
    return engine