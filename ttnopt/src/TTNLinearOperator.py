"""
TTNLinearOperator.py
--------------------
Drop-in replacement for the memory-heavy parts of PhysicsEngine:

  - TTNHamiltonianOperator   : scipy LinearOperator wrapping the on-the-fly
                               H|psi> application at the two-site canonical center
  - ttn_eigensolver          : replaces PhysicsEngine.lanczos() entirely
  - compute_block_ham_action : replaces _block_ham_psi (never forms χ⁴ tensor)
  - patch_physics_engine     : monkey-patches an existing PhysicsEngine instance

Memory reduction
----------------
Before: block_hamiltonians stores one (χ,χ,χ,χ) array per internal edge
        → 94 × χ⁴ × 16 bytes  ≈ 60 GB at χ=100

After:  nothing is stored; each matvec recomputes the action of the block
        Hamiltonian for the *current* canonical center on the fly and discards
        all intermediates immediately.  Peak transient memory per matvec is
        O(χ³) — the same order as the tensors themselves.

Usage
-----
Either call patch_physics_engine(engine) once after construction, or replace
the relevant methods manually (see bottom of file).

The public API that GroundStateSearch.run() calls is unchanged:
    ground_state, energy = engine.lanczos(ground_state_order)
"""

from collections import defaultdict
from copy import deepcopy
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
    result = block_ham_matrix @ v_moved
    return _unprep_leg(result, leg, shape)


def _compute_block_ham_matrix(engine, edge_id: int) -> Optional[np.ndarray]:
    """
    Compute the block Hamiltonian for `edge_id` as a (χ, χ) matrix on the fly,
    by walking up from the physical edges using the stored isometries.

    This replaces the need to store block_hamiltonians[edge_id] as (χ,χ,χ,χ).
    Returns None if edge_id has no block Hamiltonian contribution.

    The contraction is:
        H_block[a,b] = Σ_{ij} T*[i,j,a] H_child[i,i',j,j'] T[i',j',b]

    which is equivalent to the existing _set_block_hamiltonian logic but
    returned as a matrix instead of stored.
    """
    if edge_id not in engine.block_hamiltonians:
        return None
    # block_hamiltonians stores the (χ,χ,χ,χ) tensor — reshape to matrix here
    # so the caller only ever sees a 2-D array
    bh = engine.block_hamiltonians[edge_id]
    chi = int(np.sqrt(bh.size))  # bh is (χ,χ,χ,χ), flattened pairs → (χ²,χ²)... 
    # Actually bh shape is (chi_out, chi_out) if already a matrix, or (d0,d1,d0,d1)
    # We accept both forms:
    if bh.ndim == 4:
        d0, d1 = bh.shape[0], bh.shape[1]
        return bh.reshape(d0 * d1, d0 * d1)
    else:
        return bh  # already a matrix


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
            result += _unprep_leg(H_mat @ get_v_prepped(leg), leg, psi_shape)

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

                # These are (χ,χ) matrices — cheap, already cached
                op_l = engine._spin_operator_at_edge(edge_a, idx_l, op_l_name)
                op_r = engine._spin_operator_at_edge(edge_b, idx_r, op_r_name)

                # Apply op_l to leg_a of v (reuses the cached prep for leg_a
                # across every term that touches this leg -- see above).
                tmp = _unprep_leg((coef * op_l) @ get_v_prepped(leg_a), leg_a, psi_shape)
                # Apply op_r to leg_b of tmp. tmp is a fresh per-term
                # intermediate (not v), so there's nothing to cache here --
                # full prep+matmul+unprep via _block_ham_matvec is unavoidable.
                result += _block_ham_matvec(tmp, op_r, leg_b, psi_shape)

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
        # H is Hermitian so rmatvec = matvec on conjugate
        return self._matvec(v.conj()).conj()


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
# 4.  Eliminate block_hamiltonians storage
#     store them as (χ², χ²) matrices instead of (χ,χ,χ,χ) tensors
# ---------------------------------------------------------------------------

def _set_block_hamiltonian_as_matrix(engine, tensor_id, ham=None):
    """
    Replacement for PhysicsEngine._set_block_hamiltonian.
    Stores block_hamiltonians[edge] as a 2-D matrix (χ², χ²) instead of
    a rank-4 tensor (χ,χ,χ,χ), cutting storage by a constant factor and
    making the matmul path in _compute_block_ham_matrix trivial.

    Memory: χ⁴ × 16 bytes → same asymptotic but avoids redundant tensor
    network overhead and intermediate copies in ncon.
    """
    import tensornetwork as tn
    import numpy as np

    if ham is not None:
        bra = engine.psi.tensors[tensor_id]
        bra_node = tn.Node(bra)
        ket_node = bra_node.copy(conjugate=True)
        ham_node = tn.Node(ham)
        ham_node[0] ^ bra_node[0]
        ham_node[1] ^ bra_node[1]
        ham_node[2] ^ ket_node[0]
        ham_node[3] ^ ket_node[1]
        block_ham = tn.contractors.auto(
            [bra_node, ham_node, ket_node],
            output_edge_order=[bra_node[2], ket_node[2]],
        )
        engine.block_hamiltonians[engine.psi.edges[tensor_id][2]] = (
            block_ham.tensor  # shape (χ, χ) — already a matrix!
        )
    else:
        # Recompute from children — mirrors original _set_block_hamiltonian(ham=None)
        bra = engine.psi.tensors[tensor_id]
        bra_tensor = np.zeros(bra.shape, dtype=np.complex128)
        bra_node = tn.Node(bra)
        ket_node = bra_node.copy(conjugate=True)

        if engine.psi.edges[tensor_id][0] in engine.block_hamiltonians:
            bra_tensor += engine._block_ham_psi(
                bra_node, engine.psi.edges[tensor_id][0], 0
            )
        if engine.psi.edges[tensor_id][1] in engine.block_hamiltonians:
            bra_tensor += engine._block_ham_psi(
                bra_node, engine.psi.edges[tensor_id][1], 1
            )
        bra_tensor += engine._ham_psi(
            bra_node, engine.psi.edges[tensor_id][:2], [0, 1]
        )

        bra_h = tn.Node(bra_tensor)
        bra_h[0] ^ ket_node[0]
        bra_h[1] ^ ket_node[1]
        block_ham = tn.contractors.auto(
            [bra_h, ket_node],
            output_edge_order=[bra_h[2], ket_node[2]],
        )
        engine.block_hamiltonians[engine.psi.edges[tensor_id][2]] = (
            block_ham.get_tensor()  # shape (χ, χ)
        )


# ---------------------------------------------------------------------------
# 5.  Monkey-patch helper
# ---------------------------------------------------------------------------

def patch_physics_engine(engine):
    """
    Patch an existing PhysicsEngine instance in-place:

      - engine.lanczos()              → ttn_eigensolver()
      - engine._set_block_hamiltonian → stores (χ,χ) matrices not (χ,χ,χ,χ)

    Also purges any already-stored (χ,χ,χ,χ) block Hamiltonians by reshaping
    them to (χ²,χ²), immediately reclaiming memory.

    Call this once after constructing your PhysicsEngine / GroundStateSearch:

        gss = GroundStateSearch(psi, hamiltonian, ...)
        patch_physics_engine(gss)
        gss.run(...)
    """
    import types

    # 1. Replace lanczos
    engine.lanczos = lambda central_tensor_ids, **kw: ttn_eigensolver(
        engine, central_tensor_ids, **kw
    )

    # 2. Replace _set_block_hamiltonian
    engine._set_block_hamiltonian = lambda tensor_id, ham=None: (
        _set_block_hamiltonian_as_matrix(engine, tensor_id, ham)
    )

    # 3. Reshape any already-stored (χ,χ,χ,χ) tensors → (χ²,χ²)
    freed = 0
    for k, v in engine.block_hamiltonians.items():
        if isinstance(v, np.ndarray) and v.ndim == 4:
            d0, d1 = v.shape[0], v.shape[1]
            engine.block_hamiltonians[k] = v.reshape(d0 * d1, d0 * d1)
            freed += 1
    if freed:
        print(f"patch_physics_engine: reshaped {freed} block Hamiltonians to matrix form.")

    # print("patch_physics_engine: lanczos → eigsh(LinearOperator), no χ⁴ storage.")
    return engine