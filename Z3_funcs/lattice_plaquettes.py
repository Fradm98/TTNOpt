"""
Plaquette and link labeling for triangular lattices with open boundary
conditions, plus charge-routing coefficients for link operators.

Conventions
------------
Links from site (i,j):
    0 = "_"  -> (i+1, j)
    1 = "/"  -> (i, j+1)
    2 = "\\" = 1 - 0  -> from (i,j) to (i-1,j+1)  [= e1 - e0]

Plaquettes:
    type 0 (up,   /\\) anchored at (i,j): corners (i,j), (i+1,j), (i,j+1)
    type 1 (down, \\/) anchored at (i,j): corners (i,j), (i,j+1), (i-1,j+1)

A plaquette exists iff all three of its corner sites exist in the lattice.

Plaquette labeling order: sweep rows j = 0 .. Ly-2 (bottom to top); within
a row, sweep i ascending (left to right); at a given (i,j), type 0 is
labeled before type 1 if both exist there.

Two boundary shapes are supported:
    "parallelogram": every row spans i in [0, Lx-1]
    "hexagon":       Ly = 2*Lx - 1 required. Row j spans i in
                      [left(j), right(j)] with
                          left(j)  = -(Lx-1) + max(0, jmid - j)
                          right(j) =  (Lx-1) - max(0, j - jmid)
                      jmid = (Ly-1)//2. This is a parallelogram of width
                      2*Lx-1 with the bottom-left corner (rows j<jmid) and
                      top-right corner (rows j>jmid) sheared off.

Link-plaquette mapping (label_links):
    Each link is bordered by up to two plaquettes. The tuple order is
    (left-side, right-side) plaquette when facing along the link's
    direction -- equivalently, clockwise around the link:
        link 0 of (i,j):  ( plaquette(i,j,0),     plaquette(i+1,j-1,1) )
        link 1 of (i,j):  ( plaquette(i,j,1),     plaquette(i,j,0)     )
        link 2 of (i,j):  ( plaquette(i-1,j,0),   plaquette(i,j,1)     )
    Each entry is the plaquette's integer label if it exists, else None.
    By convention the first slot's operator is used un-daggered and the
    second slot's operator is daggered.

Charge-routing coefficients (label_link_coefficients):
    A charge q_ij sits at every site. It is transported along direction 1
    ("/") from the bottom of its column upward. On reaching the lattice's
    actual top boundary, the accumulated charge turns and is transported
    rightward along direction 0 ("_"). For the hexagon, a column that
    runs out of rows before reaching the global top (i.e. hits the
    lower-right slanted edge) instead turns onto direction 2 ("\\") at
    that point, jumping into the next column to the left, where it
    resumes climbing via direction 1.

    Each link's coefficient is given as a list of (i,j) site indices
    whose charges q_ij multiply together (empty list = coefficient 1).
"""

import numpy as np

omega = np.exp(1j * 2 * np.pi / 3)

def site_bounds(Lx, Ly, shape="parallelogram"):
    """Return dict j -> (left_i, right_i) inclusive, for each row j=0..Ly-1."""
    bounds = {}
    if shape == "parallelogram":
        for j in range(Ly):
            bounds[j] = (0, Lx - 1)
    elif shape == "hexagon":
        if Ly != 2 * Lx - 1:
            raise ValueError(f"hexagon shape requires Ly = 2*Lx-1, got Lx={Lx}, Ly={Ly}")
        jmid = (Ly - 1) // 2
        for j in range(Ly):
            left = -(Lx - 1) + max(0, jmid - j)
            right = (Lx - 1) - max(0, j - jmid)
            bounds[j] = (left, right)
    else:
        raise ValueError(f"unknown shape '{shape}'")
    return bounds


def site_exists(i, j, Lx, Ly, bounds):
    if j < 0 or j > Ly - 1:
        return False
    left, right = bounds[j]
    return left <= i <= right


def column_row_range(Lx, Ly, bounds, i):
    """
    For column i, return (j_min, j_max): the lowest and highest row j for
    which site (i,j) exists. Returns None if the column is entirely empty.
    Shape-agnostic: works directly off the bounds dict.
    """
    js = [j for j in range(Ly) if bounds[j][0] <= i <= bounds[j][1]]
    if not js:
        return None
    return min(js), max(js)


def all_columns(Lx, Ly, bounds):
    """Sorted list of every column index i that has at least one site."""
    cols = set()
    for j in range(Ly):
        l, r = bounds[j]
        cols.update(range(l, r + 1))
    return sorted(cols)

def nplaqs(Lx, Ly, shape):
    if shape == "parallelogram":
        dof = 2 * (Lx - 1) * (Ly - 1)
    elif shape == "hexagon":
        dof = 6 * (Lx - 1) ** 2
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return int(dof)

def cg_lattice(Lx, Ly, shape, cgl):
    if shape == "parallelogram":
        return int(((Lx - 1)/2**cgl) + 1), int(((Ly - 1)/2**cgl) + 1)
    elif shape == "hexagon":
        return int(((Lx - 1)/2**cgl) + 1), int(((Ly - 1)/2**cgl) + 1)


def label_plaquettes(Lx, Ly, shape="parallelogram"):
    """
    Returns:
        labels: dict (i,j,type) -> label  (type in {0,1})
        order:  list of (label, i, j, type) sorted by label
        bounds: the row bounds dict used (useful for plotting/debugging)
    """
    bounds = site_bounds(Lx, Ly, shape)

    def exists(i, j):
        return site_exists(i, j, Lx, Ly, bounds)

    labels = {}
    order = []
    label = 0

    # rows of sites span j = 0..Ly-1; plaquette row j can only involve
    # sites in rows j and j+1, so j runs 0..Ly-2
    for j in range(Ly - 1):
        left_j, right_j = bounds[j]
        left_j1, right_j1 = bounds[j + 1]
        i_min = min(left_j, left_j1) - 1
        i_max = max(right_j, right_j1) + 1
        for i in range(i_min, i_max + 1):
            has0 = exists(i, j) and exists(i + 1, j) and exists(i, j + 1)
            has1 = exists(i, j) and exists(i, j + 1) and exists(i - 1, j + 1)
            if has0:
                labels[(i, j, 0)] = label
                order.append((label, i, j, 0))
                label += 1
            if has1:
                labels[(i, j, 1)] = label
                order.append((label, i, j, 1))
                label += 1

    return labels, order, bounds


def site_plaquette_count(Lx, Ly, labels, bounds):
    """dict (i,j) -> list of plaquette types owned, for all existing sites."""
    counts = {}
    for j, (l, r) in bounds.items():
        for i in range(l, r + 1):
            owned = []
            if (i, j, 0) in labels:
                owned.append(0)
            if (i, j, 1) in labels:
                owned.append(1)
            counts[(i, j)] = owned
    return counts


def label_links(Lx, Ly, shape="parallelogram"):
    """
    For every link that exists in the lattice, return the (up to two)
    plaquette labels bordering it, ordered clockwise (left-side, right-side).

    Returns:
        link_plaquettes: dict (i,j,dir) -> (label_or_None, label_or_None)
        bounds: the row bounds dict used
    """
    bounds = site_bounds(Lx, Ly, shape)

    def exists(i, j):
        return site_exists(i, j, Lx, Ly, bounds)

    labels, order, bounds = label_plaquettes(Lx, Ly, shape)

    def plabel(i, j, t):
        return labels.get((i, j, t), None)

    link_plaquettes = {}

    for j, (l, r) in bounds.items():
        for i in range(l, r + 1):
            # link 0: (i,j) -> (i+1,j)
            if exists(i + 1, j):
                link_plaquettes[(i, j, 0)] = (plabel(i, j, 0), plabel(i + 1, j - 1, 1))
            # link 1: (i,j) -> (i,j+1)
            if exists(i, j + 1):
                link_plaquettes[(i, j, 1)] = (plabel(i, j, 1), plabel(i, j, 0))
            # link 2: (i,j) -> (i-1,j+1)
            if exists(i - 1, j + 1):
                link_plaquettes[(i, j, 2)] = (plabel(i - 1, j, 0), plabel(i, j, 1))

    return link_plaquettes, bounds


def label_link_coefficients(Lx, Ly, shape="parallelogram"):
    """
    For every link (i,j,dir), return its bordering-plaquette tuple (as in
    label_links) together with the charge-routing coefficient: the list
    of (i,j) site indices whose q_ij multiply together as the prefactor
    on that link's operator pair (empty list => coefficient 1).

    Rules (verified against the handwritten derivation):
      link 0 of (i,j):
          - if j < Ly-1 (not the top row):       coeff = []
          - if j == Ly-1 (top row, turning the
            charge rightward):                   coeff = all (r,s) with
                r ranging over every existing column from the lattice's
                leftmost column up to i, and for each such r, s ranging
                over that column's full existing row range [jmin(r), Ly-1].

      link 1 of (i,j):
          coeff = all (i,s) with s ranging from column i's lowest
          existing row jmin(i) up to j (the column's running sum so far).
          This single formula covers bulk and edge columns alike.

      link 2 of (i,j):
          Link 2 only carries a charge-routing coefficient at the single
          site where a column runs out of rows before reaching the
          lattice's global top -- i.e. exactly when link 1 of (i,j) does
          NOT exist (no site at (i,j+1)) while link 2 of (i,j) does
          (site (i-1,j+1) exists). At that site the charge handed
          leftward must carry not just column i's own accumulated charge
          but everything already swept in from every column to its right
          (those columns already handed their charge into column i's
          territory via their own link-2 handoffs further down):
                coeff = all (r,s) with r ranging from i to the rightmost
                        existing column, and for each such r, s ranging
                        over that column's full existing row range
                        [0, jmax(r)].
          Everywhere else (including every other link 2 in that same
          column, where link 1 also exists and carries the charge
          instead) -> coeff = [].
          For the parallelogram this condition never triggers, so link 2
          never carries a coefficient there.

    Returns:
        link_data: dict (i,j,dir) -> (p_left, p_right, coeff)
        bounds: the row bounds dict used
    """
    link_plaquettes, bounds = label_links(Lx, Ly, shape)

    def exists(i, j):
        return site_exists(i, j, Lx, Ly, bounds)

    cols = all_columns(Lx, Ly, bounds)
    leftmost_col = cols[0]
    rightmost_col = cols[-1]

    link_data = {}

    for (i, j, d), (p_left, p_right) in link_plaquettes.items():

        if d == 0:
            if j != Ly - 1:
                coeff = []
            else:
                coeff = []
                for r in range(leftmost_col, i + 1):
                    rng = column_row_range(Lx, Ly, bounds, r)
                    if rng is None:
                        continue
                    jmin, _ = rng
                    for s in range(jmin, Ly):
                        if exists(r, s):
                            coeff.append((r, s))

        elif d == 1:
            rng = column_row_range(Lx, Ly, bounds, i)
            jmin = rng[0] if rng is not None else j
            coeff = [(i, s) for s in range(jmin, j + 1) if exists(i, s)]

        elif d == 2:
            # charge hands off via link 2 only where link 1 doesn't exist
            # here but link 2 does -- i.e. this column has run out of rows
            link1_exists = exists(i, j + 1)
            if (not link1_exists) and exists(i - 1, j + 1):
                coeff = []
                for r in range(i, rightmost_col + 1):
                    rng = column_row_range(Lx, Ly, bounds, r)
                    if rng is None:
                        continue
                    _, jmax_r = rng
                    for s in range(0, jmax_r + 1):
                        if exists(r, s):
                            coeff.append((r, s))
            else:
                coeff = []

        else:
            raise ValueError(f"unknown link direction {d}")

        link_data[(i, j, d)] = (p_left, p_right, coeff)

    return link_data, bounds


def site_to_array_index(i, j, Lx, Ly, shape="parallelogram"):
    """
    Map a physical site (i,j) to (row, col) indices into a charges array
    that has already been put in "physical" orientation, i.e.

        charges = np.ones((Ly, Lx))            # parallelogram
        charges = np.ones((Ly, 2*Lx - 1))       # hexagon
        charges = charges[::-1]                 # row 0 = physical bottom

    Row index = j directly. Column index = i for the parallelogram, or
    i + (Lx-1) for the hexagon (so physical i = -(Lx-1) sits at column 0).
    """
    row = j
    if shape == "parallelogram":
        col = i
    elif shape == "hexagon":
        col = i + (Lx - 1)
    else:
        raise ValueError(f"unknown shape '{shape}'")
    return row, col

def array_to_site_index(row, col, Lx, Ly, shape="parallelogram"):
    """
    Map (row, col) indices into a charges array to a physical site (i, j)
    in "physical" orientation, i.e.

        charges = np.ones((Ly, Lx))            # parallelogram
        charges = np.ones((Ly, 2*Lx - 1))       # hexagon
        charges = charges[::-1]                 # row 0 = physical bottom

    Row index = j directly. Column index = i for the parallelogram, or
    i + (Lx-1) for the hexagon (so physical i = -(Lx-1) sits at column 0).
    """
    j = row
    if shape == "parallelogram":
        i = col
    elif shape == "hexagon":
        i = col - (Lx - 1)
    else:
        raise ValueError(f"unknown shape '{shape}'")
    return i, j

def make_charges_array(Lx, Ly, shape="parallelogram", fill_value=1.0):
    """
    Build a charges array in physical orientation (row 0 = bottom row,
    already flipped -- do NOT apply charges[::-1] to this), with every
    site that actually exists in the lattice set to `fill_value` and
    every other entry set to np.nan. Returned as a complex array (dtype
    complex128), since charges/coefficients in this construction are
    generally complex phases rather than plain reals.

    Shape: (Ly, Lx) for the parallelogram, (Ly, 2*Lx-1) for the hexagon.
    Useful as a safe default/template -- nan in any "outside the lattice"
    slot makes accidental use of it loud (propagates nan) instead of
    silently contributing a spurious 1.0 or other value.
    """
    bounds = site_bounds(Lx, Ly, shape)
    width = Lx if shape == "parallelogram" else 2 * Lx - 1
    charges = np.full((Ly, width), np.nan, dtype=complex)
    for j in range(Ly):
        left, right = bounds[j]
        for i in range(left, right + 1):
            r, c = site_to_array_index(i, j, Lx, Ly, shape)
            charges[r, c] = fill_value
    return charges


def evaluate_link_coefficients(Lx, Ly, charges, shape="parallelogram"):
    """
    Like label_link_coefficients, but replaces each link's `coeff` site
    list with the actual multiplied charge value looked up from `charges`.
    Values are always returned as Python complex numbers, regardless of
    whether `charges` itself is real- or complex-valued.

    Parameters
    ----------
    charges : np.ndarray
        Already row-flipped to physical orientation (see
        site_to_array_index docstring), shape (Ly, Lx) for the
        parallelogram or (Ly, 2*Lx-1) for the hexagon. May be real or
        complex-valued.

    Returns
    -------
    link_values: dict (i,j,dir) -> (p_left, p_right, value)
        `value` is the complex product of charges over that link's
        coefficient site list (1+0j if the list is empty).
    bounds: the row bounds dict used
    """
    link_data, bounds = label_link_coefficients(Lx, Ly, shape)

    link_values = {}
    for key, (p_left, p_right, coeff) in link_data.items():
        value = complex(1.0)
        for (ci, cj) in coeff:
            r, c = site_to_array_index(ci, cj, Lx, Ly, shape)
            q = charges[r, c]
            if np.isnan(q):
                raise ValueError(
                    f"link {key}: coefficient references site ({ci},{cj}), "
                    f"which is nan in the charges array (array index "
                    f"[{r},{c}]). This site should exist in the lattice -- "
                    f"check that `charges` was built for the same (Lx,Ly,shape)."
                )
            value *= complex(q)
        link_values[key] = (p_left, p_right, value)

    return link_values, bounds


def charge_routing_coefficients(Lx, Ly, shape="parallelogram", bound_state=None, R=1, chargesx=None, chargesy=None):
    """Per-link duality/charge-routing coefficient (evaluate_link_
    coefficients' raw value, NOT multiplied by g) for a specific charge
    configuration.

    Replicates create_ids_coeffs_file's own charges-array construction
    exactly (same vals=[omega,omega**2]/[omega,omega,omega] convention),
    so this always matches what a checkpoint's Hamiltonian was actually
    built with -- meant for weighting a link's raw <V_link> expectation
    value by its own coefficient (see Z3_funcs.observables.
    link_electric_fields), since <V_link> alone is not proportional to
    that link's Hamiltonian energy contribution unless its own
    coefficient happens to be real.

    Returns
    -------
    coeffs: dict (i,j,dir) -> complex coefficient (1+0j where no charge
        was swept into that link).
    """
    charges = make_charges_array(Lx, Ly, shape)
    if bound_state is not None:
        vals = [omega, omega**2] if bound_state == "meson" else [omega, omega, omega]
        if chargesx is None:
            chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state=bound_state, shape=shape, R=R)
        for i, j, val in zip(chargesx, chargesy, vals):
            r, c = site_to_array_index(i, j, Lx, Ly, shape)
            charges[r, c] = val

    link_values, _ = evaluate_link_coefficients(Lx=Lx, Ly=Ly, charges=charges, shape=shape)
    return {key: coeff for key, (p_left, p_right, coeff) in link_values.items()}


def check_coarse_graining_compatible(Lx, Ly, shape="parallelogram"):
    """
    Check whether (Lx, Ly) admit a clean disjoint tiling of the small
    plaquettes into big (2x-scale) up- and down-triangles, stepping rows
    by 2 and, within each row, columns by 2 starting from that row's own
    left bound.

    Verified empirically necessary and sufficient conditions:
        parallelogram: Lx odd AND Ly odd
        hexagon:       Lx odd (Ly = 2*Lx-1 is then automatically odd too)

    Returns (True, "") if compatible, or (False, reason) if not.
    """
    if shape == "parallelogram":
        if Lx % 2 == 0 or Ly % 2 == 0:
            return False, (
                f"parallelogram coarse-graining requires both Lx and Ly odd; "
                f"got Lx={Lx}, Ly={Ly}"
            )
        return True, ""
    elif shape == "hexagon":
        if Ly != 2 * Lx - 1:
            return False, f"hexagon requires Ly = 2*Lx-1, got Lx={Lx}, Ly={Ly}"
        if Lx % 2 == 0:
            return False, f"hexagon coarse-graining requires Lx odd; got Lx={Lx}"
        return True, ""
    else:
        return False, f"unknown shape '{shape}'"


def group_plaquettes_coarse(Lx, Ly, shape="parallelogram"):
    """
    Partition every small plaquette into disjoint groups of 4, each
    forming one big (2x-scale) triangle -- either a big up-triangle
    (/\\, "type 0") or a big down-triangle (\\/, "type 1").

    Big up-triangle anchored at (i,j): groups small plaquettes
        (i,j,0), (i+1,j,0), (i+1,j,1), (i,j+1,0)
    valid iff (i,j), (i+2,j), (i,j+2) all exist.

    Big down-triangle anchored at (i,j): groups small plaquettes
        (i,j,1), (i,j+1,1), (i-1,j+1,0), (i-1,j+1,1)
    valid iff (i,j), (i-2,j+2), (i,j+2) all exist.

    Sweep order: rows j = 0, 2, 4, ... (bottom to top); within each row,
    i starts at that row's own left bound (from site_bounds) and steps
    by 2. This makes the row-to-row transition shape-agnostic -- no
    separate (i,j+2) vs (i-2,j+2) rule is needed, the row bounds handle
    it automatically for either shape.

    Use check_coarse_graining_compatible first: this only produces a
    complete, non-overlapping cover of all small plaquettes when (Lx,Ly)
    satisfy that compatibility condition.

    Returns
    -------
    groups: list of dicts, each
        {
            "anchor": (i, j),
            "type": 0 or 1,              # 0 = big up-triangle, 1 = big down
            "plaquettes": [p0, p1, p2, p3]   # the 4 small plaquette labels
        }
        ordered by anchor row j ascending, then i ascending, then type 0
        before type 1 at the same anchor (mirroring the small-plaquette
        labeling convention).
    bounds: the row bounds dict used
    """
    labels, order, bounds = label_plaquettes(Lx, Ly, shape)

    def exists(i, j):
        return site_exists(i, j, Lx, Ly, bounds)

    def plabel(i, j, t):
        return labels.get((i, j, t), None)

    groups = []

    for j in range(0, Ly - 1, 2):
        left_j, right_j = bounds[j]
        i = left_j
        while i <= right_j:
            # big up-triangle at (i,j)
            if exists(i, j) and exists(i + 2, j) and exists(i, j + 2):
                plaqs = [
                    plabel(i, j, 0),
                    plabel(i + 1, j, 0),
                    plabel(i + 1, j, 1),
                    plabel(i, j + 1, 0),
                ]
                if None in plaqs:
                    raise ValueError(
                        f"big up-triangle anchored at ({i},{j}) has corner "
                        f"sites but a missing small plaquette: {plaqs}"
                    )
                groups.append({"anchor": (i, j), "type": 0, "plaquettes": plaqs})

            # big down-triangle at (i,j)
            if exists(i, j) and exists(i - 2, j + 2) and exists(i, j + 2):
                plaqs = [
                    plabel(i, j, 1),
                    plabel(i, j + 1, 1),
                    plabel(i - 1, j + 1, 0),
                    plabel(i - 1, j + 1, 1),
                ]
                if None in plaqs:
                    raise ValueError(
                        f"big down-triangle anchored at ({i},{j}) has corner "
                        f"sites but a missing small plaquette: {plaqs}"
                    )
                groups.append({"anchor": (i, j), "type": 1, "plaquettes": plaqs})

            i += 2

    return groups, bounds

def group_plaquettes_coarse_gen(Lx: int, Ly: int, shape: str="parallelogram", cgl: int=1, off: int=0):
    """
    Partition every small plaquette into disjoint groups of 4, each
    forming one big (2x-scale) triangle -- either a big up-triangle
    (/\\, "type 0") or a big down-triangle (\\/, "type 1").

    Big up-triangle anchored at (i,j): groups small plaquettes
        (i,j,0), (i+1,j,0), (i+1,j,1), (i,j+1,0)
    valid iff (i,j), (i+2,j), (i,j+2) all exist.

    Big down-triangle anchored at (i,j): groups small plaquettes
        (i,j,1), (i,j+1,1), (i-1,j+1,0), (i-1,j+1,1)
    valid iff (i,j), (i-2,j+2), (i,j+2) all exist.

    Sweep order: rows j = 0, 2, 4, ... (bottom to top); within each row,
    i starts at that row's own left bound (from site_bounds) and steps
    by 2. This makes the row-to-row transition shape-agnostic -- no
    separate (i,j+2) vs (i-2,j+2) rule is needed, the row bounds handle
    it automatically for either shape.

    Use check_coarse_graining_compatible first: this only produces a
    complete, non-overlapping cover of all small plaquettes when (Lx,Ly)
    satisfy that compatibility condition.

    Returns
    -------
    groups: list of dicts, each
        {
            "anchor": (i, j),
            "type": 0 or 1,              # 0 = big up-triangle, 1 = big down
            "plaquettes": [p0, p1, p2, p3]   # the 4 small plaquette labels
        }
        ordered by anchor row j ascending, then i ascending, then type 0
        before type 1 at the same anchor (mirroring the small-plaquette
        labeling convention).
    bounds: the row bounds dict used
    """
    labels, order, bounds = label_plaquettes(Lx, Ly, shape)

    def exists(i, j):
        return site_exists(i, j, Lx, Ly, bounds)

    def plabel(i, j, t):
        return labels.get((i, j, t), None)

    groups = []

    for j in range(0, Ly - 1, 2**cgl):
        left_j, right_j = bounds[j]
        i = left_j
        while i <= right_j:

            # big down-triangle at (i,j)
            if exists(i, j) and exists(i - 2**cgl, j + 2**cgl) and exists(i, j + 2**cgl):
                
                plaqs = []
                idx = 0
                for jp in range(j, j + 2**cgl):
                    ips = list(range(i, i - idx - 1, -1))
                    ips.reverse()
                    # for ip in range(i, i - idx - 1, -1):
                    for ip in ips:
                        plaqs.append(plabel(ip, jp, 1) + off)
                        if ip < i:
                            plaqs.append(plabel(ip, jp, 0) + off)
                    idx += 1

                if None in plaqs:
                    raise ValueError(
                        f"big down-triangle anchored at ({i},{j}) has corner "
                        f"sites but a missing small plaquette: {plaqs}"
                    )
                groups.append({"anchor": (i, j), "type": 1, "plaquettes": plaqs})
            
            # big up-triangle at (i,j)
            if exists(i, j) and exists(i + 2**cgl, j) and exists(i, j + 2**cgl):

                plaqs = []
                idx = 0
                for jp in range(j, j + 2**cgl):
                    for ip in range(i, i + 2**cgl - idx):
                        if ip > i:
                            plaqs.append(plabel(ip, jp, 1) + off)
                        plaqs.append(plabel(ip, jp, 0) + off)
                    idx += 1

                if None in plaqs:
                    raise ValueError(
                        f"big up-triangle anchored at ({i},{j}) has corner "
                        f"sites but a missing small plaquette: {plaqs}"
                    )
                groups.append({"anchor": (i, j), "type": 0, "plaquettes": plaqs})

            i += 2**cgl

    return groups, bounds


def verify_coarse_groups(Lx, Ly, shape):
    groups, bounds = group_plaquettes_coarse(Lx, Ly, shape)
    _, order, _ = label_plaquettes(Lx, Ly, shape)
    used = []
    for g in groups:
        used.extend(g["plaquettes"])
    perfect = sorted(used) == list(range(len(order)))
    print(f"{shape} Lx={Lx},Ly={Ly}: {len(order)} small plaquettes -> "
            f"{len(groups)} big triangles, perfect disjoint cover = {perfect}")
    return groups


def verify_coarse_graining_levels(Lx, Ly, shape):
    nplaqus = nplaqs(Lx, Ly, shape)
    cg = 1

    while True:
        group_cg, _ = group_plaquettes_coarse_gen(Lx, Ly, shape, cg)
        cg_plaqs = np.asarray([g["plaquettes"] for g in group_cg]).flatten()

        if len(cg_plaqs) != nplaqus:
            print(f"Maximum valid coarse-graining level: {cg-1}")
            return cg - 1

        Lx_cg, Ly_cg = cg_lattice(Lx, Ly, shape, cgl=cg)
        nplaqs_cg = nplaqs(Lx_cg, Ly_cg, shape)
        print(f"Coarse Graining Level {cg} is valid.")
        print(f"total plaquettes are: {nplaqus}")
        print(f"Coarse grained plaquettes at {cg} cg level are: {nplaqs_cg}")
        cg += 1


def reorder_by_next_level(group_k, group_k1):
    """
    Reorders the cg=k groups according to their parent at cg=k+1.

    Parameters
    ----------
    group_k : list of dict
    group_k1 : list of dict

    Returns
    -------
    reordered : list of dict
        group_k reordered so that children of each cg=k+1 block are consecutive.

    parent_map : list[int]
        parent_map[i] gives the index of the parent (in group_k1)
        of reordered[i].
    """

    # Convert every plaquette list to a set for fast inclusion tests
    small_sets = [set(g["plaquettes"]) for g in group_k]
    large_sets = [set(g["plaquettes"]) for g in group_k1]

    reordered = []
    parent_map = []

    for parent_idx, large in enumerate(large_sets):

        for child_idx, small in enumerate(small_sets):

            if small.issubset(large):
                reordered.append(group_k[child_idx])
                parent_map.append(parent_idx)

    return reordered, parent_map


def divide_hexagon_plaquettes(Lx,Ly,off):
    group_hex, _ = group_plaquettes_coarse_gen(Lx,Ly,shape="hexagon",cgl=0,off=off)
    ttn_labels = []

    # first group (lower right)
    hex_label_binary = []
    for g in group_hex:
        if g['anchor'] == (0,0) and g['type'] == 0:
            hex_label_binary += g['plaquettes']
        elif g['anchor'] == (1,0) and g['type'] == 1:
            hex_label_binary += g['plaquettes']

    ttn_labels += [hex_label_binary]

    # second group (upper right)
    hex_label_binary = []
    for g in group_hex:
        if g['anchor'] == (0,1) and g['type'] == 0:
            hex_label_binary += g['plaquettes']
        elif g['anchor'] == (0,1) and g['type'] == 1:
            hex_label_binary += g['plaquettes']

    ttn_labels += [hex_label_binary]

    # third group (center left)
    hex_label_binary = []
    for g in group_hex:
        if g['anchor'] == (-1,1) and g['type'] == 0:
            hex_label_binary += g['plaquettes']
    
    for g in group_hex:
        if g['anchor'] == (0,0) and g['type'] == 1:
            hex_label_binary += g['plaquettes']

    ttn_labels += [hex_label_binary]
    return ttn_labels

    
def make_symm_meson(Lx, Ly, shape, R: int=1):
    """
    Make a symmetric meson state for the hexagon or parallelogram lattice.

    Parameters
    ----------
    Lx : int
        The horizontal size of the hexagon or parallelogram.
    Ly : int
        The vertical size of the hexagon or parallelogram.
    R : int, optional
        The radius of the meson state.

    Returns
    -------
    meson_op : list of int
        The plaquette labels that make up the symmetric meson state.
    """
    if shape == "hexagon":
        center = (0, Ly//2)
    elif shape == "parallelogram":
        center = (Lx//2, Ly//2)

    if (R % 2) == 0:
        R = R // 2
        c0 = (center[0] - R, center[1])
        c1 = (center[0] + R, center[1])
    else:
        R = R // 2
        c0 = (center[0] - (R+1), center[1])
        c1 = (center[0] + R, center[1])
    return [c0[0],c1[0]], [c0[1],c1[1]] 


def make_symm_baryon(Lx, Ly, shape, R: int=1):
    """
    Make a symmetric baryon state for the hexagon or parallelogram lattice.

    Parameters
    ----------
    Lx : int
        The horizontal size of the hexagon or parallelogram.
    Ly : int
        The vertical size of the hexagon or parallelogram.

    Returns
    -------
    baryon_op : list of int
        The plaquette labels that make up the symmetric baryon state.
    """
    if shape == "hexagon":
        center = (0, Ly//2)
    elif shape == "parallelogram":
        center = (Lx//2, Ly//2)

    c0 = (center[0] - R, center[1])
    c1 = (center[0], center[1]+R)
    c2 = (center[0] + R, center[1] - R)
    return [c0[0],c1[0],c2[0]], [c0[1],c1[1],c2[1]] 


def get_coord_charges(Lx, Ly, bound_state="meson", shape="parallelogram", R: int=1):
    """
    Make a symmetric meson or baryon state for the hexagon or parallelogram lattice.

    Parameters
    ----------
    Lx : int
        The horizontal size of the hexagon or parallelogram.
    Ly : int
        The vertical size of the hexagon or parallelogram.
    bound_state : str
        "meson" or "baryon"
    R : int, optional
        The radius of the meson or baryon state.

    Returns
    -------
    charges : list of int
        The plaquette labels that make up the symmetric meson or baryon state.
    """
    if bound_state == "meson":  
        return make_symm_meson(Lx, Ly, shape=shape, R=R)
    elif bound_state == "baryon":
        return make_symm_baryon(Lx, Ly, shape=shape, R=R)
    elif bound_state == None:
        return None, None
    else:
        raise ValueError(f"Unknown bound state '{bound_state}', must be 'meson', 'baryon', or None")