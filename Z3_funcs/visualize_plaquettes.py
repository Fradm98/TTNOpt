import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from Z3_funcs.create_graph_file import nplaqs
from Z3_funcs.utils import get_folder
from Z3_funcs.lattice_plaquettes import label_plaquettes, label_links, get_coord_charges, array_to_site_index, charge_routing_coefficients
from Z3_funcs.observables import load_engine_from_tensor, link_electric_fields, get_lattice_electric_fields

# ── Geometry helpers ──────────────────────────────────────────────────────────

def site_pos(i, j, scale=1.0):
    """
    Map lattice site (i, j) to 2-D Cartesian coordinates.

    Row j is offset by j/2 in x (standard triangular / hexagonal lattice).
    The vertical spacing between rows is sqrt(3)/2 so that all triangle
    edges have the same length (= scale).
    """
    x = (i + j / 2) * scale
    y = j * (np.sqrt(3) / 2) * scale
    return np.array([x, y])


def triangle_vertices(i, j, tri_type, scale=1.0):
    """
    Return the three vertices of a triangular plaquette.

    Type 0  (upward):   corners (i,j), (i+1,j), (i,  j+1)
    Type 1  (downward): corners (i,j), (i,j+1), (i-1,j+1)
    """
    if tri_type == 0:
        corners = [(i, j), (i + 1, j), (i, j + 1)]
    else:
        corners = [(i, j), (i, j + 1), (i - 1, j + 1)]
    return np.array([site_pos(ci, cj, scale) for ci, cj in corners])


def plaquette_centroid(i, j, tri_type, scale=1.0):
    return triangle_vertices(i, j, tri_type, scale).mean(axis=0)


def nearest_center_plaquette(Lx, Ly, shape="parallelogram", scale=1.0):
    """The plaquette label whose centroid is closest to the geometric
    center of the whole lattice (mean of every plaquette's centroid).
    A reasonable default "middle" plaquette when the caller doesn't have
    a specific one in mind."""
    _, order, _ = label_plaquettes(Lx, Ly, shape)
    centroids = {label: plaquette_centroid(i, j, t, scale) for label, i, j, t in order}
    center = np.mean(list(centroids.values()), axis=0)
    return min(centroids, key=lambda lbl: np.linalg.norm(centroids[lbl] - center))


def _link_color_mapper(color_by="abs", cmap_name=None, min_val=None, max_val=None):
    """Build (mapper, metric_fn, colorbar_label, cmap, norm) for coloring
    link values, by magnitude ("abs"), phase ("arg"), or real part
    ("real").

    The value passed through metric_fn is always assumed to already be
    charge-routing-weighted (coeff_link * <V_link>, from
    charge_routing_coefficients()) by the caller (plot_plaquette_links/
    plot_lattice_links apply this unconditionally, not as an optional
    mode) -- <V_link> alone is NOT proportional to that link's actual
    Hamiltonian energy contribution unless its own coefficient happens to
    be real, so this weighting isn't a display choice, it's required for
    any of these three views to be physically meaningful once charges are
    present. For vacuum (no charges), coeff_link=1 everywhere and this is
    a no-op.

    Confinement in a Z3 gauge theory shows up as a discrete flux-quantum
    PHASE shift (0, +-120 degrees) on the string, at UNIT magnitude --
    "abs" alone can look uniformly bright even where a string is present
    (see plot_lattice_links's g=-10 case), so "arg" is often the more
    informative choice; "abs" stays default for backward compatibility /
    cases where magnitude suppression (e.g. near the phase transition) is
    the interesting signal instead. Note "abs" is actually unaffected by
    the weighting (|coeff_link|=1 always), only "arg" and "real" change.

    "real": Re(coeff_link * <V_link>) -- the g-independent Hamiltonian
    energy contribution per link (multiply by g for the actual energy;
    see the R=2 direct-string tests, where this isolates exactly the
    genuine string links, unlike the unweighted raw value which also
    lights up on Dirac-string construction artifacts).
    """
    if color_by == "abs":
        cmap_name = cmap_name or "magma"
        _min = 0.0 if min_val is None else min_val
        _max = 1.0 if max_val is None else max_val
        norm = mcolors.Normalize(vmin=_min, vmax=_max, clip=True)
        metric_fn = abs
        label = r"$|\langle V \cdot v\rangle|$"
    elif color_by == "arg":
        cmap_name = cmap_name or "twilight"
        _min = -np.pi if min_val is None else min_val
        _max = np.pi if max_val is None else max_val
        norm = mcolors.Normalize(vmin=_min, vmax=_max, clip=True)
        metric_fn = np.angle
        label = r"$\arg(coeff_{link}\langle V_{link}\rangle)$ (rad)"
    elif color_by == "real":
        cmap_name = cmap_name or "RdBu_r"
        _min = -1.0 if min_val is None else min_val
        _max = 1.0 if max_val is None else max_val
        norm = mcolors.Normalize(vmin=_min, vmax=_max, clip=True)
        metric_fn = lambda v: v.real
        label = r"$\mathrm{Re}(coeff_{link}\langle V_{link}\rangle)$"
    else:
        raise ValueError(f"color_by must be 'abs', 'arg', or 'real', got {color_by!r}")
    cmap = mpl.colormaps.get_cmap(cmap_name)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return mapper, metric_fn, label, cmap, norm


def link_site_endpoints(i, j, d):
    """The two direct-lattice sites bordering link (i,j,d) -- see
    lattice_plaquettes.py's module docstring for the direction convention."""
    if d == 0:
        return (i, j), (i + 1, j)
    elif d == 1:
        return (i, j), (i, j + 1)
    elif d == 2:
        return (i, j), (i - 1, j + 1)
    else:
        raise ValueError(f"unknown link direction {d}")


# ── Main plot function ────────────────────────────────────────────────────────

def plot_plaquettes(
    Lx, Ly, shape, bound_state, R, g, chi, device, 
    xs=None, 
    ys=None,
    precision=3,
    max_ee=None,
    min_ee=None,
    log_scale=True,
    scale=1.0,
    cmap_name="GnBu",
    figsize=(10, 9),
    edge_lw=0.8,
    edge_color="white",
    # vertex / site options
    show_vertices=True,
    vertex_size=40,
    vertex_color="0.25",          # default sites: dark gray
    charge_color="crimson",       # sites carrying a charge
    charge_size=120,
    charge_marker="*",
    charge_zorder=5,
    save=True,
):
    """
    Draw the physical triangular lattice as coloured plaquettes, with
    optional site vertices and charge markers.

    Works for both shape="hexagon" and shape="parallelogram" — the
    geometry is fully determined by label_plaquettes / site_pos, so no
    shape-specific logic is needed here.

    Parameters
    ----------
    Lx, Ly        : lattice dimensions
    shape         : "hexagon" or "parallelogram"
    bound_state   : "meson", "baryon", or None  (None → no charges drawn)
    R, g, chi     : physical / simulation parameters
    device        : "pc", "ngt", or "presto"
    precision     : decimal digits used in file path for g
    max_ee        : colour-scale upper bound (defaults to data max)
    min_ee        : colour-scale lower bound for log scale (data min > 0)
    log_scale     : use LogNorm instead of linear Normalize
    scale         : lattice constant (controls figure size in data units)
    cmap_name     : base matplotlib colormap name
    figsize       : figure size in inches
    edge_lw       : line width of triangle edges
    edge_color    : colour of triangle edges
    show_vertices : draw lattice sites as small dots
    vertex_size   : scatter marker size for ordinary sites
    vertex_color  : colour of ordinary lattice sites
    charge_color  : colour of sites that carry a charge
    charge_size   : scatter marker size for charged sites
    charge_marker : matplotlib marker style for charges
    charge_zorder : z-order for charge markers (draw on top)
    save          : whether to save the figure to disk
    """

    # ── paths ─────────────────────────────────────────────────────────────────
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fdimarca/projects/5_Z3"

    # ── colormap (trim the very light end) ────────────────────────────────────
    cmap_base = mpl.colormaps.get_cmap(cmap_name)
    cmap_arr = cmap_base(np.linspace(0.2, 1, 256))
    cmap = mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_custom", cmap_arr)

    # ── lattice info & plaquette labels ───────────────────────────────────────
    N = nplaqs(Lx, Ly, shape)
    _, order, _ = label_plaquettes(Lx, Ly, shape)
    # order: list of (label, i, j, type)  — label == leaf-node index in TTN

    # ── read entanglement data ─────────────────────────────────────────────────
    folder = get_folder(Lx, Ly, shape, bound_state=bound_state, R=R, chargesx=xs, chargesy=ys, device=device)
    df = pd.read_csv(f"{folder}/g_{g:.{precision}f}/run_chi-{chi}/basic.csv")

    # Build a dict  leaf_node_index -> entanglement entropy
    # The edge connecting a leaf node to its parent carries the entropy.
    # A leaf node is any node with index < N.
    ee = {}
    for _, row in df.iterrows():
        n1, n2 = int(row["node1"]), int(row["node2"])
        if n1 < N:
            ee[n1] = row["entanglement"]
        elif n2 < N:
            ee[n2] = row["entanglement"]

    # ── normalisation ─────────────────────────────────────────────────────────
    values = np.array([ee.get(lbl, 0.0) for lbl, *_ in order])

    _max = max_ee if max_ee is not None else values.max()

    if log_scale:
        _min = min_ee if min_ee is not None else values[values > 0].min()
        norm = mcolors.LogNorm(vmin=_min, vmax=_max, clip=True)
    else:
        _min = min_ee if min_ee is not None else 0.0
        norm = mcolors.Normalize(vmin=_min, vmax=_max, clip=True)

    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # ── draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # Collect every unique lattice site that appears in at least one plaquette.
    # We use a set of (i, j) pairs to avoid duplicates.
    all_sites = set()
    for label, i, j, tri_type in order:
        verts = triangle_vertices(i, j, tri_type, scale)
        val = ee.get(label, 0.0)
        color = mapper.to_rgba(val)
        poly = mpatches.Polygon(
            verts,
            closed=True,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=edge_lw,
        )
        ax.add_patch(poly)

        # Accumulate sites
        if tri_type == 0:
            all_sites.update([(i, j), (i + 1, j), (i, j + 1)])
        else:
            all_sites.update([(i, j), (i, j + 1), (i - 1, j + 1)])

        # Optional: label each triangle with its plaquette index
        # centroid = verts.mean(axis=0)
        # ax.text(*centroid, str(label), ha="center", va="center",
        #         fontsize=6, color="black")

    # Autoscale after adding patches
    ax.autoscale_view()

    # ── lattice vertices ──────────────────────────────────────────────────────
    if show_vertices:
        # Determine which sites carry charges
        charge_sites = set()
        if bound_state is not None:
            if xs is None:
                xs, ys = get_coord_charges(Lx, Ly, bound_state=bound_state, shape=shape, R=R)
            charge_sites = set([array_to_site_index(y,x,Lx,Ly,shape) for x,y in zip(xs, ys)])

        # Split sites into ordinary vs charged
        ordinary = [s for s in all_sites if s not in charge_sites]
        charged  = [s for s in all_sites if s in charge_sites]

        if ordinary:
            pts = np.array([site_pos(i, j, scale) for i, j in ordinary])
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=vertex_size,
                color=vertex_color,
                zorder=charge_zorder - 1,
                linewidths=0,
            )

        if charged:
            pts = np.array([site_pos(i, j, scale) for i, j in charged])
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=charge_size,
                color=charge_color,
                marker=charge_marker,
                zorder=charge_zorder,
                linewidths=0.5,
                edgecolors="white",
            )

    # ── colorbar ──────────────────────────────────────────────────────────────
    ax_cb = fig.add_axes([0.05, 0.25, 0.025, 0.5])
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)
    cb.outline.set_visible(False)
    ax_cb.yaxis.tick_left()
    ax_cb.set_title("Entanglement\nentropy", fontsize=9, pad=6)

    # ── charge legend (only when charges are present) ─────────────────────────
    if show_vertices and bound_state is not None and charge_sites:
        legend_handles = [
            mpl.lines.Line2D(
                [], [],
                marker=charge_marker,
                color="none",
                markerfacecolor=charge_color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                markersize=np.sqrt(charge_size) * 0.6,
                label=f"Charge ({bound_state})",
            ),
            mpl.lines.Line2D(
                [], [],
                marker="o",
                color="none",
                markerfacecolor=vertex_color,
                markersize=np.sqrt(vertex_size) * 0.6,
                label="Lattice site",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            fontsize=8,
        )

    fig.tight_layout()

    if save:
        out = (
            f"{drive_path}/figures/"
            f"plaq_{Lx}_{Ly}_{shape}_{bound_state}_cxs_{xs}_cys_{ys}_R_{R}_g_{g:.{precision}f}_chi_{chi}.pdf"
        )
        plt.savefig(out, transparent=True, bbox_inches="tight")
        print(f"Saved to {out}")

    return fig, ax


# ── Link-resolved electric field plot ───────────────────────────────────────

def plot_plaquette_links(
    Lx, Ly, shape, bound_state, R, g, chi, device,
    plaquette_label=None,
    xs=None,
    ys=None,
    precision=3,
    color_by="abs",
    max_val=None,
    min_val=None,
    scale=1.0,
    cmap_name=None,
    figsize=(8, 7),
    lattice_edge_lw=0.5,
    lattice_edge_color="0.85",
    anchor_edge_color="black",
    anchor_lw=2.0,
    link_lw=10.0,
    stub_length=0.4,
    show_vertices=True,
    vertex_size=20,
    vertex_color="0.5",
    save=True,
):
    """
    Draw the electric-field <V.v> link values connecting plaquette_label to
    its neighbors, on top of the full lattice (drawn faintly for context).

    Unlike plot_plaquettes (which just reads a pre-saved basic.csv), this
    loads a saved TTN checkpoint from tensors.hdf5, reconstructs a
    queryable engine, and computes each bordering link's value live via
    link_electric_fields/expval_general -- these expectation values are
    not saved during the sweep loop.

    Interior links (both bordering plaquettes exist) are drawn as a
    segment from plaquette_label's centroid to the neighbor's centroid.
    Boundary links (only one side exists) are drawn as a short stub from
    the centroid, through the actual direct-lattice link's midpoint,
    extended outward by stub_length.

    Parameters mirror plot_plaquettes where they overlap.
    plaquette_label : the plaquette whose bordering links to display (e.g.
        a geometrically central one, from label_plaquettes' labeling). If
        None, defaults to nearest_center_plaquette(Lx, Ly, shape) -- NOT
        "every link in the lattice" (use plot_lattice_links for that).
    color_by : "abs" (magnitude, bounded in [0,1], default), "arg" (phase
        in radians, [-pi,pi]), or "real" (Re(coeff_link*<V_link>), the
        g-independent Hamiltonian energy contribution, [-1,1]) -- see
        _link_color_mapper. Every mode is computed from each link's value
        already weighted by its own charge_routing_coefficients()
        coefficient -- not an optional mode, always applied, since
        <V_link> alone is NOT proportional to that link's actual energy
        contribution unless its own coefficient happens to be real (a
        link can look strongly nontrivial while contributing exactly
        like vacuum once its own coefficient is accounted for, and vice
        versa). For vacuum (no charges) this weighting is a no-op.
    max_val, min_val : colour-scale bounds, defaulted per color_by by
        _link_color_mapper (fixed rather than auto-scaled from the data,
        so different plaquettes/g values stay directly comparable).
    """
    if plaquette_label is None:
        plaquette_label = nearest_center_plaquette(Lx, Ly, shape, scale)

    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fdimarca/projects/5_Z3"

    ten_file = f"{drive_path}/tensors.hdf5"

    engine = load_engine_from_tensor(
        ten_file, Lx, Ly, shape, chi, g,
        bound_state=bound_state, R=R, precision=precision,
        chargesx=xs, chargesy=ys,
    )
    links = link_electric_fields(engine, plaquette_label, Lx, Ly, shape)

    _, order, _ = label_plaquettes(Lx, Ly, shape)
    centroid_by_label = {
        label: plaquette_centroid(i, j, tri_type, scale)
        for label, i, j, tri_type in order
    }
    anchor_i, anchor_j, anchor_type = next(
        (i, j, t) for lbl, i, j, t in order if lbl == plaquette_label
    )
    anchor_centroid = centroid_by_label[plaquette_label]

    mapper, metric_fn, cbar_label, cmap, norm = _link_color_mapper(
        color_by, cmap_name, min_val, max_val
    )
    # Charge-routing weighting is always applied, not an optional mode --
    # <V_link> alone isn't physically meaningful once charges are present
    # (see _link_color_mapper's docstring). For vacuum this is a no-op
    # (coeff_link=1 everywhere).
    coeffs = charge_routing_coefficients(
        Lx, Ly, shape, bound_state=bound_state, R=R, chargesx=xs, chargesy=ys
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # full lattice, faint, for geographic context
    all_sites = set()
    for label, i, j, tri_type in order:
        verts = triangle_vertices(i, j, tri_type, scale)
        poly = mpatches.Polygon(
            verts, closed=True, facecolor="none",
            edgecolor=lattice_edge_color, linewidth=lattice_edge_lw,
        )
        ax.add_patch(poly)
        if tri_type == 0:
            all_sites.update([(i, j), (i + 1, j), (i, j + 1)])
        else:
            all_sites.update([(i, j), (i, j + 1), (i - 1, j + 1)])

    if show_vertices:
        pts = np.array([site_pos(i, j, scale) for i, j in all_sites])
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=vertex_size, color=vertex_color, linewidths=0, zorder=2,
        )

    # link segments
    for (i, j, d), info in links.items():
        metric_value = coeffs[(i, j, d)] * info["value"]
        color = mapper.to_rgba(metric_fn(metric_value))
        if info["neighbor"] is not None:
            end = centroid_by_label[info["neighbor"]]
        else:
            p0, p1 = link_site_endpoints(i, j, d)
            midpoint = (site_pos(*p0, scale) + site_pos(*p1, scale)) / 2
            direction = midpoint - anchor_centroid
            unit_dir = direction / (np.linalg.norm(direction) + 1e-12)
            end = anchor_centroid + unit_dir * (np.linalg.norm(direction) + stub_length)
        ax.plot(
            [anchor_centroid[0], end[0]], [anchor_centroid[1], end[1]],
            color=color, linewidth=link_lw, solid_capstyle="round", zorder=3,
        )

    # highlight the anchor plaquette
    anchor_verts = triangle_vertices(anchor_i, anchor_j, anchor_type, scale)
    ax.add_patch(mpatches.Polygon(
        anchor_verts, closed=True, facecolor="none",
        edgecolor=anchor_edge_color, linewidth=anchor_lw, zorder=4,
    ))

    ax.autoscale_view()

    # ── colorbar ──────────────────────────────────────────────────────────
    ax_cb = fig.add_axes([0.05, 0.25, 0.025, 0.5])
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)
    cb.outline.set_visible(False)
    ax_cb.yaxis.tick_left()
    ax_cb.set_title(cbar_label, fontsize=9, pad=6)

    fig.tight_layout()

    if save:
        out = (
            f"{drive_path}/figures/"
            f"links_{Lx}_{Ly}_{shape}_{bound_state}_cxs_{xs}_cys_{ys}_R_{R}_"
            f"g_{g:.{precision}f}_chi_{chi}_plaq_{plaquette_label}_{color_by}.pdf"
        )
        plt.savefig(out, transparent=True, bbox_inches="tight")
        print(f"Saved to {out}")

    return fig, ax, links


def plot_lattice_links(
    Lx, Ly, shape, bound_state, R, g, chi, device,
    xs=None,
    ys=None,
    precision=3,
    color_by="abs",
    max_val=None,
    min_val=None,
    scale=1.0,
    cmap_name=None,
    figsize=(10, 9),
    link_lw=14.0,
    link_shrink=0.15,
    show_vertices=True,
    vertex_size=25,
    vertex_color="0.15",
    show_charges=True,
    charge_color="crimson",
    charge_marker="*",
    charge_size=280,
    force_recompute=False,
    save=True,
):
    """
    Draw the electric-field <V.v> value on every link of the whole
    lattice, directly on the direct-lattice geometry (site (i,j) to its
    neighbor site, per label_links / link_site_endpoints) -- unlike
    plot_plaquette_links, which draws on the plaquette dual graph around
    one chosen plaquette.

    Loads a saved TTN checkpoint (tensors.hdf5) and gets every link's
    value via get_lattice_electric_fields, which caches the result inside
    the checkpoint's own tensors.hdf5 group -- computing every link is
    expensive (each one may move the canonical center and re-prime the
    whole tree), so this only pays that cost once per (config, g, chi);
    later calls just read the cache. Pass force_recompute=True to ignore
    an existing cache (e.g. after re-running the checkpoint at the same
    g/chi).

    color_by : "abs" (magnitude, bounded in [0,1], default), "arg" (phase
        in radians, [-pi,pi]), or "real" (Re(coeff_link*<V_link>), the
        g-independent Hamiltonian energy contribution, [-1,1]) -- see
        _link_color_mapper. Every mode is computed from each link's value
        already weighted by its own charge_routing_coefficients()
        coefficient -- not an optional mode, always applied, since
        <V_link> alone is NOT proportional to that link's actual energy
        contribution unless its own coefficient happens to be real (see
        the R=2 direct-string tests: the unweighted raw value lit up 6
        links -- 2 genuine + 4 Dirac-string construction artifacts --
        while weighting isolates exactly the 2 genuine ones). For vacuum
        (no charges) this weighting is a no-op. Deep confinement shows up
        as a discrete flux-quantum PHASE shift at UNIT magnitude (e.g.
        g=-10: every link sits at |value|=1, so "abs" looks uniformly
        bright even directly on the string -- "arg" or "real" is what
        actually reveals it there).
    max_val, min_val : colour-scale bounds, defaulted per color_by by
        _link_color_mapper.

    Parameters mirror plot_plaquette_links where they overlap.
    """
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fdimarca/projects/5_Z3"

    ten_file = f"{drive_path}/tensors.hdf5"

    links = get_lattice_electric_fields(
        ten_file, Lx, Ly, shape, chi, g,
        bound_state=bound_state, R=R, precision=precision,
        chargesx=xs, chargesy=ys, force_recompute=force_recompute,
    )

    mapper, metric_fn, cbar_label, cmap, norm = _link_color_mapper(
        color_by, cmap_name, min_val, max_val
    )
    # Charge-routing weighting is always applied, not an optional mode --
    # <V_link> alone isn't physically meaningful once charges are present
    # (see _link_color_mapper's docstring). For vacuum this is a no-op
    # (coeff_link=1 everywhere).
    coeffs = charge_routing_coefficients(
        Lx, Ly, shape, bound_state=bound_state, R=R, chargesx=xs, chargesy=ys
    )
    resolved_chargesx, resolved_chargesy = xs, ys
    if bound_state is not None and resolved_chargesx is None:
        resolved_chargesx, resolved_chargesy = get_coord_charges(
            Lx, Ly, bound_state=bound_state, shape=shape, R=R
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    all_sites = set()
    for (i, j, d), info in links.items():
        p0, p1 = link_site_endpoints(i, j, d)
        all_sites.update([p0, p1])
        pos0, pos1 = site_pos(*p0, scale), site_pos(*p1, scale)
        metric_value = coeffs[(i, j, d)] * info["value"]
        color = mapper.to_rgba(metric_fn(metric_value))
        # shrink each segment in from both ends so thick links converging
        # on a shared vertex leave a visible gap instead of piling into a
        # blob that swallows the vertex marker
        direction = pos1 - pos0
        draw0 = pos0 + direction * link_shrink
        draw1 = pos1 - direction * link_shrink
        ax.plot(
            [draw0[0], draw1[0]], [draw0[1], draw1[1]],
            color=color, linewidth=link_lw, solid_capstyle="round", zorder=2,
        )

    if show_vertices:
        pts = np.array([site_pos(i, j, scale) for i, j in all_sites])
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=vertex_size, color=vertex_color, linewidths=0, zorder=3,
        )

    if show_charges and resolved_chargesx is not None:
        charge_pts = np.array([
            site_pos(cx, cy, scale) for cx, cy in zip(resolved_chargesx, resolved_chargesy)
        ])
        ax.scatter(
            charge_pts[:, 0], charge_pts[:, 1],
            s=charge_size, color=charge_color, marker=charge_marker,
            edgecolors="white", linewidths=0.8, zorder=4,
        )

    ax.autoscale_view()

    # ── colorbar ──────────────────────────────────────────────────────────
    ax_cb = fig.add_axes([0.05, 0.25, 0.025, 0.5])
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm)
    cb.outline.set_visible(False)
    ax_cb.yaxis.tick_left()
    ax_cb.set_title(cbar_label, fontsize=9, pad=6)

    fig.tight_layout()

    if save:
        out = (
            f"{drive_path}/figures/"
            f"lattice_links_{Lx}_{Ly}_{shape}_{bound_state}_cxs_{xs}_cys_{ys}_R_{R}_"
            f"g_{g:.{precision}f}_chi_{chi}_{color_by}.pdf"
        )
        plt.savefig(out, transparent=True, bbox_inches="tight")
        print(f"Saved to {out}")

    return fig, ax, links


# ── GIF maker ────────────────────────────────────────────────────────────────

def make_gif(
    g_values,
    Lx, Ly, shape, bound_state, R, chi, device,
    xs=None,
    ys=None,
    precision=2,
    log_scale=True,
    scale=1.0,
    cmap_name="GnBu",
    figsize=(10, 9),
    edge_lw=0.8,
    edge_color="white",
    show_vertices=True,
    vertex_size=40,
    vertex_color="0.25",
    charge_color="crimson",
    charge_size=300,
    charge_marker="*",
    charge_zorder=10,
    fps=3,
    dpi=120,
    g_label_pos=(0.72, 0.04),
    g_label_fontsize=14,
):
    """
    Render one frame per value in g_values and stitch them into an
    animated GIF.

    The colour scale (min_ee / max_ee) is fixed globally across ALL frames
    so that colours are directly comparable across g values.

    Parameters
    ----------
    g_values        : iterable of g values, e.g. np.arange(-20, 1, 1)
    fps             : frames per second in the output GIF
    dpi             : raster resolution per frame (120 ≈ good screen quality)
    g_label_pos     : (x, y) in axes-fraction coords for the "g = …" label
    g_label_fontsize: font size of that label
    All other parameters mirror plot_plaquettes.

    Returns
    -------
    out_path : str — path to the saved GIF
    """
    import imageio.v2 as imageio
    import io

    g_values = list(g_values)

    # ── drive path ────────────────────────────────────────────────────────────
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fdimarca/projects/5_Z3"

    # ── pass 1: read all CSVs and compute the global EE range ─────────────────
    N = nplaqs(Lx, Ly, shape)
    _, order, _ = label_plaquettes(Lx, Ly, shape)

    all_values = []
    dfs = {}
    for g in g_values:
        folder = get_folder(Lx, Ly, shape, bound_state=bound_state, chargesx=xs, chargesy=ys, R=R)
        csv_path = f"{folder}/g_{g:.{precision}f}/run_chi-{chi}/basic.csv"
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Warning: missing data for g={g:.{precision}f}, skipping.")
            continue
        dfs[g] = df
        ee = {}
        for _, row in df.iterrows():
            n1, n2 = int(row["node1"]), int(row["node2"])
            if n1 < N:
                ee[n1] = row["entanglement"]
            elif n2 < N:
                ee[n2] = row["entanglement"]
        all_values.extend(ee.get(lbl, 0.0) for lbl, *_ in order)

    all_values = np.array(all_values)
    global_max = all_values.max()
    if log_scale:
        global_min = all_values[all_values > 0].min()
    else:
        global_min = all_values.min()

    print(f"Global EE range: [{global_min:.3e}, {global_max:.3e}]")

    # ── pass 2: render one frame per g value ──────────────────────────────────
    frames = []
    for g in g_values:
        if g not in dfs:
            continue

        fig, ax = plot_plaquettes(
            Lx=Lx, Ly=Ly, shape=shape, bound_state=bound_state,
            R=R, g=g, chi=chi, device=device, precision=precision,
            xs=xs, ys=ys, min_ee=global_min, max_ee=global_max,
            log_scale=log_scale, scale=scale, cmap_name=cmap_name,
            figsize=figsize, edge_lw=edge_lw, edge_color=edge_color,
            show_vertices=show_vertices, vertex_size=vertex_size,
            vertex_color=vertex_color, charge_color=charge_color,
            charge_size=charge_size, charge_marker=charge_marker,
            charge_zorder=charge_zorder,
            save=False,
        )

        # g annotation in the corner
        ax.annotate(
            f"$g = {g:.{precision}f}$",
            xy=g_label_pos, xycoords="axes fraction",
            fontsize=g_label_fontsize, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    if not frames:
        raise RuntimeError("No frames rendered — check that CSV files exist for the requested g values.")

    # ── write GIF ─────────────────────────────────────────────────────────────
    out_path = (
        f"{drive_path}/figures/"
        f"plaq_{Lx}_{Ly}_{shape}_{bound_state}_{R}_{chi}_sweep_g.gif"
    )
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"GIF saved -> {out_path}  ({len(frames)} frames @ {fps} fps)")
    return out_path


def make_links_gif(
    g_values,
    Lx, Ly, shape, bound_state, R, chi, device,
    xs=None,
    ys=None,
    precision=3,
    color_by="abs",
    max_val=None,
    min_val=None,
    scale=1.0,
    cmap_name=None,
    figsize=(10, 9),
    link_lw=14.0,
    link_shrink=0.15,
    show_vertices=True,
    vertex_size=25,
    vertex_color="0.15",
    show_charges=True,
    charge_color="crimson",
    charge_marker="*",
    charge_size=280,
    fps=3,
    dpi=120,
    g_label_pos=(0.72, 0.04),
    g_label_fontsize=14,
    force_recompute=False,
):
    """
    Render one plot_lattice_links frame per value in g_values and stitch
    them into an animated GIF -- the link-based analog of make_gif (which
    animates plot_plaquettes' entanglement-entropy view instead).

    Unlike make_gif, no separate "compute the global color range" pass is
    needed: plot_lattice_links' color scales are already fixed per
    color_by (see _link_color_mapper: "abs" always [0,1], "arg" always
    [-pi,pi], "real" always [-1,1]), not auto-scaled from the data, so
    frames stay directly comparable across g as long as color_by/min_val/
    max_val are held fixed for the whole sweep (the default).

    Requires a saved tensors.hdf5 checkpoint (with gauge_tensor) for
    every g in g_values already present -- this does NOT run any ground-
    state search itself, only loads and queries already-converged
    checkpoints (missing ones are skipped with a warning, matching
    make_gif's handling of missing basic.csv files).

    Parameters
    ----------
    g_values : iterable of g values, e.g. np.arange(-10, -1, 1)
    fps, dpi, g_label_pos, g_label_fontsize : as in make_gif.
    All other parameters mirror plot_lattice_links.

    Returns
    -------
    out_path : str -- path to the saved GIF
    """
    import imageio.v2 as imageio
    import io

    g_values = list(g_values)
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fdimarca/projects/5_Z3"

    frames = []
    for g in g_values:
        try:
            fig, ax, links = plot_lattice_links(
                Lx=Lx, Ly=Ly, shape=shape, bound_state=bound_state,
                R=R, g=g, chi=chi, device=device, precision=precision,
                xs=xs, ys=ys, color_by=color_by, max_val=max_val, min_val=min_val,
                scale=scale, cmap_name=cmap_name, figsize=figsize,
                link_lw=link_lw, link_shrink=link_shrink,
                show_vertices=show_vertices, vertex_size=vertex_size, vertex_color=vertex_color,
                show_charges=show_charges, charge_color=charge_color,
                charge_marker=charge_marker, charge_size=charge_size,
                force_recompute=force_recompute, save=False,
            )
        except (KeyError, FileNotFoundError) as e:
            print(f"Warning: missing checkpoint for g={g:.{precision}f}, skipping ({e}).")
            continue

        ax.annotate(
            f"$g = {g:.{precision}f}$",
            xy=g_label_pos, xycoords="axes fraction",
            fontsize=g_label_fontsize, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    if not frames:
        raise RuntimeError(
            "No frames rendered — check that tensors.hdf5 checkpoints exist for the requested g values."
        )

    out_path = (
        f"{drive_path}/figures/"
        f"lattice_links_gif_{Lx}_{Ly}_{shape}_{bound_state}_cxs_{xs}_cys_{ys}_R_{R}_"
        f"chi_{chi}_{color_by}_sweep_g.gif"
    )
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"GIF saved -> {out_path}  ({len(frames)} frames @ {fps} fps)")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Single frame
    # plot_plaquettes(
    #     Lx=13, Ly=3,
    #     shape="parallelogram",
    #     bound_state="meson",
    #     R=1, g=-10, chi=40,
    #     device="pc", precision=3,
    #     # xs=[2,6],
    #     # ys=[1,1],
    #     log_scale=False,
    # )
    plot_lattice_links(
            Lx=3, Ly=5,
            # shape="parallelogram",
            shape="hexagon",
            # bound_state="meson",
            bound_state="baryon",
            R=1, g=-10, chi=27,
            device="pc", precision=3,
            color_by="arg",
            xs=[-1, -1, 2],
            ys=[1, 4, 1],
            link_lw=10.0,
        )


    # make_links_gif(
    #     g_values=np.linspace(-10, 0, 21),
    #     Lx=3, Ly=5,
    #     # shape="parallelogram",
    #     shape="hexagon",
    #     # bound_state="meson",
    #     bound_state="baryon",
    #     R=1, g=-10, chi=27,
    #     device="pc", precision=3,
    #     color_by="abs",
    #     # xs=[2,6],
    #     # ys=[1,1],
    #     # link_lw=5.0,
    #     )
    
    # GIF sweeping g from -20 to 0
    # g_values = [0.1,0.14,0.18,0.21,0.25,0.29,0.33,0.37,0.4,0.44,0.48,0.56,0.59,0.63,0.67,0.71,0.75,0.78,0.82,0.86,0.90]
    # g_values = np.linspace(0.1,2,51)
    # g_values = [-g for g in g_values]

    # make_gif(
    #     g_values=g_values,
    #     Lx=3, Ly=5,
    #     shape="hexagon",
    #     bound_state="meson",
    #     R=1, chi=40,
    #     device="pc", precision=2,
    #     log_scale=False,
    #     fps=4,
    #     dpi=150,
    # )
    # make_gif(
    #     g_values=g_values,
    #     Lx=3, Ly=3,
    #     shape="parallelogram",
    #     bound_state="meson",
    #     R=1, chi=40,
    #     device="pc", precision=2,
    #     log_scale=False,
    #     fps=4,
    #     dpi=150,
    # )

    plt.show()