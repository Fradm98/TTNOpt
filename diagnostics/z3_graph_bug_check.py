"""
z3_graph_bug_check.py
----------------------
Standalone (non-destructive) check of the suspected create_graph() bug:
the "parent labels" grouping call (create_graph_file.py:78,
group_plaquettes_coarse_gen(cgl=0, ...) on the reduced lattice) is never
reordered via reorder_by_next_level the way the children grouping is
(create_graph_file.py:58) -- so the position-based zip() pairing the code
relies on silently mismatches whenever the two sweeps' anchor visitation
order diverges (which happens whenever the top-level cluster split doesn't
align with a clean row split).

This reimplements create_graph() with an optional fix (reorder the
line-78 grouping the same way as the children), WITHOUT touching the real
Z3_funcs/create_graph_file.py, and compares buggy vs fixed edge lists +
renders both as plaquette-colored-by-ancestor plots for eyeballing.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Z3_funcs.lattice_plaquettes import (
    group_plaquettes_coarse_gen, reorder_by_next_level, divide_hexagon_plaquettes,
    verify_coarse_graining_levels, cg_lattice, nplaqs,
)
from Z3_funcs.visualize_plaquettes import triangle_vertices


def create_graph_impl(Lx, Ly, shape="parallelogram", fix=False):
    dof = nplaqs(Lx, Ly, shape)
    ttn_labels = []
    cgl_max = verify_coarse_graining_levels(Lx, Ly, shape)

    Lx_cg, Ly_cg = Lx, Ly
    nplaqs_cgl = -dof / 2
    for cgl in range(cgl_max):
        groups_cgl, _ = group_plaquettes_coarse_gen(Lx=Lx_cg, Ly=Ly_cg, shape=shape, cgl=1, off=int(dof + 2 * nplaqs_cgl))
        groups_max, _ = group_plaquettes_coarse_gen(Lx=Lx_cg, Ly=Ly_cg, shape=shape, cgl=cgl_max - cgl, off=int(dof + 2 * nplaqs_cgl))

        groups_cgl_r, map_1_max = reorder_by_next_level(group_k=groups_cgl, group_k1=groups_max)

        groups_cgl_flat = np.array([g['plaquettes'] for g in groups_cgl_r]).flatten()
        groups_cgl_binary = np.array_split(groups_cgl_flat, len(groups_cgl_flat) // 2)

        if cgl >= 1:
            dof += 3 * nplaqs_cgl

        Lx_cg, Ly_cg = cg_lattice(Lx, Ly, shape, cgl + 1)
        nplaqs_cgl = nplaqs(Lx_cg, Ly_cg, shape)

        sites_cgl_child = np.array([i for i in range(dof, dof + 2 * nplaqs_cgl)])
        ttn_labels += [binary.tolist() + [int(i)] for i, binary in zip(sites_cgl_child, groups_cgl_binary)]

        groups_cgl_next, _ = group_plaquettes_coarse_gen(Lx=Lx_cg, Ly=Ly_cg, shape=shape, cgl=0, off=int(dof + 2 * nplaqs_cgl))

        if fix:
            # THE FIX: reorder this grouping the same way the children were
            # reordered, against the SAME next-level clustering reference
            # (groups_max, re-expressed at this new off/scale) -- keeps the
            # position-based zip() below valid.
            groups_max_next, _ = group_plaquettes_coarse_gen(
                Lx=Lx_cg, Ly=Ly_cg, shape=shape, cgl=cgl_max - cgl - 1, off=int(dof + 2 * nplaqs_cgl)
            ) if (cgl_max - cgl - 1) > 0 else (groups_cgl_next, None)
            if (cgl_max - cgl - 1) > 0:
                groups_cgl_next, _ = reorder_by_next_level(group_k=groups_cgl_next, group_k1=groups_max_next)

        groups_cgl_flat = np.array([g['plaquettes'] for g in groups_cgl_next]).flatten()
        sites_cgl_binary = np.array_split(sites_cgl_child, len(sites_cgl_child) // 2)
        ttn_labels += [binary.tolist() + [int(i)] for i, binary in zip(groups_cgl_flat, sites_cgl_binary)]

    if shape == "hexagon":
        sites_cgl_binary = divide_hexagon_plaquettes(Lx_cg, Ly_cg, off=int(dof + 2 * nplaqs_cgl))
        sites_cgl_child = np.array([i for i in range(int(dof + 3 * nplaqs_cgl), int(dof + 3 * nplaqs_cgl + 3))])
        ttn_labels += [binary + [int(i)] for i, binary in zip(sites_cgl_child, sites_cgl_binary)]
        ttn_labels += [sites_cgl_child.tolist()]

    if shape == "parallelogram":
        dof += 3 * nplaqs_cgl
        while (nplaqs_cgl % 2 == 0) and (nplaqs_cgl > 2):
            sites_cgl_binary = np.array_split(groups_cgl_flat, len(groups_cgl_flat) // 2)
            nplaqs_cgl = int(nplaqs_cgl / 2)
            sites_cgl_child = np.array([i for i in range(int(dof), int(dof + nplaqs_cgl))])
            ttn_labels += [binary.tolist() + [int(i)] for i, binary in zip(sites_cgl_child, sites_cgl_binary)]
            groups_cgl_flat = sites_cgl_child.copy()
            dof += nplaqs_cgl

        if nplaqs_cgl == 2:
            ttn_labels[-1][-1] = ttn_labels[-2][-1]
        if nplaqs_cgl == 3:
            ttn_labels += [sites_cgl_child]

    return ttn_labels


def edges_to_parent_map(ttn_labels):
    parent = {}
    for c0, c1, p in ttn_labels:
        parent[int(c0)] = int(p)
        parent[int(c1)] = int(p)
    return parent


def ancestor_at_or_above(label, parent, threshold):
    """Walk up until we hit a label >= threshold (first ancestor at/after
    a given tree 'generation' cutoff -- used to color base plaquettes by
    their coarse-grained group)."""
    cur = label
    while cur < threshold and cur in parent:
        cur = parent[cur]
    return cur


def compare(Lx, Ly, shape, group_threshold_low, group_threshold_high, title_tag):
    buggy = create_graph_impl(Lx, Ly, shape, fix=False)
    fixed = create_graph_impl(Lx, Ly, shape, fix=True)

    diffs = [(b, f) for b, f in zip(buggy, fixed) if b != f]
    print(f"\n=== {title_tag}: {len(diffs)} differing tree rows out of {len(buggy)} ===")
    for b, f in diffs:
        print(f"  buggy: {b}   fixed: {f}")

    n = nplaqs(Lx, Ly, shape)
    parent_buggy = edges_to_parent_map(buggy)
    parent_fixed = edges_to_parent_map(fixed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, parent, label in [(axes[0], parent_buggy, "current (buggy)"), (axes[1], parent_fixed, "fixed")]:
        cmap = plt.get_cmap("tab20")
        groups_seen = {}
        for lab in range(n):
            i, j, t = None, None, None
        # recompute (i,j,t) from label_plaquettes for plotting
        from Z3_funcs.lattice_plaquettes import label_plaquettes
        labels_map, _, bounds = label_plaquettes(Lx, Ly, shape)
        inv = {v: k for k, v in labels_map.items()}
        for lab in range(n):
            ijt = inv[lab]
            anc = ancestor_at_or_above(lab, parent, group_threshold_low)
            anc_top = ancestor_at_or_above(lab, parent, group_threshold_high)
            if anc_top not in groups_seen:
                groups_seen[anc_top] = len(groups_seen)
            color = cmap(groups_seen[anc_top] % 20)
            verts = triangle_vertices(ijt[0], ijt[1], ijt[2])
            ax.add_patch(mpatches.Polygon(verts, closed=True, facecolor=color, edgecolor="black", linewidth=0.8))
            c = verts.mean(axis=0)
            ax.text(c[0], c[1], str(lab), ha="center", va="center", fontsize=7)
        ax.set_title(f"{title_tag} -- {label}\ncolored by ancestor >= {group_threshold_high}")
        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
    fig.tight_layout()
    out = f"/Users/fradm/Desktop/TTNOpt/diagnostics/graph_bug_check_{title_tag}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved: {out}")
    return buggy, fixed


if __name__ == "__main__":
    b33, f33 = compare(3, 3, "parallelogram", group_threshold_low=8, group_threshold_high=8, title_tag="3x3_parallelogram")
    b55, f55 = compare(5, 5, "parallelogram", group_threshold_low=48, group_threshold_high=56, title_tag="5x5_parallelogram")
    b35, f35 = compare(3, 5, "hexagon", group_threshold_low=24, group_threshold_high=24, title_tag="3x5_hexagon")
    b59, f59 = compare(5, 9, "hexagon", group_threshold_low=96, group_threshold_high=96 + 24, title_tag="5x9_hexagon")
