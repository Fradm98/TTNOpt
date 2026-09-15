"""
z3_graph_levels_combined.py
-----------------------------
Renders two coarse-graining levels of the (now-fixed) production
create_graph() side by side, for a given lattice -- used to eyeball-verify
the tree structure after the reorder_by_next_level fix applied to
Z3_funcs/create_graph_file.py:78.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from Z3_funcs.create_graph_file import create_graph
from Z3_funcs.lattice_plaquettes import label_plaquettes
from Z3_funcs.visualize_plaquettes import triangle_vertices


def parent_map_from(ttn_labels):
    pm = {}
    for row in ttn_labels:
        row = [int(x) for x in row]
        if len(row) == 3:
            c0, c1, p = row
            pm[c0] = p
            pm[c1] = p
    return pm


def ancestor_at_or_above(label, parent, threshold):
    cur = label
    while cur < threshold and cur in parent:
        cur = parent[cur]
    return cur


def plot_level(ax, Lx, Ly, shape, n, parent, threshold, title, merge_pairs=None):
    labels_map, _, bounds = label_plaquettes(Lx, Ly, shape)
    inv = {v: k for k, v in labels_map.items()}
    cmap = plt.get_cmap("tab20")
    groups_seen = {}
    merge_lookup = {}
    if merge_pairs:
        for gid, pair in enumerate(merge_pairs):
            for lab in pair:
                merge_lookup[lab] = gid
    for lab in range(n):
        ijt = inv[lab]
        anc = ancestor_at_or_above(lab, parent, threshold)
        key = merge_lookup.get(anc, anc)
        if key not in groups_seen:
            groups_seen[key] = len(groups_seen)
        color = cmap(groups_seen[key] % 20)
        verts = triangle_vertices(ijt[0], ijt[1], ijt[2])
        ax.add_patch(mpatches.Polygon(verts, closed=True, facecolor=color, edgecolor="black", linewidth=0.6))
        c = verts.mean(axis=0)
        ax.text(c[0], c[1], str(lab), ha="center", va="center", fontsize=6)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    return len(groups_seen)


def combined(Lx, Ly, shape, n, level_a, level_b, out_tag, merge_pairs_a=None, merge_pairs_b=None):
    ttn = create_graph(Lx, Ly, shape)
    ttn = [[int(x) for x in row] for row in ttn]
    parent = parent_map_from(ttn)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    na = plot_level(axes[0], Lx, Ly, shape, n, parent, level_a[0],
                     f"{Lx}x{Ly} {shape} -- {level_a[1]}", merge_pairs_a)
    nb = plot_level(axes[1], Lx, Ly, shape, n, parent, level_b[0],
                     f"{Lx}x{Ly} {shape} -- {level_b[1]}", merge_pairs_b)
    fig.tight_layout()
    out = f"/Users/fradm/Desktop/TTNOpt/diagnostics/{out_tag}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved: {out}  (panel A groups={na}, panel B groups={nb})")


if __name__ == "__main__":
    # 5x5 parallelogram: level1 (8) + final (2, via merged {56,57}/{58,59})
    combined(5, 5, "parallelogram", 32,
             level_a=(48, "cg level 1 (8 plaquettes)"),
             level_b=(56, "final level (2 plaquettes)"),
             out_tag="graph_final_5x5_parallelogram",
             merge_pairs_b=[(56, 57), (58, 59)])

    # 5x9 hexagon: level1 (24) + final (6)
    combined(5, 9, "hexagon", 96,
             level_a=(144, "cg level 1 (24 plaquettes)"),
             level_b=(180, "final level (6 plaquettes)"),
             out_tag="graph_final_5x9_hexagon")

    # 13x5 parallelogram: cg level 2 (6) + final level (3)
    combined(13, 5, "parallelogram", 96,
             level_a=(180, "cg level 2 (6 plaquettes)"),
             level_b=(186, "final level (3 plaquettes)"),
             out_tag="graph_final_13x5_parallelogram")

    # 13x9 parallelogram: cg level 2 (12) + final level (3)
    combined(13, 9, "parallelogram", 192,
             level_a=(360, "cg level 2 (12 plaquettes)"),
             level_b=(378, "final level (3 plaquettes)"),
             out_tag="graph_final_13x9_parallelogram")
