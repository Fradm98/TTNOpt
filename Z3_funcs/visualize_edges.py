# import packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

import numpy as np
import pandas as pd

from Z3_funcs.create_graph_file import nplaqs
from Z3_funcs.utils import get_folder


def plot(Lx, Ly, shape, bound_state, R, g, chi, device, precision=3,max_ee=0.015):
    if device == "pc":
        drive_path = "D:work/projects/5_Z3"
    elif device == "ngt":
        drive_path = "/eos/user/f/fdimarca/projects/5_Z3"
    elif device == "presto":
        drive_path = "/home/fdimarca/projects/5_Z3"

    # settings
    edge_colormap = "GnBu"
    edge_colormap_modified = mpl.colormaps.get_cmap("GnBu")
    edge_colormap_modified = edge_colormap_modified(np.linspace(0.2, 1, 256))
    edge_colormap = mpl.colors.LinearSegmentedColormap.from_list("GnBu_custom", edge_colormap_modified)
    figsize = (12, 10)

    # adjust size settings according to the number of qubits
    N = nplaqs(Lx,Ly,shape)
    default_node_size = 800 / np.log2(N)
    leaf_node_size = 1400 / np.log2(N) 
    edge_width = 24 / np.log2(N)
    font_size = 72 / np.log2(N)

    # read csv file
    folder = get_folder(Lx, Ly, shape, bound_state=bound_state, R=R)
    df = pd.read_csv(f"{folder}/g_{g:.{precision}f}/run_chi-{chi}/basic.csv")

    # create networkx tree graph
    G = nx.Graph()
    nodes = set(np.concatenate([df["node1"].values, df["node2"].values]))
    leaf_nodes, default_nodes = [], []
    for node in nodes:
        G.add_node(node, label=node)
        if int(node) < N:
            leaf_nodes.append(node)
        else:
            default_nodes.append(node)
    for i, row in df.iterrows():
        G.add_edge(int(row["node1"]), int(row["node2"]), weight=row["entanglement"])

    # define node labels
    node_labels = {}
    for node in leaf_nodes:
        node_labels[node] = node

    # define edge colors
    # edge_weights = [G.edges[e]["weight"] for e in G.edges]
    # ew_range = 0.0, max_ee
    # enorm = mpl.colors.Normalize(*ew_range, clip=True)
    # emapper = mpl.cm.ScalarMappable(norm=enorm, cmap=edge_colormap)
    # edge_colors = [emapper.to_rgba(x) for x in edge_weights]

    # define edge colors
    edge_weights = [G.edges[e]["weight"] for e in G.edges]
    # ew_range = (min(edge_weights), max_ee)  # LogNorm requires positive values
    ew_range = (min(edge_weights), max(edge_weights))  # LogNorm requires positive values
    # ew_range = (1e-3, max_ee)  # LogNorm requires positive values
    enorm = mcolors.LogNorm(vmin=ew_range[0], vmax=ew_range[1], clip=True)
    emapper = mpl.cm.ScalarMappable(norm=enorm, cmap=edge_colormap)
    edge_colors = [emapper.to_rgba(x) for x in edge_weights]

    # adjust node positions
    pos = nx.nx_agraph.graphviz_layout(G, prog="neato")

    # draw graph
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    nx.draw_networkx_nodes(G, pos, nodelist=default_nodes, node_size=default_node_size, node_color="black", ax=ax)
    #nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_size=leaf_node_size, node_color="none", edgecolors="blue", ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_width, edge_color=edge_colors, ax=ax, alpha=1.0)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=font_size, ax=ax, font_family="Times New Roman")

    # plot colorbar
    ax_l = fig.add_axes([0.05, 0.25, 0.02, 0.5])
    cb_l = mpl.colorbar.ColorbarBase(
        ax_l, cmap=edge_colormap, norm=enorm
    )
    cb_l.outline.set_visible(False)
    ax_l.yaxis.tick_left()
    ax_l.set(title="Entanglement\n entropy")

    plt.savefig(f"{drive_path}/figures/TTN_{Lx}_{Ly}_{shape}_{bound_state}_{R}_{g:.{precision}f}_{chi}.pdf", transparent=True)

if __name__ == "__main__":
    g = -0.48
    precision = 3
    # g_values = [10,20]

    Lx = 3
    Ly = 5
    shape = "hexagon"
    # shape = "parallelogram"
    bound_state = "baryon"
    bound_state = "meson"
    # bound_state = None
    R = 1
    chi = 40
    precision = 2
    device = "pc"
    plot(Lx, Ly, shape, bound_state=bound_state, R=R, g=g, chi=chi, device=device, precision=precision, max_ee=1)