import pandas as pd
import numpy as np
from ttnopt.src.TTN import random_tree as rt
from ttnopt.src.TTN import TreeTensorNetwork as ttn

def get_rnd_tree(Lx,Ly,shape,path,chi):
    # load the graph structure of the tensors
    edges = pd.read_csv(f"{path}/edges_{Lx}_{Ly}_{shape}.dat", header=None)
    links = [edges.iloc[i, :].values.tolist() for i in range(len(edges))]

    # physical dimension
    d = 3

    tree = ttn(links, top_edge_id=links[-1][-1])
    tensors = [np.zeros((d,d,chi)) if edge[0] in tree.physical_edges else np.zeros((chi,chi,chi)) for edge in tree.edges]

    rnd_tree = rt(links, tensors, top_edge_id=links[-1][-1])
    rnd_tree.init_random(edges=tree.edges, top_edge_id=tree.top_edge_id, edge_dims=rnd_tree.edge_dims, init_bond_dimension=chi)
    return rnd_tree