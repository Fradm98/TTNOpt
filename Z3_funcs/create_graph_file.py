from Z3_funcs.lattice_plaquettes import (group_plaquettes_coarse_gen, 
                                         make_charges_array,
                                         site_to_array_index,
                                         evaluate_link_coefficients, 
                                         verify_coarse_graining_levels,
                                         reorder_by_next_level,
                                         divide_hexagon_plaquettes,
                                         get_coord_charges,
                                         nplaqs,
                                         cg_lattice)
import numpy as np

omega = np.exp(1j*2*np.pi/3)

def create_graph(Lx: int,Ly: int, shape: str="parallelogram"):
    """
    create_graph

    This function creates the graph of the TTN of a (direct) triangular lattice with the 
    shape of a parallelogram or hexagon.
    To do so, it checks two things:
        1. valid coarse graining levels
        2. final tensors
    1. The first step involves the coarse graining procedure. The ideal unit lattice found in our case
    is of four (dual) plaquettes. Since a quaternary tree would be expensive, we divide in three
    perfect binary trees (two children and one parent).
    Indeed, with four plaquettes correctly oriented as a up /\\ or down \\/ triangle, we recover coarse-grained
    plaquette. This can be iterated n times.
    2. According to the shape of the lattice, we could end up with just, 2 or 3 coarse-grained plaquettes, or multiple
    of these numbers.
    Let's see all the cases:
        - If nplaqs_cg == 2: then we find the canonical center of our TTN and the graph map is ready!
        - If nplaqs_cg == 3: we can find the canoncial center adding one perfect binary tree
        - If (nplaqs_cg % 2) == 0: build perfect binary trees until you fall in the previous case
        - all other cases are not implemented

    """    
    # -------------------------
    # Coarse Graining procedure
    # -------------------------
    # number of plaquettes in the direct lattice
    dof = nplaqs(Lx, Ly, shape)
    ttn_labels = []
    
    # check maximum coarse graining and if size is compatible with our procedure
    cgl_max = verify_coarse_graining_levels(Lx, Ly, shape)

    # TODO: assert if the cgl exists

    Lx_cg, Ly_cg = Lx, Ly
    nplaqs_cgl = - dof/2
    for cgl in range(cgl_max):
        # find first coarse-grained plaquettes at a certain coarse grained lattice
        groups_cgl, _ = group_plaquettes_coarse_gen(Lx=Lx_cg, Ly=Ly_cg, shape=shape, cgl=1, off=int(dof + 2*nplaqs_cgl))
        groups_max, _ = group_plaquettes_coarse_gen(Lx=Lx_cg, Ly=Ly_cg, shape=shape, cgl=cgl_max-cgl, off=int(dof + 2*nplaqs_cgl))

        # compare with the maximum one to reorder the plaquettes
        groups_cgl_r, map_1_max = reorder_by_next_level(group_k=groups_cgl, group_k1=groups_max)
        
        # build the perfect binary trees children
        groups_cgl_flat = np.array([g['plaquettes'] for g in groups_cgl_r]).flatten()
        groups_cgl_binary = np.array_split(groups_cgl_flat, len(groups_cgl_flat)//2)

        if cgl >= 1:
            # from cgl = 1 we add all previous tensors
            dof += 3*nplaqs_cgl

        # find the coarse grained lattice size and number of plaquettes
        Lx_cg, Ly_cg = cg_lattice(Lx, Ly, shape, cgl+1)
        nplaqs_cgl = nplaqs(Lx_cg, Ly_cg, shape)

        # create the first layer of the perfect binary tree at a certain cgl
        # the four triangles (children) get into couples
        sites_cgl_child = np.array([i for i in range(dof, dof + 2*nplaqs_cgl)])
        ttn_labels += [binary.tolist() + [int(i)] for i, binary in zip(sites_cgl_child, groups_cgl_binary)]

        # get the parent labels of the next coarse-grained lattice
        groups_cgl, _ = group_plaquettes_coarse_gen(Lx=Lx_cg, Ly=Ly_cg, shape=shape, cgl=0, off=int(dof + 2*nplaqs_cgl))
        groups_cgl_flat = np.array([g['plaquettes'] for g in groups_cgl]).flatten()
        # create the second layer of the perfect binary tree at a certain cgl
        # the coupled four triangles (children) get connected to the (parent) coarse-grained sites
        sites_cgl_binary = np.array_split(sites_cgl_child, len(sites_cgl_child)//2)
        ttn_labels += [binary.tolist() + [int(i)] for i, binary in zip(groups_cgl_flat, sites_cgl_binary)]

    # -----------------------
    # Final Tensors procedure
    # -----------------------
    if shape == "hexagon":
        sites_cgl_binary = divide_hexagon_plaquettes(Lx_cg, Ly_cg, off=int(dof + 2*nplaqs_cgl))
        sites_cgl_child = np.array([i for i in range(int(dof + 3*nplaqs_cgl), int(dof + 3*nplaqs_cgl + 3))])
        ttn_labels += [binary + [int(i)] for i, binary in zip(sites_cgl_child, sites_cgl_binary)]
        ttn_labels += [sites_cgl_child.tolist()]

    if shape == "parallelogram":
        dof += 3*nplaqs_cgl
        while (nplaqs_cgl % 2 == 0) and (nplaqs_cgl > 2):
            sites_cgl_binary = np.array_split(groups_cgl_flat, len(groups_cgl_flat)//2)
            nplaqs_cgl = int(nplaqs_cgl / 2)

            sites_cgl_child = np.array([i for i in range(int(dof), int(dof + nplaqs_cgl))])
            ttn_labels += [binary.tolist() + [int(i)] for i, binary in zip(sites_cgl_child, sites_cgl_binary)]
            groups_cgl_flat = sites_cgl_child.copy()
            dof += nplaqs_cgl

    
        if nplaqs_cgl == 2:
            ttn_labels[-1][-1] = ttn_labels[-2][-1]

        if nplaqs_cgl == 3:
            ttn_labels += [sites_cgl_child]


    ttn_labels = [str(label) for tensor in ttn_labels for label in tensor]
    ttn_labels = np.array_split(np.asarray(ttn_labels), len(ttn_labels)//3)
    return ttn_labels


def save_edges_file(Lx: int,Ly: int, shape: str="parallelogram", tensor_folder: str=None):
    # create edge list
    edges = create_graph(Lx=Lx, Ly=Ly, shape=shape)
    
    with open(f"{tensor_folder}/edges_{Lx}_{Ly}_{shape}.dat", "w") as f:
        for e in edges:
            string = ",".join(e)
            f.write(string + "\n")

def save_basic_file(Lx: int,Ly: int, shape: str="parallelogram", tensor_folder: str=None):
    # create edge list
    edges = create_graph(Lx=Lx, Ly=Ly, shape=shape)
    edges = [[str(edge[0])]+[str(edge[-1])]+["0"]+["0"]+["12"] for edge in edges] + [[str(edge[1])]+[str(edge[-1])]+["1"]+["0"]+["12"] for edge in edges]

    with open(f"{tensor_folder}/basic_{Lx}_{Ly}_{shape}.csv", "w") as f:
        f.write("node1,node2,entanglement,error,bond\n")
        for e in edges:
            string = ",".join(e)
            f.write(string + "\n")

def create_ids_coeffs_file(Lx: int,
                           Ly: int, 
                           shape: str="parallelogram", 
                           g: float=1.0, 
                           bound_state: str=None,
                           R: int=1, 
                           chargesx: list=None, 
                           chargesy: list=None, 
                           filename: str=""):
    charges = make_charges_array(Lx, Ly, shape)
    if bound_state is not None:
        vals = [omega,omega**2] if bound_state == "meson" else [omega,omega,omega]
        if chargesx is None:
            chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state=bound_state, shape=shape, R=R)

        for i,j,val in zip(chargesx, chargesy, vals):
            r, c = site_to_array_index(i, j, Lx, Ly, shape=shape)
            charges[r, c] = val

    print(charges)

    links, _ = evaluate_link_coefficients(Lx=Lx, Ly=Ly, charges=charges, shape=shape)
    links = [(vals[0],vals[1],g*vals[2]) for key, vals in links.items()]
    arr = np.array([str(val) for vals in links for val in vals])
    edges = np.array_split(arr, len(arr)//3)
    with open(filename, "w") as f:
        for e in edges:
            string = ",".join(e)
            f.write(string + "\n")

# Lx = 3
# Ly = 5
# g = -2.763265306122449
# g = -1
# shape = "parallelogram"
# shape = "hexagon"
# chargesx = [0,1]
# chargesy = [0,0]
# chargesx = None
# chargesy = None
# save_edges_file(Lx, Ly, shape, tensor_folder="Z3_data/tensors")
# save_basic_file(Lx, Ly, shape, tensor_folder="Z3_data/tensors")
# create_ids_coeffs_file(Lx, Ly, shape, g, chargesx, chargesy, filename="Z3_data")