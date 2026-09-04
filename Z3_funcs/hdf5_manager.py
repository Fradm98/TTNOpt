"""
hdf5_manager.py
===============
I/O layer for TTN ground state search data.

Two HDF5 files are managed:
  - observables.h5 : energies, entanglement, errors, ef_z, graph
  - tensors.h5     : TTN tensor arrays

Path helpers
------------
All internal HDF5 paths are built by `_obs_base_path` and `_ten_base_path`,
which encode the hierarchy:

  vacuum  : {shape}/Lx{Lx}_Ly{Ly}/vacuum/
  nq      : {shape}/Lx{Lx}_Ly{Ly}/{n}q/{config}/

where {config} = "x{x0}-{x1}_y{y0}-{y1}" for charged sectors.
"""

import os
import random
import time
import numpy as np
import pandas as pd
import h5py
from Z3_funcs.lattice_plaquettes import get_coord_charges

# ---------------------------------------------------------------------------
# Concurrent-access helper
# ---------------------------------------------------------------------------

def _open_h5(path, mode, max_retries=8, base_delay=0.5, max_delay=8.0):
    """
    Open an HDF5 file, retrying with exponential backoff + jitter on lock
    contention. Multiple gss jobs (e.g. separate parallel R/g/Lx sweeps) can
    share the same observables.h5/tensors.hdf5 via output.save_tensors, so
    two processes briefly overlapping on a write is expected, not an error --
    h5py raises BlockingIOError (errno 11, "unable to lock file") in that
    case. Only BlockingIOError is retried; anything else (missing file,
    permissions, corruption) propagates immediately.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return h5py.File(path, mode)
        except BlockingIOError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay + random.uniform(0, delay))
            delay = min(delay * 2, max_delay)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _config_key(chargesx, chargesy):
    """
    Build the charge-configuration group name.

    Parameters
    ----------
    chargesx : list[int] or None
    chargesy : list[int] or None

    Returns
    -------
    sector : str   e.g. "vacuum", "2q", "3q"
    config : str or None
        None for vacuum (config level is skipped),
        e.g. "x2-10_y1-1" for 2q sector.
    """
    if chargesx is None:
        return "vacuum", None
    n = len(chargesx)
    sector = f"{n}-q"
    xs = "-".join(str(x) for x in chargesx)
    ys = "-".join(str(y) for y in chargesy)
    config = f"x{xs}_y{ys}"
    return sector, config


def _obs_base_path(shape, Lx, Ly, chargesx, chargesy):
    """
    Base path inside observables.h5 down to the config level.

    vacuum : {shape}/Lx{Lx}_Ly{Ly}/vacuum
    nq     : {shape}/Lx{Lx}_Ly{Ly}/{n}q/{config}
    """
    sector, config = _config_key(chargesx, chargesy)
    size = f"_Lx{Lx}_Ly{Ly}"
    if config is None:
        return f"shape_{shape}/{size}/runs_{sector}"
    return f"{shape}/{size}/{sector}/{config}"


def _ten_base_path(shape, Lx, Ly, chargesx, chargesy):
    """Same logic for tensors.h5."""
    return _obs_base_path(shape, Lx, Ly, chargesx, chargesy)


def _g_key(g, precision=3):
    """Consistent group name for a coupling value."""
    return f"g_{g:.{precision}f}"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def save_graph(obs_file, shape, Lx, Ly, edges):
    """
    Save the TTN graph (edge list) once per (shape, Lx, Ly).
    Skipped if already present.

    Parameters
    ----------
    obs_file : str  path to observables.h5
    shape    : str
    Lx, Ly   : int
    edges    : list[list[int]]  gss.psi.edges
    """
    path = f"{shape}/Lx{Lx}_Ly{Ly}/graph/edges"
    with _open_h5(obs_file, "a") as f:
        if path not in f:
            arr = np.array(edges, dtype=np.int32)
            f.create_dataset(path, data=arr)


# ---------------------------------------------------------------------------
# EF_Z
# ---------------------------------------------------------------------------

def save_ef_z(obs_file, shape, Lx, Ly, chargesx, chargesy, g, ef_z_path):
    """
    Save the EF_Z file content for a given (config, g).
    Stored as a 2D float array (real parts of the complex coefficients).
    Skipped if already present.

    Parameters
    ----------
    obs_file   : str   path to observables.h5
    ef_z_path  : str   path to the EF_Z.dat file on disk
    """
    base = _obs_base_path(shape, Lx, Ly, chargesx, chargesy)
    hdf_path = f"{base}/ef_z/{_g_key(g)}"
    with _open_h5(obs_file, "a") as f:
        if hdf_path not in f:
            # EF_Z rows: site_i, site_j, complex_coefficient
            data = []
            with open(ef_z_path, "r") as fz:
                for line in fz:
                    parts = line.strip().split(",")
                    if len(parts) == 3:
                        data.append([complex(p) for p in parts])
            arr = np.array(data)   # shape (n_links, 3), complex
            # store real and imag separately for portability
            grp = f.require_group(f"{base}/ef_z")
            ds = grp.create_dataset(_g_key(g), data=arr.view(np.float64).reshape(*arr.shape, 2))
            ds.attrs["columns"] = ["site_i", "site_j", "coeff_re_im"]


# ---------------------------------------------------------------------------
# Observables  (basic.csv)
# ---------------------------------------------------------------------------

def save_basic(obs_file, shape, Lx, Ly, chargesx, chargesy, g, chi, basic_csv_path):
    """
    Save per-edge data from basic.csv for a given (config, g, chi).

    Datasets written under  {base}/chi{chi}/g{g:.6f}/:
        node1        int32  (n_edges,)
        node2        int32  (n_edges,)
        energy       float64 (n_edges,)
        entanglement float64 (n_edges,)
        error        float64 (n_edges,)

    Parameters
    ----------
    basic_csv_path : str  path to the basic.csv produced by gss
    """
    base = _obs_base_path(shape, Lx, Ly, chargesx, chargesy)
    g_key = _g_key(g)
    chi_path = f"{base}/chi{chi}"
    grp_path = f"{chi_path}/{g_key}"

    df = pd.read_csv(basic_csv_path)

    with _open_h5(obs_file, "a") as f:
        if grp_path in f:
            del f[grp_path]          # overwrite if rerunning same point
        grp = f.require_group(grp_path)
        grp.create_dataset("node1",        data=df["node1"].values.astype(np.int32))
        grp.create_dataset("node2",        data=df["node2"].values.astype(np.int32))
        grp.create_dataset("energy",       data=df["energy"].values.astype(np.float64))
        grp.create_dataset("entanglement", data=df["entanglement"].values.astype(np.float64))
        grp.create_dataset("error",        data=df["error"].values.astype(np.float64))
        grp.attrs["g"] = float(g)
        grp.attrs["chi"] = int(chi)


def append_converged_energy(obs_file, shape, Lx, Ly, chargesx, chargesy, g, chi, energy):
    """
    Append the converged (last-sweep) scalar energy and the g value
    to the 1-D summary datasets  {base}/chi{chi}/g  and  {base}/chi{chi}/energy.

    Datasets are created on first call and extended on subsequent ones,
    so values accumulate as the g-loop progresses.

    Parameters
    ----------
    energy : float   converged energy (last row of basic.csv)
    """
    base = _obs_base_path(shape, Lx, Ly, chargesx, chargesy)
    chi_path = f"{base}/chi{chi}"

    with _open_h5(obs_file, "a") as f:
        grp = f.require_group(chi_path)

        for name, value in [("g", float(g)), ("energy", float(energy))]:
            if name not in grp:
                # create resizable 1-D dataset
                grp.create_dataset(
                    name,
                    data=np.array([value]),
                    maxshape=(None,),
                    chunks=(64,),
                )
            else:
                ds = grp[name]
                ds.resize(ds.shape[0] + 1, axis=0)
                ds[-1] = value


# ---------------------------------------------------------------------------
# Tensors
# ---------------------------------------------------------------------------

def save_tensor(ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi, psi, chargesx=None, chargesy=None):
    """
    Save all TTN tensors from gss.psi for a given (config, g, chi).

    Each tensor is stored as a separate dataset:
        {base}/g_{g:.{precision}f}/run_chi-{chi}/tensor_{id}   shape as in psi.tensors[id]

    The edge list is stored alongside:
        {base}/g_{g:.{precision}f}/run_chi-{chi}/edges          int32 (n_tensors, 3)

    Group attributes: g (float), chi (int).

    Parameters
    ----------
    psi : TreeTensorNetwork   gss.psi after the sweep block
    """
    if chargesx is None:
        chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state, shape, R)
    base = _ten_base_path(shape, Lx, Ly, chargesx, chargesy)
    g_key = _g_key(g, precision)
    chi_str = f"{chi}"
    grp_path = f"{base}/{g_key}/run_chi-{chi_str.rjust(4, '0')}"

    with _open_h5(ten_file, "a") as f:
        if grp_path in f:
            del f[grp_path]          # overwrite if rerunning same point
        grp = f.require_group(grp_path)
        grp.attrs["g"] = float(g)
        grp.attrs["chi"] = int(chi)

        # edges array
        grp.create_dataset("edges", data=np.array(psi.edges, dtype=np.int32))

        # individual tensors
        for tensor_id, tensor in enumerate(psi.tensors):
            if tensor is not None and tensor.size > 0:
                grp.create_dataset(
                    f"tensor_{tensor_id}",
                    data=tensor.astype(np.complex128),
                    compression="gzip",
                    compression_opts=4,
                )
        
        # store top_edge_id and canonical_center_edge_id for full reconstruction
        grp.attrs["top_edge_id"] = int(psi.top_edge_id)
        grp.attrs["canonical_center_edge_id"] = int(psi.canonical_center_edge_id)
        if psi.norm is not None:
            grp.attrs["norm"] = float(psi.norm)

        # gauge tensor at the canonical center — required to resume a sweep
        # (GroundStateSearch.run() dereferences it on the very first step)
        if psi.gauge_tensor is not None:
            grp.create_dataset(
                "gauge_tensor",
                data=np.asarray(psi.gauge_tensor).astype(np.complex128),
                compression="gzip",
                compression_opts=4,
            )


# ---------------------------------------------------------------------------
# Reader helpers
# ---------------------------------------------------------------------------

def load_energy_vs_g(obs_file, shape, Lx, Ly, chargesx, chargesy, chi):
    """
    Load the converged energy summary for a given (config, chi).

    Returns
    -------
    g      : np.ndarray  (n_points,)
    energy : np.ndarray  (n_points,)
    """
    base = _obs_base_path(shape, Lx, Ly, chargesx, chargesy)
    chi_path = f"{base}/chi{chi}"
    with _open_h5(obs_file, "r") as f:
        g      = f[f"{chi_path}/g"][:]
        energy = f[f"{chi_path}/energy"][:]
    return g, energy


def load_basic(obs_file, shape, Lx, Ly, chargesx, chargesy, g, chi):
    """
    Load per-edge data for a specific (config, g, chi).

    Returns
    -------
    pd.DataFrame with columns: node1, node2, energy, entanglement, error
    """
    base = _obs_base_path(shape, Lx, Ly, chargesx, chargesy)
    grp_path = f"{base}/chi{chi}/{_g_key(g)}"
    with _open_h5(obs_file, "r") as f:
        grp = f[grp_path]
        df = pd.DataFrame({
            "node1":        grp["node1"][:],
            "node2":        grp["node2"][:],
            "energy":       grp["energy"][:],
            "entanglement": grp["entanglement"][:],
            "error":        grp["error"][:],
        })
    return df


def tensor_exists(ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi, chargesx=None, chargesy=None):
    """
    Check whether a saved TTN checkpoint exists for (config, g, chi), without
    raising if the file or group is missing. Mirrors load_tensor's path
    construction so the two never drift apart. Used for warm-start fallback.
    """
    if not os.path.exists(ten_file):
        return False
    if chargesx is None:
        chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state, shape, R)
    base = _ten_base_path(shape, Lx, Ly, chargesx, chargesy)
    g_key = _g_key(g, precision)
    chi_str = f"{chi}"
    grp_path = f"{base}/{g_key}/run_chi-{chi_str.rjust(4, '0')}"
    with _open_h5(ten_file, "r") as f:
        return grp_path in f


def load_tensor(ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi, chargesx=None, chargesy=None):
    """
    Load TTN tensors and reconstruct a TreeTensorNetwork object.

    Returns
    -------
    edges   : list[list[int]]
    tensors : list[np.ndarray]
    attrs   : dict  (top_edge_id, canonical_center_edge_id, norm, g, chi)
    """
    from ttnopt.src.TTN import TreeTensorNetwork

    if chargesx is None:
        chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state, shape, R)
    base = _ten_base_path(shape, Lx, Ly, chargesx, chargesy)
    g_key = _g_key(g, precision)
    chi_str = f"{chi}"
    grp_path = f"{base}/{g_key}/run_chi-{chi_str.rjust(4, '0')}"

    with _open_h5(ten_file, "r") as f:
        grp = f[grp_path]
        edges = grp["edges"][:].tolist()
        n_tensors = len(edges)
        tensors = [grp[f"tensor_{i}"][:] for i in range(n_tensors)]
        attrs = dict(grp.attrs)
        gauge_tensor = grp["gauge_tensor"][:] if "gauge_tensor" in grp else None

    psi = TreeTensorNetwork(
        edges=edges,
        tensors=tensors,
        top_edge_id=int(attrs["top_edge_id"]),
    )
    psi.canonical_center_edge_id = int(attrs["canonical_center_edge_id"])
    if "norm" in attrs:
        psi.norm = float(attrs["norm"])
    if gauge_tensor is not None:
        psi.gauge_tensor = gauge_tensor

    return psi, attrs


# ---------------------------------------------------------------------------
# Link electric-field cache (Z3_funcs.observables.link_electric_fields,
# whole-lattice mode)
# ---------------------------------------------------------------------------

def _ef_links_grp_path(shape, Lx, Ly, bound_state, R, g, precision, chi, chargesx, chargesy):
    if chargesx is None:
        chargesx, chargesy = get_coord_charges(Lx, Ly, bound_state, shape, R)
    base = _ten_base_path(shape, Lx, Ly, chargesx, chargesy)
    g_key = _g_key(g, precision)
    chi_str = f"{chi}"
    return f"{base}/{g_key}/run_chi-{chi_str.rjust(4, '0')}"


def save_link_electric_fields(ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi, links, chargesx=None, chargesy=None):
    """
    Cache computed link electric-field values -- the whole-lattice result
    of Z3_funcs.observables.link_electric_fields(engine, plaquette_label=
    None, ...) -- nested inside the SAME tensors.hdf5 group as the tensor
    checkpoint they were computed from (under "ef_links"). This ties the
    cache's lifetime to that specific checkpoint: save_tensor already
    deletes and recreates its whole group on overwrite, so a stale
    checkpoint can never leave a stale cache behind.

    links : dict (i, j, d) -> {"p_left", "p_right", "value", "kind"}
        as returned by link_electric_fields(engine, None, Lx, Ly, shape).
    """
    grp_path = _ef_links_grp_path(shape, Lx, Ly, bound_state, R, g, precision, chi, chargesx, chargesy)

    keys = sorted(links.keys())
    ijd = np.array(keys, dtype=np.int32)
    p_left = np.array(
        [links[k]["p_left"] if links[k]["p_left"] is not None else -1 for k in keys],
        dtype=np.int32,
    )
    p_right = np.array(
        [links[k]["p_right"] if links[k]["p_right"] is not None else -1 for k in keys],
        dtype=np.int32,
    )
    values = np.array([links[k]["value"] for k in keys], dtype=np.complex128)
    kinds = [links[k]["kind"] for k in keys]

    with _open_h5(ten_file, "a") as f:
        if grp_path not in f:
            raise KeyError(
                f"No tensor checkpoint at {grp_path} in {ten_file} -- "
                f"save_tensor must be called before caching link electric fields."
            )
        grp = f[grp_path]
        if "ef_links" in grp:
            del grp["ef_links"]
        sub = grp.create_group("ef_links")
        sub.create_dataset("ijd", data=ijd)
        sub.create_dataset("p_left", data=p_left)
        sub.create_dataset("p_right", data=p_right)
        sub.create_dataset("value", data=values.view(np.float64).reshape(-1, 2))
        sub.create_dataset("kind", data=kinds, dtype=h5py.string_dtype())


def link_electric_fields_exist(ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi, chargesx=None, chargesy=None):
    """Check whether cached link electric-field values exist for this
    checkpoint, without raising if the file/group is missing. Mirrors
    save_link_electric_fields' path construction so the two never drift
    apart."""
    if not os.path.exists(ten_file):
        return False
    grp_path = _ef_links_grp_path(shape, Lx, Ly, bound_state, R, g, precision, chi, chargesx, chargesy)
    with _open_h5(ten_file, "r") as f:
        return f"{grp_path}/ef_links" in f


def load_link_electric_fields(ten_file, shape, Lx, Ly, bound_state, R, g, precision, chi, chargesx=None, chargesy=None):
    """Load cached link electric-field values, in the same format
    link_electric_fields(engine, None, ...) returns ("neighbor" is always
    None here, since that field only carries meaning relative to a
    specific plaquette_label, which the whole-lattice cache doesn't
    have)."""
    grp_path = _ef_links_grp_path(shape, Lx, Ly, bound_state, R, g, precision, chi, chargesx, chargesy)

    with _open_h5(ten_file, "r") as f:
        sub = f[f"{grp_path}/ef_links"]
        ijd = sub["ijd"][:]
        p_left = sub["p_left"][:]
        p_right = sub["p_right"][:]
        values = sub["value"][:].view(np.complex128).reshape(-1)
        kinds = [k.decode() if isinstance(k, bytes) else k for k in sub["kind"][:]]

    links = {}
    for row, pl, pr, val, kind in zip(ijd, p_left, p_right, values, kinds):
        key = (int(row[0]), int(row[1]), int(row[2]))
        links[key] = {
            "p_left": None if pl == -1 else int(pl),
            "p_right": None if pr == -1 else int(pr),
            "neighbor": None,
            "value": complex(val),
            "kind": kind,
        }
    return links


def load_ef_z(obs_file, shape, Lx, Ly, chargesx, chargesy, g):
    """
    Load EF_Z data for a given (config, g).

    Returns
    -------
    np.ndarray of shape (n_links, 3), complex128
        columns: site_i, site_j, coefficient
    """
    base = _obs_base_path(shape, Lx, Ly, chargesx, chargesy)
    ds_path = f"{base}/ef_z/{_g_key(g)}"
    with _open_h5(obs_file, "r") as f:
        raw = f[ds_path][:]          # shape (n_links, 3, 2)
    return raw.view(np.complex128).reshape(raw.shape[0], raw.shape[1])


def list_available(obs_file):
    """
    Print a summary of all (shape, size, sector, config, chi) combinations
    present in observables.h5, with the number of g points stored.
    """
    def _walk(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/energy"):
            parts = name.split("/")
            print(f"  {'/'.join(parts[:-1]):<60}  n_g = {obj.shape[0]}")
    with _open_h5(obs_file, "r") as f:
        f.visititems(_walk)
