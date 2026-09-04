import os, tempfile, time

import numpy as np
from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine
from ttnopt.hamiltonian import hamiltonian
from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.create_ttn import get_rnd_tree
from Z3_funcs.hdf5_manager import save_tensor, load_tensor, tensor_exists

# Run from the repo root, e.g.:
#   OMP_NUM_THREADS=<n> MKL_NUM_THREADS=<n> OPENBLAS_NUM_THREADS=<n> python chi_scan_diagnostic.py
# Pick <n> to match your cluster allocation (cores per job), same convention as sweep.py.

Lx, Ly, shape_ = 5, 5, "parallelogram"
g = -1.0
chis = [10, 50, 100, 150]
N_SWEEPS = 5
PRECISION = 3

# gss.psi.tensors are checkpointed here (via the same save_tensor/load_tensor
# mechanism sweep.py already uses for g-continuation warm-starts) right after
# every completed chi stage -- so a job killed mid-run (walltime cutoff,
# preemption, node issue) can resume from the last finished chi instead of
# redoing the whole ladder from chi=10. Adjust the path if your working
# directory layout differs -- this assumes the script runs with cwd at the
# repo root (as chi_scan_diagnostic.pbs's `cd $PBS_O_WORKDIR` does) and a
# sibling "5_Z3/logs" directory that already exists.
CHECKPOINT_FILE = "../5_Z3/logs/chi_scan_checkpoint.hdf5"
os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

N = nplaqs(Lx, Ly, shape_)
tmpdir = tempfile.mkdtemp()
tensor_folder = os.path.join(tmpdir, "tensors")
os.makedirs(tensor_folder, exist_ok=True)
save_edges_file(Lx, Ly, shape_, tensor_folder=tensor_folder)
filename = os.path.join(tmpdir, "EF_Z.dat")
create_ids_coeffs_file(Lx, Ly, shape_, g, None, filename=filename)  # vacuum
mf = 1 / g

system_cfg = DotMap({
    "N": N, "spin_size": "1", "Lx": Lx, "Ly": Ly, "shape": shape_,
    "model": {"type": "z3", "file": filename},
    "MF_X": mf, "EF_Z": filename,
})

print(f"Chained warm-start chi-scan: Lx={Lx}, Ly={Ly}, vacuum, g={g}, chis={chis}, "
      f"{N_SWEEPS} sweeps each (chi[k] warm-started from chi[k-1]'s converged ansatz)", flush=True)

# Resume from the most advanced checkpointed chi stage, if any (found by
# scanning from the largest chi down, so a partially-completed run always
# picks up the furthest point actually reached).
start_idx = 0
resume_chi = None
for i in range(len(chis) - 1, -1, -1):
    if tensor_exists(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, chis[i]):
        resume_chi = chis[i]
        start_idx = i + 1
        break

ham = hamiltonian(system_cfg)
if resume_chi is not None:
    psi, _ = load_tensor(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, resume_chi)
    print(f"Found checkpoint at chi={resume_chi} -- resuming from chis[{start_idx}:]={chis[start_idx:]}",
          flush=True)
    gss = GroundStateSearch(psi, ham, init_bond_dim=resume_chi, max_bond_dim=resume_chi)
else:
    psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=shape_, path=tensor_folder, chi=chis[0])
    gss = GroundStateSearch(psi, ham, init_bond_dim=chis[0], max_bond_dim=chis[0])
patch_physics_engine(gss)

ref_edge = gss.psi.top_edge_id
print(f"top_edge_id (reference edge) = {ref_edge}", flush=True)

stage_results = {}

for chi in chis[start_idx:]:
    gss.max_bond_dim = chi
    # Re-center on ref_edge before each stage so the sweep tour reliably
    # revisits it (cheap, lossless -- max_bond_dim set above already
    # bounds the true rank at every cut, since chi only increases).
    gss.move_canonical_center(ref_edge)
    gss._prime_renormalized_operators()
    t0 = time.time()
    gss.run(
        opt_structure=0,
        max_num_sweep=N_SWEEPS,
        verbose=True,
        diagnostic_edges=[ref_edge],
        energy_convergence_threshold=0.0,
        entanglement_convergence_threshold=0.0,
    )
    elapsed = time.time() - t0
    print(f"\nchi={chi} stage done in {elapsed:.0f}s total\n", flush=True)

    # Checkpoint immediately, before the next (more expensive) chi stage
    # runs -- if that next stage dies partway through, this one's hours of
    # compute are already safe on disk.
    save_tensor(CHECKPOINT_FILE, shape_, Lx, Ly, None, None, g, PRECISION, chi, gss.psi)
    print(f"checkpoint saved: chi={chi} -> {CHECKPOINT_FILE}", flush=True)

    # run() resets self.local_update_diagnostics/self.convergence_history
    # to [] at the START of every call (they do NOT accumulate across
    # calls) -- so each stage's results must be captured right here,
    # before the next stage's run() call wipes them.
    stage_results[chi] = {
        "local_update_diagnostics": list(gss.local_update_diagnostics),
        "convergence_history": list(gss.convergence_history),
    }

    # Print (and flush) this stage's results IMMEDIATELY, not after the
    # whole chis loop finishes -- a later stage (e.g. a bigger chi running
    # out of memory) must not be able to wipe out an already-completed
    # stage's results just because the process dies before the final
    # summary block would have printed them.
    print(f"{'='*110}\n--- chi={chi} reference-edge trace and local-update diagnostics ---\n{'='*110}")
    print(f"{'sweep':>5} {'ref_energy':>18} {'ref_energy_reldiff':>19} {'ref_trunc_err':>15} "
          f"{'ref_lanczos_resid':>18} {'ref_retried':>11} {'sweep_retries':>13}")
    for rec in stage_results[chi]["convergence_history"]:
        print(f"{rec['sweep']:>5} {rec['ref_energy']:>18.10f} {rec['ref_energy_reldiff']:>19.6e} "
              f"{rec['ref_truncation_error']:>15.6e} {rec['ref_lanczos_residual']:>18.6e} "
              f"{str(rec['ref_lanczos_retried']):>11} {rec['num_lanczos_retries']:>13}")

    lu_recs = stage_results[chi]["local_update_diagnostics"]
    if lu_recs:
        gains = [rec["E_before"] - rec["E_lanczos"] for rec in lu_recs]
        losses = [rec["E_after_truncation"] - rec["E_lanczos"] for rec in lu_recs]
        print(f"local_update_diagnostics (ref_edge={ref_edge}): {len(lu_recs)} records")
        print(f"  lanczos_gain: mean={np.mean(gains):.6e}, last={gains[-1]:.6e}")
        print(f"  trunc_loss:   mean={np.mean(losses):.6e}, last={losses[-1]:.6e}")
        print(f"  trunc_loss/lanczos_gain ratio: mean={np.mean(losses)/np.mean(gains):.4f}, "
              f"last={losses[-1]/gains[-1]:.4f}")
    else:
        print(f"local_update_diagnostics (ref_edge={ref_edge}): 0 records -- ref_edge was not "
              f"revisited this stage (unexpected; investigate before trusting this stage's data)")

    # cross-chi summary of everything completed SO FAR, reprinted after
    # every stage so the log always has the latest full picture even if
    # a later, bigger chi stage crashes the process
    print(f"\n--- cross-chi summary so far (completed: {sorted(stage_results.keys())}) ---")
    print(f"{'chi':>6} {'mean_lanczos_gain':>20} {'mean_trunc_loss':>18} {'ratio':>10}")
    for c in chis:
        if c not in stage_results:
            break
        recs = stage_results[c]["local_update_diagnostics"]
        if not recs:
            print(f"{c:>6} {'(no records)':>20}")
            continue
        g_ = [r["E_before"] - r["E_lanczos"] for r in recs]
        l_ = [r["E_after_truncation"] - r["E_lanczos"] for r in recs]
        print(f"{c:>6} {np.mean(g_):>20.6e} {np.mean(l_):>18.6e} {np.mean(l_)/np.mean(g_):>10.4f}")
    print("", flush=True)

print("\nDONE")
