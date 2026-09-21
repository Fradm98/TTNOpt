"""
z3_5x5_engine_diagnostics.py
--------------------------------
Consolidated tool for the 5x5, vacuum, g=-1.0 investigation into why
chi=20's superblock energy trajectory sometimes plateaus at a suspicious
flat E~-60 (superblock_energy_trajectory.py) and why patch_physics_engine
diverges from the state's true <psi|H|psi>. Replaces what had sprawled
into six separate one-off scripts (independent_energy_trajectory.py,
chi20_independent_energy_trajectory.py, chi20_trap_warmstart.py,
ab_test_linop_vs_dense_chi20.py, superblock_energy_trajectory_optstruct.py,
compare_superblock_trajectories_optstruct.py, plot_chi20_patched_vs_unpatched.py,
plot_chi20_energies_patched_vs_unpatched.py) -- one file, four subcommands.

Findings so far (see each subcommand's docstring for how they were reached):
  - patch_physics_engine() (TTNLinearOperator.py) makes the sweep's own
    tracked ref_energy diverge from the state's true, independently-
    computed <psi|H|psi> by ~1e-3..1e-2 after the first ~30 updates --
    reproducible at both chi=20 and chi=50 (bug size is chi-independent),
    when patched is run as its OWN independent trajectory.
  - The original (unpatched) dense lanczos() does NOT show this: tracked
    and independent energy agree to ~1e-14..1e-7 (float noise) throughout.
  - `abtest`'s SHADOW methodology (same edge, same frozen state, dense's
    result is the only one ever written back -- see that subcommand's
    docstring for why the original independent-trajectories version was
    flawed) shows dense and linop agreeing to ~1e-12..1e-13 at EVERY
    update across several sweeps. So the local matvec/eigensolver is not
    the bug -- confirmed twice now, once via direct _apply_ham_psi vs
    _apply_ham_psi_matvec comparison and once via this shadow sweep. The
    ~1e-2 divergence must come from something that only appears when
    linop's own (still locally-correct) choices are allowed to
    accumulate across an independently-evolving trajectory -- exactly
    the kind of sensitivity a near-degenerate spectrum produces.
  - For the SAME random seed, unpatched can itself get stuck at the
    trivial E=-60 cold-start state for its entire run while patched
    escapes it -- neither path is simply "the reliable one". `seedscan`
    exists to find out how often unpatched does this across seeds.
  - opt_structure=1/2 (tree restructuring) show the same oscillation
    band as opt_structure=0 at chi=20/50 -- no visible improvement.

Subcommands
-----------
  trajectory   Run one sweep-recording session at a given chi, optionally
               patched, optionally opt_structure-restructured, optionally
               with a full independent_energy() cross-check after every
               single update (expensive; ~0.5-2s/update depending on chi).
               Writes one CSV under $DRIVE_PATH/results/energy_data.

  abtest       ONE real (dense) trajectory; at every update, before the
               result is written back to psi.tensors, also solves the
               exact same frozen two-site problem via ttn_eigensolver()
               as a "shadow" that never feeds back into the state. This
               is a true same-edge, same-state comparison at every
               single update with zero compounding -- unlike letting
               dense and linop run as two independent trajectories
               (which only agrees at update 0, then the tiniest
               eigenvector difference makes the two states diverge, so
               "edge X" from each side stops being the same state).
               Prints a report, writes nothing.

  warmstart    Run chi=CHI_LOW to convergence/plateau, checkpoint that
               exact state, then continue -- independently, from that same
               untouched checkpoint -- at each of CHI_TARGETS, to see
               whether a cold-start trap survives being handed a bigger
               bond dimension. Writes one CSV+PNG per target chi under
               diagnostics/z3_5x5_warmstart_results/.

  seedscan     Run N independent random-seed trajectories at one chi
               (unpatched by default, since that's the path whose -60
               cold-start trap needs statistics), record each seed's
               final energy, and report the distribution (mean, std, how
               many seeds land near -60 vs escape). Writes one CSV +
               one summary PNG.

  chiladder    Proper (not the abbreviated 2-cheap-sweep) chi ladder,
               unpatched by default -- since the production `gss` CLI
               entry point patches unconditionally, the "converged
               chi=50" reference used earlier in this investigation is
               NOT independent ground truth, so this does an honest
               convergence study entirely within the unpatched path:
               energy AND entanglement entropy at every chi, to see
               directly whether the state is converging (entropy
               plateauing) or still growing. Can optionally continue the
               warm-started top-of-ladder state to a higher target chi,
               timed, to test whether a good warm start lets dense
               Lanczos finish in practical time. Writes one CSV + two
               PNGs (trajectory and chi-vs-final-value summary).

  compare      Build a comparison plot from CSVs already written by
               `trajectory` runs -- either a chi-ladder overlay for one
               opt_structure (kind=opt_structure) or a patched-vs-
               unpatched overlay, diff and/or raw energies, for one chi
               (kind=patched_vs_unpatched).

Run from the repo root. Examples:

  # chi=20, patched, 5 sweeps, with the independent-energy cross-check
  python diagnostics/z3_5x5_engine_diagnostics.py trajectory --chi 20 --patch --sweeps 5 --independent

  # same, unpatched (dense) -- for the patched-vs-unpatched comparison
  python diagnostics/z3_5x5_engine_diagnostics.py trajectory --chi 20 --sweeps 5 --independent

  # then build both comparison plots from those two CSVs
  python diagnostics/z3_5x5_engine_diagnostics.py compare --kind patched_vs_unpatched --chi 20

  # opt_structure=1 chi ladder + its comparison plot
  for chi in 20 50 100; do
    python diagnostics/z3_5x5_engine_diagnostics.py trajectory --chi $chi --patch --opt-structure 1 --sweeps 5
  done
  python diagnostics/z3_5x5_engine_diagnostics.py compare --kind opt_structure --opt-structure 1

  # matched dense-vs-patched A/B, chi=20
  python diagnostics/z3_5x5_engine_diagnostics.py abtest --chi 20

  # chi=20 cold-start trap continued into chi=50 and chi=100
  python diagnostics/z3_5x5_engine_diagnostics.py warmstart --chi-low 20 --chi-targets 50 100

  # 15-seed statistics scan, chi=20 unpatched
  python diagnostics/z3_5x5_engine_diagnostics.py seedscan --chi 20 --sweeps 10 --n-seeds 15

  # proper chi=9->18->27 ladder, unpatched, energy+entropy at each stage,
  # then a warm-started, timed attempt at chi=50
  python diagnostics/z3_5x5_engine_diagnostics.py chiladder --chis 9 18 27 --sweeps 3 3 15 --continue-to 50
"""
import argparse
import copy
import csv
import os
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import numpy as np
import tensornetwork as tn
from dotmap import DotMap

from ttnopt.src import GroundStateSearch
from ttnopt.src.TTNLinearOperator import patch_physics_engine, ttn_eigensolver
from ttnopt.hamiltonian import hamiltonian as build_hamiltonian
from Z3_funcs.create_graph_file import save_edges_file, create_ids_coeffs_file, nplaqs
from Z3_funcs.create_ttn import get_rnd_tree
from Z3_funcs.hdf5_manager import save_tensor, load_tensor
from Z3_funcs.independent_energy import independent_energy

Lx, Ly, SHAPE = 5, 5, "parallelogram"
G = -1.0
PRECISION = 3
DRIVE_PATH = "/Users/fradm/Desktop/projects/5_Z3"
DATA_DIR = f"{DRIVE_PATH}/results/energy_data"
FIG_DIR = f"{DRIVE_PATH}/figures"
WARMSTART_DIR = os.path.join(os.path.dirname(__file__), "z3_5x5_warmstart_results")


# ── shared setup ─────────────────────────────────────────────────────────

def make_hamiltonian(g=G, lx=None, ly=None):
    lx = Lx if lx is None else lx
    ly = Ly if ly is None else ly
    N = nplaqs(lx, ly, SHAPE)
    tmpdir = tempfile.mkdtemp()
    tensor_folder = os.path.join(tmpdir, "tensors")
    os.makedirs(tensor_folder, exist_ok=True)
    save_edges_file(lx, ly, SHAPE, tensor_folder=tensor_folder)
    filename = os.path.join(tmpdir, "EF_Z.dat")
    create_ids_coeffs_file(lx, ly, SHAPE, g, None, filename=filename)  # vacuum
    system_cfg = DotMap({
        "N": N, "spin_size": "1", "Lx": lx, "Ly": ly, "shape": SHAPE,
        "model": {"type": "z3", "file": filename},
        "MF_X": 1.0 / g, "EF_Z": filename,
    })
    return build_hamiltonian(system_cfg), tensor_folder


def warmup_ladder(target_chi, warmup_chis=(10, 30, 50)):
    return [c for c in warmup_chis if c < target_chi] + [target_chi]


def run_warmup(gss, ref_edge, chi_ladder, opt_structure, patched,
                warmup_sweeps=2, lanczos_tol=1e-6, lanczos_maxiter=150):
    for chi in chi_ladder[1:]:
        gss.max_bond_dim = chi
        gss.move_canonical_center(ref_edge)
        gss._prime_renormalized_operators()
        kwargs = dict(
            opt_structure=opt_structure, max_num_sweep=warmup_sweeps, verbose=True,
            energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0,
        )
        if patched:  # original lanczos() doesn't accept these kwargs at all
            kwargs["lanczos_tol"] = lanczos_tol
            kwargs["lanczos_maxiter"] = lanczos_maxiter
        gss.run(**kwargs)


# ── trajectory ───────────────────────────────────────────────────────────

def cmd_trajectory(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    ham, tensor_folder = make_hamiltonian(lx=args.lx, ly=args.ly)

    patch_label = "patched" if args.patch else "unpatched-dense"
    size_prefix = "" if (args.lx, args.ly) == (Lx, Ly) else f"{args.lx}x{args.ly}_"
    tag = f"{size_prefix}g{G:.3f}_chi{args.chi}_{patch_label}_optstruct{args.opt_structure}"
    if args.lanczos_tol is not None:
        tag += f"_tol{args.lanczos_tol:.0e}"
    if args.independent:
        tag += "_independent_energy"
    print(f"trajectory: {args.lx}x{args.ly}, chi={args.chi}, {patch_label}, "
          f"opt_structure={args.opt_structure}, {args.sweeps} sweeps, "
          f"independent_energy={'on' if args.independent else 'off'}, "
          f"lanczos_tol={'default' if args.lanczos_tol is None else args.lanczos_tol}",
          flush=True)

    chi_ladder = warmup_ladder(args.chi)
    psi = get_rnd_tree(Lx=args.lx, Ly=args.ly, shape=SHAPE, path=tensor_folder, chi=chi_ladder[0])
    gss = GroundStateSearch(psi, ham, init_bond_dim=chi_ladder[0], max_bond_dim=chi_ladder[0])
    if args.patch:
        patch_physics_engine(gss)
    ref_edge = gss.psi.top_edge_id
    run_warmup(gss, ref_edge, chi_ladder, args.opt_structure, args.patch)

    if args.lanczos_tol is not None and not args.patch:
        # GroundStateSearch.run()'s own lanczos_tol=... knob forwards to a
        # kwarg named "tol" (see the module docstring / run()'s
        # lanczos_kwargs construction), which only matches the PATCHED
        # ttn_eigensolver's signature -- the original PhysicsEngine.lanczos()
        # names its tolerance parameter lanczos_tol instead, so run()'s own
        # public knob can never reach it (raises TypeError if you try). The
        # only way to override dense's tolerance is a direct monkey-patch.
        _original_lanczos = gss.lanczos
        _tol = args.lanczos_tol
        gss.lanczos = lambda central_tensor_ids, **kw: _original_lanczos(
            central_tensor_ids, lanczos_tol=_tol, **kw
        )

    records = []
    if args.independent:
        # Hook: entanglement_entropy_at_physical_bond is called EXACTLY
        # once per superblock update inside GroundStateSearch.run(), right
        # after psi.tensors/gauge_tensor are overwritten with that update's
        # result -- entanglement_entropy itself is called several times
        # per update (once directly, again per physical bond inside this
        # very method) so it is NOT a safe once-per-update hook.
        original_ee_at_bond = gss.entanglement_entropy_at_physical_bond
        counter = {"i": 0}

        def patched_ee_at_bond(psi_tensor, psi_edges):
            result = original_ee_at_bond(psi_tensor, psi_edges)
            idx = counter["i"]
            traj_rec = gss.superblock_energy_trajectory[idx]
            indep_e = independent_energy(gss.psi, gss.hamiltonian)
            records.append({
                "update_index": idx, "sweep": traj_rec["sweep"], "edge_id": traj_rec["edge_id"],
                "ref_energy": traj_rec["energy"], "independent_energy": indep_e,
            })
            print(f"    update {idx}: ref_energy={traj_rec['energy']:.8f}  "
                  f"independent_energy={indep_e:.8f}  "
                  f"diff={abs(traj_rec['energy'] - indep_e):.3e}", flush=True)
            counter["i"] += 1
            return result

        gss.entanglement_entropy_at_physical_bond = patched_ee_at_bond

    main_run_kwargs = dict(
        opt_structure=args.opt_structure, max_num_sweep=args.sweeps, verbose=True,
        energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0,
    )
    if args.lanczos_tol is not None and args.patch:
        # Patched ttn_eigensolver DOES accept the "tol" kwarg run() forwards
        # lanczos_tol into -- see the note above the monkey-patch for why
        # this only works for patched, not for the unpatched engine.
        main_run_kwargs["lanczos_tol"] = args.lanczos_tol
    gss.run(**main_run_kwargs)

    csv_path = os.path.join(DATA_DIR, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if args.independent:
            writer.writerow(["update_index", "sweep", "edge_id", "ref_energy", "independent_energy"])
            for r in records:
                writer.writerow([r["update_index"], r["sweep"], r["edge_id"],
                                  r["ref_energy"], r["independent_energy"]])
        else:
            writer.writerow(["update_index", "sweep", "edge_id", "energy"])
            for i, r in enumerate(gss.superblock_energy_trajectory):
                writer.writerow([i, r["sweep"], r["edge_id"], r["energy"]])
    print(f"\nsaved: {csv_path}")
    print("DONE")


# ── abtest ───────────────────────────────────────────────────────────────
#
# Methodology note (fixed after review): letting dense and linop run as two
# INDEPENDENT trajectories from the same start (the original version of
# this test) only gives a clean comparison at update 0. From update 1 on,
# even a tiny difference in which eigenvector each solver picks means the
# two states have already diverged -- so "edge X's energy" from each side
# is being computed on two DIFFERENT underlying tensor states that merely
# share an edge_id, not a true same-state comparison.
#
# Fix: run ONE real sweep with the trusted dense engine (it alone ever
# writes back to psi.tensors), and at every single update, BEFORE that
# write-back happens, also evaluate ttn_eigensolver() on the exact same
# frozen two-site block + cached block_hamiltonians/edge_spin_operators
# dense just used -- a "shadow" call whose result is recorded but never
# fed back into the evolving state. This guarantees every comparison is
# same-edge, same-state, with zero compounding across the sweep.

def cmd_abtest(args):
    print(f"A/B test (shadow method): dense lanczos() vs TTNLinearOperator, chi={args.chi}, "
          f"seed={args.seed}, {args.sweeps} sweep(s), one real (dense) trajectory + a "
          f"same-state linop shadow at every update", flush=True)

    ham, tensor_folder = make_hamiltonian()
    np.random.seed(args.seed)
    psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=SHAPE, path=tensor_folder, chi=args.chi)
    gss = GroundStateSearch(psi, ham, init_bond_dim=args.chi, max_bond_dim=args.chi)
    ref_edge = gss.psi.top_edge_id

    shadow_records = []
    original_lanczos = gss.lanczos  # PhysicsEngine.lanczos, bound method -- never patched here

    def shadow_lanczos(central_tensor_ids, **kw):
        # Authoritative result: this is what actually gets written back to
        # psi.tensors via decompose_two_tensors, further down in run().
        ground_state, energy_dense = original_lanczos(central_tensor_ids, **kw)
        # Shadow: same central_tensor_ids, called before any write-back, so
        # engine.psi.tensors/block_hamiltonians are IDENTICAL to what
        # original_lanczos just saw -- same edge, same state, by construction.
        ground_state_shadow, energy_linop = ttn_eigensolver(gss, central_tensor_ids)
        # ttn_eigensolver sets these as a side effect on gss for its own
        # ArpackNoConvergence diagnostics -- reset so they don't leak into
        # run()'s own dense-path bookkeeping just below.
        gss.last_lanczos_retried = False
        gss.last_lanczos_residual = float("nan")
        # Eigenvector overlap test (not just eigenvalue): both are unit-norm
        # (dense normalizes explicitly; eigsh returns unit-norm vectors), so
        # |<v_dense|v_shadow>| = 1 iff they're the same vector up to a global
        # phase, and < 1 to the extent they're actually different vectors
        # (e.g. a different linear combination within a near-degenerate
        # subspace) -- directly tests whether the eigenVALUE agreement the
        # energy-only shadow test found also means the eigenVECTORS agree.
        v_dense = np.asarray(ground_state.tensor).ravel()
        v_shadow = np.asarray(ground_state_shadow.tensor).ravel()
        overlap = abs(np.vdot(v_dense, v_shadow)) / (np.linalg.norm(v_dense) * np.linalg.norm(v_shadow))
        shadow_records.append({
            "energy_dense": energy_dense, "energy_linop": energy_linop, "overlap": overlap,
        })
        return ground_state, energy_dense

    gss.lanczos = shadow_lanczos
    gss.run(opt_structure=0, max_num_sweep=args.sweeps, verbose=True, diagnostic_edges=[ref_edge],
            energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)

    trajectory = gss.superblock_energy_trajectory  # same order/length as shadow_records
    n = len(trajectory)
    assert len(shadow_records) == n, f"shadow/trajectory length mismatch: {len(shadow_records)} vs {n}"

    print(f"\n{'update':>7} {'edge_id':>8} {'sweep':>6} {'dense (authoritative)':>22} {'linop (shadow)':>18} "
          f"{'|dE|':>12} {'1-|<v_d|v_s>|':>14}")
    max_diff = 0.0
    max_diff_i = -1
    min_overlap = 1.0
    min_overlap_i = -1
    for i in range(n):
        t, s = trajectory[i], shadow_records[i]
        d = abs(t["energy"] - s["energy_linop"])
        if d > max_diff:
            max_diff, max_diff_i = d, i
        if s["overlap"] < min_overlap:
            min_overlap, min_overlap_i = s["overlap"], i
        if i < 10 or i > n - 10 or d == max_diff or s["overlap"] == min_overlap:
            print(f"{i:>7} {t['edge_id']:>8} {t['sweep']:>6} {s['energy_dense']:>22.12f} "
                  f"{s['energy_linop']:>18.12f} {d:>12.3e} {1 - s['overlap']:>14.3e}")

    print(f"\n--- summary over {n} same-edge, same-state comparisons ---")
    print(f"max |dense - linop| energy diff = {max_diff:.6e}  (at update {max_diff_i}, "
          f"edge_id={trajectory[max_diff_i]['edge_id']}, sweep={trajectory[max_diff_i]['sweep']})")
    print(f"max (1 - |<v_dense|v_shadow>|) eigenvector mismatch = {1 - min_overlap:.6e}  "
          f"(at update {min_overlap_i}, edge_id={trajectory[min_overlap_i]['edge_id']}, "
          f"sweep={trajectory[min_overlap_i]['sweep']})")
    if min_overlap > 1 - 1e-6:
        print("-> eigenvectors match to ~1e-6 or better at EVERY update: hypothesis 'different "
              "eigenvector' is NOT supported by this run -- look downstream of the eigensolver instead "
              "(decompose_two_tensors / how the vector is truncated back into tensors).")
    else:
        print(f"-> eigenvectors genuinely differ (overlap dropped to {min_overlap:.6f} at some update): "
              "supports the 'patched picks a different eigenvector' hypothesis.")
    print(f"final energy (ref edge, dense/authoritative trajectory): {gss.energy.get(ref_edge)}")

    # save + plot overlap and energy-diff vs update index together
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = f"abtest_eigenvector_overlap_chi{args.chi}_seed{args.seed}_sweeps{args.sweeps}"
    csv_path = os.path.join(DATA_DIR, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update_index", "sweep", "edge_id", "energy_dense", "energy_linop", "abs_energy_diff", "eigenvector_overlap"])
        for i in range(n):
            t, s = trajectory[i], shadow_records[i]
            writer.writerow([i, t["sweep"], t["edge_id"], s["energy_dense"], s["energy_linop"],
                              abs(t["energy"] - s["energy_linop"]), s["overlap"]])
    print(f"saved: {csv_path}")

    fig, (ax_e, ax_v) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    idxs = list(range(n))
    ax_e.plot(idxs, [abs(trajectory[i]["energy"] - shadow_records[i]["energy_linop"]) for i in range(n)],
              "-", color="#822727", linewidth=1.0, marker=".", markersize=3)
    ax_e.set_yscale("log")
    ax_e.set_ylabel("|E_dense - E_linop| (shadow, same state)", fontsize=11)
    ax_e.set_title(f"Shadow test: energy diff and eigenvector overlap, chi={args.chi}, "
                   f"seed={args.seed}, {args.sweeps} sweeps", fontsize=13)
    ax_e.grid(alpha=0.3, which="both")
    ax_v.plot(idxs, [1 - shadow_records[i]["overlap"] for i in range(n)],
              "-", color="#2a78d6", linewidth=1.0, marker=".", markersize=3)
    ax_v.set_yscale("log")
    ax_v.set_xlabel("cumulative superblock update index", fontsize=11)
    ax_v.set_ylabel("1 - |<v_dense|v_shadow>|", fontsize=11)
    ax_v.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, f"{tag}.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved: {out_png}")
    print("DONE")


# ── warmstart ────────────────────────────────────────────────────────────

def cmd_warmstart(args):
    os.makedirs(WARMSTART_DIR, exist_ok=True)
    ham, _ = make_hamiltonian()

    print(f"[chi={args.chi_low}] warm-up ladder -> {args.chi_low}, then {args.sweeps} recorded sweeps", flush=True)
    chi_ladder = warmup_ladder(args.chi_low)
    _, tensor_folder = make_hamiltonian()  # fresh tmpdir just for the tree file
    psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=SHAPE, path=tensor_folder, chi=chi_ladder[0])
    gss = GroundStateSearch(psi, ham, init_bond_dim=chi_ladder[0], max_bond_dim=chi_ladder[0])
    patch_physics_engine(gss)
    ref_edge = gss.psi.top_edge_id
    run_warmup(gss, ref_edge, chi_ladder, opt_structure=0, patched=True)

    gss.run(opt_structure=0, max_num_sweep=args.sweeps, verbose=True, diagnostic_edges=[ref_edge],
            energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
    chi_low_trajectory = gss.superblock_energy_trajectory
    ref_energy = gss.energy[ref_edge]

    indep_e = independent_energy(gss.psi, gss.hamiltonian)
    print(f"\n[chi={args.chi_low}] ref_energy={ref_energy:.10f}  independent_energy={indep_e:.10f}  "
          f"diff={abs(ref_energy - indep_e):.6e}", flush=True)

    checkpoint_file = os.path.join(tempfile.mkdtemp(), "checkpoint.hdf5")
    save_tensor(checkpoint_file, SHAPE, Lx, Ly, None, None, G, PRECISION, args.chi_low, gss.psi)
    print(f"[chi={args.chi_low}] checkpoint saved, final energy = {chi_low_trajectory[-1]['energy']:.10f}\n",
          flush=True)

    for target_chi in args.chi_targets:
        print(f"[chi={args.chi_low} -> chi={target_chi}] loading fresh checkpoint copy", flush=True)
        psi_loaded, _ = load_tensor(checkpoint_file, SHAPE, Lx, Ly, None, None, G, PRECISION, args.chi_low)
        gss2 = GroundStateSearch(psi_loaded, ham, init_bond_dim=args.chi_low, max_bond_dim=target_chi)
        patch_physics_engine(gss2)
        ref_edge2 = gss2.psi.top_edge_id
        gss2.run(opt_structure=0, max_num_sweep=args.sweeps, verbose=True, diagnostic_edges=[ref_edge2],
                 energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
        continuation = gss2.superblock_energy_trajectory
        final_energy = continuation[-1]["energy"]
        print(f"[chi={args.chi_low} -> chi={target_chi}] final energy after {args.sweeps} more sweeps "
              f"= {final_energy:.10f}", flush=True)

        tag = f"chi{args.chi_low}_to_chi{target_chi}"
        csv_path = os.path.join(WARMSTART_DIR, f"{tag}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["update_index", "phase", "sweep", "edge_id", "energy"])
            for i, r in enumerate(chi_low_trajectory):
                writer.writerow([i, f"chi{args.chi_low}", r["sweep"], r["edge_id"], r["energy"]])
            offset = len(chi_low_trajectory)
            for i, r in enumerate(continuation):
                writer.writerow([offset + i, f"chi{target_chi}", r["sweep"], r["edge_id"], r["energy"]])

        idx0 = list(range(len(chi_low_trajectory)))
        en0 = [r["energy"] for r in chi_low_trajectory]
        offset = len(chi_low_trajectory)
        idx1 = list(range(offset, offset + len(continuation)))
        en1 = [r["energy"] for r in continuation]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(idx0, en0, "-", color="#2b6cb0", linewidth=0.8, marker=".", markersize=2, label=f"chi={args.chi_low}")
        ax.plot(idx1, en1, "-", color="#c05621", linewidth=0.8, marker=".", markersize=2,
                label=f"chi={target_chi} (warm-started from chi={args.chi_low})")
        ax.axvline(offset - 0.5, color="red", linestyle="--", linewidth=1.0, label="bond dimension bumped here")
        ax.set_xlabel("cumulative superblock update index")
        ax.set_ylabel("superblock Lanczos energy")
        ax.set_title(f"chi={args.chi_low} plateau warm-started into chi={target_chi} -- g={G}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(WARMSTART_DIR, f"{tag}.png"), dpi=200)
        plt.close(fig)
        print(f"saved: {csv_path} and matching .png")
    print("DONE")


# ── seedscan ─────────────────────────────────────────────────────────────

def cmd_seedscan(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    patch_label = "patched" if args.patch else "unpatched-dense"
    print(f"seedscan: chi={args.chi}, {patch_label}, {args.sweeps} sweeps, "
          f"{args.n_seeds} seeds ({args.seed_start}..{args.seed_start + args.n_seeds - 1})",
          flush=True)

    ham, tensor_folder = make_hamiltonian()
    chi_ladder = warmup_ladder(args.chi)

    results = []
    trajectories = {}  # seed -> list of {update_index, sweep, edge_id, energy, phase}, INCLUDING warmup
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        np.random.seed(seed)
        psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=SHAPE, path=tensor_folder, chi=chi_ladder[0])
        gss = GroundStateSearch(psi, ham, init_bond_dim=chi_ladder[0], max_bond_dim=chi_ladder[0])
        if args.patch:
            patch_physics_engine(gss)
        ref_edge = gss.psi.top_edge_id

        # Inline the warmup ladder (instead of the shared run_warmup() helper)
        # so we can capture each stage's trajectory before the next run() call
        # resets it -- without this, the recorded trajectory below starts
        # AFTER warmup has already brought the state close to converged,
        # hiding the actual "different starts converge together" ramp-up.
        full_traj = []
        for chi in chi_ladder[1:]:
            gss.max_bond_dim = chi
            gss.move_canonical_center(ref_edge)
            gss._prime_renormalized_operators()
            kwargs = dict(opt_structure=0, max_num_sweep=2, verbose=False,
                          energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
            if args.patch:
                kwargs["lanczos_tol"] = 1e-6
                kwargs["lanczos_maxiter"] = 150
            gss.run(**kwargs)
            full_traj.extend(dict(r, phase=f"warmup_chi{chi}") for r in gss.superblock_energy_trajectory)

        gss.run(opt_structure=0, max_num_sweep=args.sweeps, verbose=False,
                energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
        full_traj.extend(dict(r, phase="recorded") for r in gss.superblock_energy_trajectory)

        final_energy = gss.energy.get(ref_edge)
        stuck = abs(final_energy - (-60.0)) < args.stuck_tol
        results.append((seed, final_energy, stuck))
        trajectories[seed] = full_traj
        print(f"  seed {seed}: final_energy={final_energy:.10f}  "
              f"{'STUCK near -60' if stuck else 'escaped'}", flush=True)

    seeds = [r[0] for r in results]
    energies = [r[1] for r in results]
    stuck_flags = [r[2] for r in results]
    n_stuck = sum(stuck_flags)
    mean_e = float(np.mean(energies))
    std_e = float(np.std(energies))
    escaped_energies = [e for e, s in zip(energies, stuck_flags) if not s]

    tag = f"g{G:.3f}_chi{args.chi}_{patch_label}_seedscan"
    csv_path = os.path.join(DATA_DIR, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "final_energy", "stuck_near_minus60"])
        for seed, e, s in results:
            writer.writerow([seed, e, int(s)])
    print(f"\nsaved: {csv_path}")

    print(f"\n--- summary over {args.n_seeds} seeds (chi={args.chi}, {patch_label}, {args.sweeps} sweeps) ---")
    print(f"mean final energy = {mean_e:.6f}  (std = {std_e:.6f})")
    print(f"seeds stuck near -60 (within {args.stuck_tol}): {n_stuck}/{args.n_seeds}")
    if escaped_energies:
        print(f"mean final energy AMONG ESCAPED seeds only = {np.mean(escaped_energies):.6f} "
              f"(std = {np.std(escaped_energies):.6f}, n={len(escaped_energies)})")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#822727" if s else "#2a78d6" for s in stuck_flags]
    ax.scatter(seeds, energies, c=colors, s=60, zorder=3)
    ax.axhline(mean_e, color="gray", linestyle="--", linewidth=1.2,
               label=f"mean over all seeds = {mean_e:.4f}")
    ax.axhline(-60.0, color="#822727", linestyle=":", linewidth=1.0, alpha=0.6,
               label="trivial cold-start value (-60)")
    ax.set_xlabel("random seed")
    ax.set_ylabel("final energy (ref edge)")
    ax.set_title(f"chi={args.chi} ({patch_label}): final energy across {args.n_seeds} random seeds "
                 f"-- g={G}, {args.sweeps} sweeps\n"
                 f"{n_stuck}/{args.n_seeds} stuck near -60 (red), {args.n_seeds - n_stuck} escaped (blue)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(FIG_DIR, f"{tag}.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved: {out_png}")

    # full per-update trajectories (warmup + recorded), all seeds overlaid
    traj_csv_path = os.path.join(DATA_DIR, f"{tag}_trajectories.csv")
    with open(traj_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "update_index", "phase", "sweep", "edge_id", "energy"])
        for seed, traj in trajectories.items():
            for i, r in enumerate(traj):
                writer.writerow([seed, i, r["phase"], r["sweep"], r["edge_id"], r["energy"]])
    print(f"saved: {traj_csv_path}")

    # index at which the "recorded" (post-warmup) phase begins -- same for
    # every seed, since the warmup schedule is identical regardless of seed
    recorded_start = next(i for i, r in enumerate(next(iter(trajectories.values()))) if r["phase"] == "recorded")

    fig, ax = plt.subplots(figsize=(12, 7))
    for seed, traj in trajectories.items():
        idx = list(range(len(traj)))
        en = [r["energy"] for r in traj]
        color = "#822727" if abs(en[-1] - (-60.0)) < args.stuck_tol else "#2a78d6"
        ax.plot(idx, en, "-", color=color, linewidth=1.2, alpha=0.35)
    ax.axvline(recorded_start - 0.5, color="black", linestyle="-", linewidth=1.0, alpha=0.6,
               label="warmup ends / recorded sweeps begin")
    ax.axhline(mean_e, color="gray", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"mean final energy = {mean_e:.6f}")
    ax.set_xlabel("cumulative superblock update index (includes warmup)", fontsize=12)
    ax.set_ylabel("superblock Lanczos energy", fontsize=12)
    ax.set_title(f"chi={args.chi} ({patch_label}): superblock energy trajectories from a fresh random tree, "
                 f"{args.n_seeds} random seeds overlaid (alpha=0.35) -- g={G}",
                 fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_traj_png = os.path.join(FIG_DIR, f"{tag}_trajectories.png")
    fig.savefig(out_traj_png, dpi=200)
    plt.close(fig)
    print(f"saved: {out_traj_png}")
    print("DONE")


# ── chiladder ────────────────────────────────────────────────────────────
#
# Proper (not the abbreviated 2-cheap-sweep) chi ladder, unpatched by
# default: since ttnopt/ground_state_search.py -- the actual `gss` CLI
# entry point sweep.py drives -- calls patch_physics_engine() UNCONDITIONALLY
# (no flag to disable it), the "well-converged chi=50" reference used
# earlier in this investigation was ITSELF produced by the patched path,
# not an independent ground truth. So comparing unpatched chi=20 against
# it to invoke the variational principle was comparing two things that
# could both carry the same issue -- not a clean argument. This subcommand
# instead does a real, honest convergence study entirely within the
# unpatched (trusted) path: energy AND entanglement entropy at each chi,
# to see directly whether the state is converging (entropy plateauing) or
# still growing into a larger chi. Optionally continues (still unpatched)
# from the converged top-of-ladder state to a higher target chi, timed,
# to test whether warm-starting from an already-good state lets the
# original dense Lanczos (no restart, no subspace cap -- see abtest's
# docstring for why that's slow from a cold/random start) actually
# complete in practical time.

def cmd_chiladder(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    g = args.g if args.g is not None else G
    patch_label = "patched" if args.patch else "unpatched-dense"
    sweeps_per_stage = args.sweeps if len(args.sweeps) == len(args.chis) else [args.sweeps[0]] * len(args.chis)
    print(f"chiladder: chis={args.chis}, sweeps/stage={sweeps_per_stage}, {patch_label}, g={g}", flush=True)

    ham, tensor_folder = make_hamiltonian(g=g)
    psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=SHAPE, path=tensor_folder, chi=args.chis[0])
    gss = GroundStateSearch(psi, ham, init_bond_dim=args.chis[0], max_bond_dim=args.chis[0])
    if args.patch:
        patch_physics_engine(gss)
    ref_edge = gss.psi.top_edge_id

    records = []
    stage_summaries = []
    cum_sweep = 0
    for stage_idx, (chi, n_sweeps) in enumerate(zip(args.chis, sweeps_per_stage)):
        if stage_idx > 0:
            gss.max_bond_dim = chi
            gss.move_canonical_center(ref_edge)
            gss._prime_renormalized_operators()
        print(f"\n[chi={chi}] {n_sweeps} sweeps", flush=True)
        for s in range(n_sweeps):
            gss.run(opt_structure=0, max_num_sweep=1, verbose=False,
                    energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
            cum_sweep += 1
            e = gss.energy.get(ref_edge)
            ent_ref = gss.entanglement.get(ref_edge)
            ent_max = max(gss.entanglement.values()) if gss.entanglement else float("nan")
            records.append({"chi": chi, "sweep_in_stage": s + 1, "cumulative_sweep": cum_sweep,
                             "energy": e, "entropy_ref": ent_ref, "entropy_max": ent_max})
            print(f"  sweep {s + 1}/{n_sweeps}: E={e:.10f}  S_ref={ent_ref:.6f}  S_max={ent_max:.6f}", flush=True)
        stage_summaries.append(dict(records[-1]))

    print(f"\n--- chi ladder summary ({patch_label}) ---")
    print(f"{'chi':>6} {'final_energy':>16} {'S_ref':>10} {'S_max':>10}")
    for r in stage_summaries:
        print(f"{r['chi']:>6} {r['energy']:>16.10f} {r['entropy_ref']:>10.6f} {r['entropy_max']:>10.6f}")

    if args.continue_to:
        target = args.continue_to
        print(f"\n[chi={args.chis[-1]} -> chi={target}] warm-started continuation, "
              f"{args.continue_sweeps} sweep(s), timing each", flush=True)
        gss.max_bond_dim = target
        gss.move_canonical_center(ref_edge)
        gss._prime_renormalized_operators()
        for s in range(args.continue_sweeps):
            t0 = time.time()
            gss.run(opt_structure=0, max_num_sweep=1, verbose=False,
                    energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
            dt = time.time() - t0
            cum_sweep += 1
            e = gss.energy.get(ref_edge)
            ent_ref = gss.entanglement.get(ref_edge)
            ent_max = max(gss.entanglement.values()) if gss.entanglement else float("nan")
            records.append({"chi": target, "sweep_in_stage": s + 1, "cumulative_sweep": cum_sweep,
                             "energy": e, "entropy_ref": ent_ref, "entropy_max": ent_max})
            print(f"  [chi={target}] sweep {s + 1}/{args.continue_sweeps} took {dt:.1f}s: "
                  f"E={e:.10f}  S_ref={ent_ref:.6f}  S_max={ent_max:.6f}", flush=True)
        stage_summaries.append(dict(records[-1]))

    tag = f"g{g:.3f}_{patch_label}_chiladder_{'-'.join(str(c) for c in args.chis)}"
    if args.continue_to:
        tag += f"_to{args.continue_to}"
    csv_path = os.path.join(DATA_DIR, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chi", "sweep_in_stage", "cumulative_sweep", "energy", "entropy_ref", "entropy_max"])
        for r in records:
            writer.writerow([r["chi"], r["sweep_in_stage"], r["cumulative_sweep"],
                              r["energy"], r["entropy_ref"], r["entropy_max"]])
    print(f"\nsaved: {csv_path}")

    # trajectory plot: energy + entropy vs cumulative sweep, stage boundaries marked
    chis_seen = [r["chi"] for r in records]
    boundaries = [i - 0.5 for i in range(1, len(chis_seen)) if chis_seen[i] != chis_seen[i - 1]]
    fig, (ax_e, ax_s) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    xs = [r["cumulative_sweep"] for r in records]
    ax_e.plot(xs, [r["energy"] for r in records], "-o", color="#2a78d6", markersize=3)
    for b in boundaries:
        ax_e.axvline(b, color="red", linestyle="--", linewidth=0.8)
    ax_e.set_ylabel("energy (ref edge)")
    ax_e.set_title(f"chi ladder {args.chis}{f' -> {args.continue_to}' if args.continue_to else ''} "
                    f"({patch_label}) -- g={g}")
    ax_s.plot(xs, [r["entropy_ref"] for r in records], "-o", color="#eb6834", markersize=3, label="S (ref edge)")
    ax_s.plot(xs, [r["entropy_max"] for r in records], "-o", color="#822727", markersize=3, label="S (max over all edges)")
    for b in boundaries:
        ax_s.axvline(b, color="red", linestyle="--", linewidth=0.8)
    ax_s.set_xlabel("cumulative sweep")
    ax_s.set_ylabel("entanglement entropy")
    ax_s.legend()
    fig.tight_layout()
    out_traj = os.path.join(FIG_DIR, f"{tag}_trajectory.png")
    fig.savefig(out_traj, dpi=200)
    plt.close(fig)
    print(f"saved: {out_traj}")

    # summary plot: final energy and final entropy vs chi
    chis_summary = [r["chi"] for r in stage_summaries]
    fig, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(12, 5))
    ax_e.plot(chis_summary, [r["energy"] for r in stage_summaries], "o-", color="#2a78d6")
    ax_e.set_xlabel("chi")
    ax_e.set_ylabel("final energy")
    ax_e.set_title("energy vs chi")
    ax_e.grid(alpha=0.3)
    ax_s.plot(chis_summary, [r["entropy_ref"] for r in stage_summaries], "o-", color="#eb6834", label="S (ref edge)")
    ax_s.plot(chis_summary, [r["entropy_max"] for r in stage_summaries], "o-", color="#822727", label="S (max)")
    ax_s.set_xlabel("chi")
    ax_s.set_ylabel("final entanglement entropy")
    ax_s.set_title("entropy vs chi -- plateau = converged, still rising = need larger chi")
    ax_s.legend()
    ax_s.grid(alpha=0.3)
    fig.tight_layout()
    out_summary = os.path.join(FIG_DIR, f"{tag}_summary.png")
    fig.savefig(out_summary, dpi=200)
    plt.close(fig)
    print(f"saved: {out_summary}")
    print("DONE")


# ── chiladder-gscan ──────────────────────────────────────────────────────
#
# Same chi ladder as `chiladder`, repeated over a grid of g values (matching
# z3_ed_3x3_scan.py's g convention: |g| from g_min to g_max), unpatched by
# default. Only the FINAL energy/entropy per (g, chi) is kept -- no per-
# sweep trajectory here, since the point is E(chi,g) and S(chi,g) surfaces,
# not individual convergence trajectories (use `chiladder` at one g for that).

def cmd_chiladder_gscan(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    patch_label = "patched" if args.patch else "unpatched-dense"
    sweeps_per_stage = args.sweeps if len(args.sweeps) == len(args.chis) else [args.sweeps[0]] * len(args.chis)
    g_values = np.linspace(args.g_min, args.g_max, args.n_g)
    print(f"chiladder-gscan: chis={args.chis}, sweeps/stage={sweeps_per_stage}, {patch_label}, "
          f"{args.n_g} g-points from {args.g_min} to {args.g_max}", flush=True)

    results = []  # (g, chi, energy, entropy_ref, entropy_max)
    for gi, g in enumerate(g_values):
        g = float(g)
        ham, tensor_folder = make_hamiltonian(g)
        psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=SHAPE, path=tensor_folder, chi=args.chis[0])
        gss = GroundStateSearch(psi, ham, init_bond_dim=args.chis[0], max_bond_dim=args.chis[0])
        if args.patch:
            patch_physics_engine(gss)
        ref_edge = gss.psi.top_edge_id

        for stage_idx, (chi, n_sweeps) in enumerate(zip(args.chis, sweeps_per_stage)):
            if stage_idx > 0:
                gss.max_bond_dim = chi
                gss.move_canonical_center(ref_edge)
                gss._prime_renormalized_operators()
            gss.run(opt_structure=0, max_num_sweep=n_sweeps, verbose=False,
                    energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
            e = gss.energy.get(ref_edge)
            ent_ref = gss.entanglement.get(ref_edge)
            ent_max = max(gss.entanglement.values()) if gss.entanglement else float("nan")
            results.append((g, chi, e, ent_ref, ent_max))
            print(f"  [{gi + 1}/{args.n_g}] g={g:.3f} chi={chi}: E={e:.8f}  "
                  f"S_ref={ent_ref:.6f}  S_max={ent_max:.6f}", flush=True)

    tag = f"{patch_label}_chiladder_gscan_{'-'.join(str(c) for c in args.chis)}"
    csv_path = os.path.join(DATA_DIR, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["g", "chi", "energy", "entropy_ref", "entropy_max"])
        writer.writerows(results)
    print(f"\nsaved: {csv_path}")

    colors = {args.chis[0]: "#2a78d6", args.chis[1] if len(args.chis) > 1 else -1: "#eb6834",
              args.chis[2] if len(args.chis) > 2 else -2: "#1baf7a"}
    fallback_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#822727", "#4a3aa7"]
    for i, c in enumerate(args.chis):
        colors.setdefault(c, fallback_colors[i % len(fallback_colors)])

    fig, ax = plt.subplots(figsize=(10, 6))
    for chi in args.chis:
        rows = [r for r in results if r[1] == chi]
        g_abs = [abs(r[0]) for r in rows]
        ax.plot(g_abs, [r[2] for r in rows], "o-", color=colors[chi], label=f"chi={chi}")
    ax.set_xlabel("|g|")
    ax.set_ylabel("final energy")
    ax.set_title(f"E(chi, g) -- {patch_label}, chis={args.chis}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_e = os.path.join(FIG_DIR, f"{tag}_energy.png")
    fig.savefig(out_e, dpi=200)
    plt.close(fig)
    print(f"saved: {out_e}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for chi in args.chis:
        rows = [r for r in results if r[1] == chi]
        g_abs = [abs(r[0]) for r in rows]
        ax.plot(g_abs, [r[4] for r in rows], "o-", color=colors[chi], label=f"chi={chi}")
    ax.set_xlabel("|g|")
    ax.set_ylabel("final entanglement entropy (max over all edges)")
    ax.set_title(f"S(chi, g) -- {patch_label}, chis={args.chis}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_s = os.path.join(FIG_DIR, f"{tag}_entropy.png")
    fig.savefig(out_s, dpi=200)
    plt.close(fig)
    print(f"saved: {out_s}")
    print("DONE")


# ── compare ──────────────────────────────────────────────────────────────

def _sweep_boundaries(sweep):
    return [i - 0.5 for i in range(1, len(sweep)) if sweep[i] != sweep[i - 1]]


def _style_log_yaxis(ax, ymin, ymax):
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(), numticks=20))
    ax.grid(True, which="major", axis="both", alpha=0.35)


def cmd_compare_opt_structure(args):
    chis = [20, 50, 100]
    colors = {20: "#2b6cb0", 50: "#c05621", 100: "#2f855a"}
    data = {}
    for chi in chis:
        path = os.path.join(DATA_DIR, f"g{G:.3f}_chi{chi}_patched_optstruct{args.opt_structure}.csv")
        if not os.path.exists(path):
            print(f"!! no data for chi={chi} at {path} -- skipping !!")
            continue
        idx, sweep, energy = [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                idx.append(int(row["update_index"]))
                sweep.append(int(row["sweep"]))
                energy.append(float(row["energy"]))
        data[chi] = {"idx": idx, "sweep": sweep, "energy": energy}
        print(f"chi={chi}: {len(idx)} updates, {max(sweep)} sweeps")

    if not data:
        raise RuntimeError(f"no trajectory CSVs found for opt_structure={args.opt_structure}")

    longest = max(data.values(), key=lambda d: len(d["idx"]))
    boundaries = _sweep_boundaries(longest["sweep"])

    for suffix, min_sweep in [("", None), ("_zoom", 3)]:
        fig, ax = plt.subplots(figsize=(12, 6))
        max_idx = 0
        for chi, d in data.items():
            if min_sweep is None:
                idx, en = d["idx"], d["energy"]
            else:
                idx, en = zip(*[(i, e) for i, s, e in zip(d["idx"], d["sweep"], d["energy"]) if s >= min_sweep])
            ax.plot(idx, en, "-", color=colors.get(chi), linewidth=0.8, marker=".", markersize=2, label=f"chi={chi}")
            max_idx = max(max_idx, max(idx))
        for b in boundaries:
            if b <= max_idx:
                ax.axvline(b, color="red", linestyle="--", linewidth=0.6, alpha=1)
        ax.set_xlabel("cumulative superblock update index")
        ax.set_ylabel("superblock Lanczos energy")
        title_suffix = f" (sweep >= {min_sweep}, zoomed)" if min_sweep else ""
        ax.set_title(f"Superblock energy trajectory comparison{title_suffix} -- g={G}, "
                     f"chis={sorted(data.keys())}, opt_structure={args.opt_structure}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"superblock_trajectory_compare_g{G:.3f}_optstruct{args.opt_structure}{suffix}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"saved: {out}")
    print("DONE")


def cmd_compare_lattice_sizes(args):
    def load(path):
        idx, gap = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                idx.append(int(row["update_index"]))
                gap.append(abs(float(row["ref_energy"]) - float(row["independent_energy"])))
        return idx, gap

    patch_label = "patched" if args.patch else "unpatched-dense"
    series = []
    for lx, ly, chi in args.sizes:
        size_prefix = "" if (lx, ly) == (Lx, Ly) else f"{lx}x{ly}_"
        path = os.path.join(DATA_DIR, f"{size_prefix}g{G:.3f}_chi{chi}_{patch_label}_optstruct0_independent_energy.csv")
        if not os.path.exists(path):
            patch_flag = "--patch " if args.patch else ""
            print(f"!! no data at {path} -- run e.g.\n"
                  f"  python {os.path.basename(__file__)} trajectory --chi {chi} --lx {lx} --ly {ly} "
                  f"{patch_flag}--sweeps <N> --independent\n   first -- skipping !!")
            continue
        idx, gap = load(path)
        if args.max_updates is not None:
            idx, gap = zip(*[(i, g) for i, g in zip(idx, gap) if i <= args.max_updates]) if idx else ([], [])
        series.append((lx, ly, chi, idx, gap))
        print(f"{lx}x{ly}, chi={chi}: {len(idx)} updates (after truncation), max gap={max(gap):.3e}")

    if not series:
        raise RuntimeError("no data found for any requested (lx, ly, chi)")

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#822727", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (lx, ly, chi, idx, gap) in enumerate(series):
        ax.plot(idx, gap, "-", color=colors[i % len(colors)], linewidth=1.0, marker=".", markersize=3,
                label=f"{lx}x{ly} (chi={chi})")
    ymax = max(max(g) for _, _, _, _, g in series) * 3
    _style_log_yaxis(ax, 1e-15, ymax)
    if args.max_updates is not None:
        ax.set_xlim(0, args.max_updates)
    ax.set_xlabel("cumulative superblock update index", fontsize=12)
    ax.set_ylabel("|ref_energy - independent_energy|", fontsize=12)
    ax.set_title(f"{patch_label} tracked-vs-independent energy gap across lattice sizes -- g={G}", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    tag = f"{patch_label}_gap_by_lattice_size_" + "-".join(f"{lx}x{ly}chi{chi}" for lx, ly, chi, *_ in series)
    out = os.path.join(FIG_DIR, f"{tag}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved: {out}")
    print("DONE")


def cmd_compare_patched_vs_unpatched(args):
    def load(path):
        idx, sweep, ref, indep = [], [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                idx.append(int(row["update_index"]))
                sweep.append(int(row["sweep"]))
                ref.append(float(row["ref_energy"]))
                indep.append(float(row["independent_energy"]))
        return idx, sweep, ref, indep

    # Same size_prefix convention cmd_trajectory uses when writing these, so
    # a non-default lattice (e.g. --lx 3 --ly 3) can be compared too.
    size_prefix = "" if (args.lx, args.ly) == (Lx, Ly) else f"{args.lx}x{args.ly}_"
    size_label = f"{args.lx}x{args.ly}"
    p_path = os.path.join(DATA_DIR, f"{size_prefix}g{G:.3f}_chi{args.chi}_patched_optstruct0_independent_energy.csv")
    u_path = os.path.join(DATA_DIR, f"{size_prefix}g{G:.3f}_chi{args.chi}_unpatched-dense_optstruct0_independent_energy.csv")
    for p in (p_path, u_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found -- run e.g.\n"
                f"  python {os.path.basename(__file__)} trajectory --lx {args.lx} --ly {args.ly} "
                f"--chi {args.chi} --patch --independent\n"
                f"  python {os.path.basename(__file__)} trajectory --lx {args.lx} --ly {args.ly} "
                f"--chi {args.chi} --independent"
            )
    idx_p, sweep_p, ref_p, indep_p = load(p_path)
    idx_u, sweep_u, ref_u, indep_u = load(u_path)
    gap_p = [abs(r - i) for r, i in zip(ref_p, indep_p)]
    gap_u = [abs(r - i) for r, i in zip(ref_u, indep_u)]

    # 1. cleaner-axis unpatched-only gap plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx_u, gap_u, "-", color="#822727", linewidth=1.0, marker=".", markersize=3)
    for b in _sweep_boundaries(sweep_u):
        ax.axvline(b, color="red", linestyle="--", linewidth=0.6, alpha=0.6)
    _style_log_yaxis(ax, 1e-15, 1e-5)
    ax.set_xlabel("cumulative superblock update index", fontsize=12)
    ax.set_ylabel("|ref_energy - independent_energy|", fontsize=12)
    ax.set_title(f"{size_label} chi={args.chi} (unpatched-dense) tracked-vs-independent energy gap\ng={G}", fontsize=13)
    fig.tight_layout()
    out1 = os.path.join(FIG_DIR, f"{size_prefix}g{G:.3f}_chi{args.chi}_unpatched_diff_readable.png")
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    print(f"saved: {out1}")

    # 2. combined gap overlay
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx_p, gap_p, "-", color="#eb6834", linewidth=1.0, marker=".", markersize=3, label="patched")
    ax.plot(idx_u, gap_u, "-", color="#2a78d6", linewidth=1.0, marker=".", markersize=3, label="unpatched (dense)")
    _style_log_yaxis(ax, 1e-15, 1e-1)
    ax.set_xlabel("cumulative superblock update index", fontsize=12)
    ax.set_ylabel("|ref_energy - independent_energy|", fontsize=12)
    ax.set_title(f"{size_label} chi={args.chi}: patched vs unpatched tracked-vs-independent gap -- g={G}", fontsize=13)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    out2 = os.path.join(FIG_DIR, f"{size_prefix}g{G:.3f}_chi{args.chi}_patched_vs_unpatched_diff.png")
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    print(f"saved: {out2}")

    # 3. raw energies overlay (full range + zoom to the shorter run's range)
    for suffix, n in [("", None), ("_zoom", len(idx_u))]:
        fig, ax = plt.subplots(figsize=(13, 7))
        sl = slice(0, n)
        ax.plot(idx_p[sl], ref_p[sl], "-", color="#eb6834", linewidth=1.3, label="patched: ref_energy")
        ax.plot(idx_p[sl], indep_p[sl], "--", color="#c05621", linewidth=1.0, alpha=0.85, label="patched: independent_energy")
        ax.plot(idx_u, ref_u, "-", color="#2a78d6", linewidth=1.3, label="unpatched: ref_energy")
        ax.plot(idx_u, indep_u, "--", color="#184f95", linewidth=1.0, alpha=0.85, label="unpatched: independent_energy")
        for b in _sweep_boundaries(sweep_u):
            ax.axvline(b, color="gray", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("cumulative superblock update index", fontsize=12)
        ax.set_ylabel("energy", fontsize=12)
        ax.set_title(f"chi={args.chi}: raw energies, patched vs unpatched -- g={G}", fontsize=13)
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out3 = os.path.join(FIG_DIR, f"g{G:.3f}_chi{args.chi}_energies_patched_vs_unpatched{suffix}.png")
        fig.savefig(out3, dpi=200)
        plt.close(fig)
        print(f"saved: {out3}")
    print("DONE")


# ── trunc-floor ──────────────────────────────────────────────────────────
#
# Professor's test: is the chi=20 E~-60 plateau set by the two-site SVD/
# truncation step itself, or by the trajectory/initialization (a genuine
# local optimum of what two-site Lanczos can find in that subspace)?
#
# run()'s existing diagnostic_edges/local_update_diagnostics instrumentation
# (GroundStateSearch.py, the instrument_this_update block) already records,
# for any edge in diagnostic_edges, both:
#   E_lanczos          -- the raw eigenvalue lanczos()/ttn_eigensolver()
#                          returns for that update, BEFORE decompose_two_tensors
#   E_after_truncation -- the Rayleigh quotient of the SAME two-site block
#                          reconstructed from u,s,v AFTER the SVD truncation
#                          back to max_bond_dim
# at zero extra matvec cost beyond the 2 extra Rayleigh quotients per
# instrumented update. Passing every internal edge id as diagnostic_edges
# instruments every single update in every sweep (opt_structure=0 only).
#
# Stage 1: chi=20, UNPATCHED (the trusted engine), cold start -> convergence.
# Stage 2: warm-start that exact converged state into chi=CHI_TARGET,
# PATCHED (chi=100 skipped for now per instruction), continued for more
# sweeps, same full instrumentation.
#
# If E_after_truncation tracks E_lanczos closely in BOTH stages, truncation
# is not the bottleneck -- the floor is set by what two-site Lanczos itself
# can find (trajectory/initialization). If E_after_truncation sits well
# above E_lanczos (truncation throwing away improvement Lanczos found),
# especially once chi grows to 50, that implicates the truncation step.

def cmd_truncfloor(args):
    os.makedirs(WARMSTART_DIR, exist_ok=True)
    ham, _ = make_hamiltonian()

    print(f"[trunc-floor test] chi={args.chi_low} unpatched -> warm start -> "
          f"chi={args.chi_target} patched, full E_lanczos vs E_after_truncation "
          f"instrumentation on every update", flush=True)

    chi_ladder = warmup_ladder(args.chi_low)
    _, tensor_folder = make_hamiltonian()
    psi = get_rnd_tree(Lx=Lx, Ly=Ly, shape=SHAPE, path=tensor_folder, chi=chi_ladder[0])
    gss = GroundStateSearch(psi, ham, init_bond_dim=chi_ladder[0], max_bond_dim=chi_ladder[0])
    # deliberately NOT patched -- this is the trusted stage-1 engine
    ref_edge = gss.psi.top_edge_id
    run_warmup(gss, ref_edge, chi_ladder, opt_structure=0, patched=False)

    diag_edges_low = list(set(e[2] for e in gss.psi.edges))
    gss.run(opt_structure=0, max_num_sweep=args.sweeps_low, verbose=True,
            diagnostic_edges=diag_edges_low,
            energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
    low_diag = list(gss.local_update_diagnostics)
    low_traj = list(gss.superblock_energy_trajectory)
    print(f"[chi={args.chi_low} unpatched] final energy = {low_traj[-1]['energy']:.10f}  "
          f"({len(low_diag)} instrumented updates)", flush=True)

    checkpoint_file = os.path.join(tempfile.mkdtemp(), "checkpoint.hdf5")
    save_tensor(checkpoint_file, SHAPE, Lx, Ly, None, None, G, PRECISION, args.chi_low, gss.psi)

    psi_loaded, _ = load_tensor(checkpoint_file, SHAPE, Lx, Ly, None, None, G, PRECISION, args.chi_low)
    gss2 = GroundStateSearch(psi_loaded, ham, init_bond_dim=args.chi_low, max_bond_dim=args.chi_target)
    patch_physics_engine(gss2)
    diag_edges_target = list(set(e[2] for e in gss2.psi.edges))
    gss2.run(opt_structure=0, max_num_sweep=args.sweeps_target, verbose=True,
             diagnostic_edges=diag_edges_target,
             energy_convergence_threshold=0.0, entanglement_convergence_threshold=0.0)
    target_diag = list(gss2.local_update_diagnostics)
    target_traj = list(gss2.superblock_energy_trajectory)
    print(f"[chi={args.chi_target} patched, warm-started] final energy = "
          f"{target_traj[-1]['energy']:.10f}  ({len(target_diag)} instrumented updates)", flush=True)

    tag = f"truncfloor_chi{args.chi_low}unpatched_to_chi{args.chi_target}patched"
    csv_path = os.path.join(WARMSTART_DIR, f"{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["update_index", "phase", "sweep", "edge_id", "E_before", "E_lanczos", "E_after_truncation"])
        for i, r in enumerate(low_diag):
            writer.writerow([i, f"chi{args.chi_low}_unpatched", r["sweep"], r["edge_id"],
                              r["E_before"], r["E_lanczos"], r["E_after_truncation"]])
        offset = len(low_diag)
        for i, r in enumerate(target_diag):
            writer.writerow([offset + i, f"chi{args.chi_target}_patched", r["sweep"], r["edge_id"],
                              r["E_before"], r["E_lanczos"], r["E_after_truncation"]])
    print(f"saved: {csv_path}")

    def unpack(diag, offset=0):
        idx = list(range(offset, offset + len(diag)))
        el = [r["E_lanczos"] for r in diag]
        ea = [r["E_after_truncation"] for r in diag]
        return idx, el, ea

    idx0, el0, ea0 = unpack(low_diag)
    idx1, el1, ea1 = unpack(target_diag, offset=len(low_diag))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax = axes[0]
    ax.plot(idx0, el0, "-", color="#2b6cb0", linewidth=0.7, marker=".", markersize=2,
            label=f"chi={args.chi_low} unpatched: E_lanczos")
    ax.plot(idx0, ea0, "--", color="#63b3ed", linewidth=0.9, marker=".", markersize=2,
            label=f"chi={args.chi_low} unpatched: E_after_truncation")
    ax.plot(idx1, el1, "-", color="#c05621", linewidth=0.7, marker=".", markersize=2,
            label=f"chi={args.chi_target} patched: E_lanczos")
    ax.plot(idx1, ea1, "--", color="#f6ad55", linewidth=0.9, marker=".", markersize=2,
            label=f"chi={args.chi_target} patched: E_after_truncation")
    ax.axvline(len(low_diag) - 0.5, color="red", linestyle=":", linewidth=1.2,
               label="bond dim bumped here (warm start)")
    ax.set_ylabel("energy")
    ax.set_title(f"raw Lanczos vs post-truncation energy, every update -- g={G}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    diff0 = [ea - el for ea, el in zip(ea0, el0)]
    diff1 = [ea - el for ea, el in zip(ea1, el1)]
    ax2.plot(idx0, diff0, "-", color="#2b6cb0", linewidth=0.8, marker=".", markersize=2,
             label=f"chi={args.chi_low} unpatched")
    ax2.plot(idx1, diff1, "-", color="#c05621", linewidth=0.8, marker=".", markersize=2,
             label=f"chi={args.chi_target} patched")
    ax2.axhline(0.0, color="black", linewidth=0.6)
    ax2.axvline(len(low_diag) - 0.5, color="red", linestyle=":", linewidth=1.2)
    ax2.set_xlabel("cumulative superblock update index")
    ax2.set_ylabel("E_after_truncation - E_lanczos")
    ax2.set_title("truncation cost per update (~0 => truncation is lossless for the energy)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(WARMSTART_DIR, f"{tag}.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved: {out_png}")
    print("DONE")


def cmd_compare(args):
    if args.kind == "opt_structure":
        cmd_compare_opt_structure(args)
    elif args.kind == "lattice_size":
        cmd_compare_lattice_sizes(args)
    else:
        cmd_compare_patched_vs_unpatched(args)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_traj = sub.add_parser("trajectory")
    p_traj.add_argument("--chi", type=int, required=True)
    p_traj.add_argument("--lx", type=int, default=Lx, help="lattice width (default 5, matching this file's main investigation)")
    p_traj.add_argument("--ly", type=int, default=Ly, help="lattice height (default 5)")
    p_traj.add_argument("--patch", action="store_true")
    p_traj.add_argument("--opt-structure", type=int, default=0, choices=[0, 1, 2])
    p_traj.add_argument("--sweeps", type=int, default=5)
    p_traj.add_argument("--independent", action="store_true",
                         help="compute independent_energy() after every update (expensive)")
    p_traj.add_argument("--lanczos-tol", type=float, default=None,
                         help="override the local eigensolver's convergence tolerance "
                              "(dense default 1e-13, patched/eigsh default 1e-10) -- "
                              "e.g. --lanczos-tol 1e-10 on an unpatched run tests whether "
                              "tolerance ALONE (same algorithm) reproduces patched's divergence")
    p_traj.set_defaults(func=cmd_trajectory)

    p_ab = sub.add_parser("abtest")
    p_ab.add_argument("--chi", type=int, default=20)
    p_ab.add_argument("--seed", type=int, default=42)
    p_ab.add_argument("--sweeps", type=int, default=1)
    p_ab.set_defaults(func=cmd_abtest)

    p_ws = sub.add_parser("warmstart")
    p_ws.add_argument("--chi-low", type=int, default=20)
    p_ws.add_argument("--chi-targets", type=int, nargs="+", default=[50, 100])
    p_ws.add_argument("--sweeps", type=int, default=15)
    p_ws.set_defaults(func=cmd_warmstart)

    p_ss = sub.add_parser("seedscan")
    p_ss.add_argument("--chi", type=int, default=20)
    p_ss.add_argument("--patch", action="store_true")
    p_ss.add_argument("--sweeps", type=int, default=10)
    p_ss.add_argument("--n-seeds", type=int, default=15)
    p_ss.add_argument("--seed-start", type=int, default=0)
    p_ss.add_argument("--stuck-tol", type=float, default=0.005,
                       help="|final_energy - (-60)| below this counts as 'stuck'")
    p_ss.set_defaults(func=cmd_seedscan)

    p_cl = sub.add_parser("chiladder")
    p_cl.add_argument("--g", type=float, default=None, help="override module-level G for this run")
    p_cl.add_argument("--chis", type=int, nargs="+", default=[9, 18, 27])
    p_cl.add_argument("--sweeps", type=int, nargs="+", default=[3, 3, 15],
                       help="sweeps per stage; one value repeats for all stages")
    p_cl.add_argument("--patch", action="store_true")
    p_cl.add_argument("--continue-to", type=int, default=None,
                       help="optionally continue the warm-started top-of-ladder state to this chi")
    p_cl.add_argument("--continue-sweeps", type=int, default=3)
    p_cl.set_defaults(func=cmd_chiladder)

    p_clg = sub.add_parser("chiladder-gscan")
    p_clg.add_argument("--chis", type=int, nargs="+", default=[9, 18, 27])
    p_clg.add_argument("--sweeps", type=int, nargs="+", default=[3, 3, 10])
    p_clg.add_argument("--patch", action="store_true")
    p_clg.add_argument("--g-min", type=float, default=-1.5)
    p_clg.add_argument("--g-max", type=float, default=-0.1)
    p_clg.add_argument("--n-g", type=int, default=15)
    p_clg.set_defaults(func=cmd_chiladder_gscan)

    p_tf = sub.add_parser("trunc-floor")
    p_tf.add_argument("--chi-low", type=int, default=20)
    p_tf.add_argument("--chi-target", type=int, default=50)
    p_tf.add_argument("--sweeps-low", type=int, default=15)
    p_tf.add_argument("--sweeps-target", type=int, default=15)
    p_tf.set_defaults(func=cmd_truncfloor)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--kind", choices=["opt_structure", "patched_vs_unpatched", "lattice_size"], required=True)
    p_cmp.add_argument("--opt-structure", type=int, default=0, choices=[0, 1, 2])
    p_cmp.add_argument("--chi", type=int, default=20)
    p_cmp.add_argument("--lx", type=int, default=Lx,
                        help="kind=patched_vs_unpatched only: lattice width of the trajectory data to compare")
    p_cmp.add_argument("--ly", type=int, default=Ly,
                        help="kind=patched_vs_unpatched only: lattice height of the trajectory data to compare")

    def _size_triplet(s):
        lx, ly, chi = s.lower().split("x")
        return (int(lx), int(ly), int(chi))

    p_cmp.add_argument("--sizes", type=_size_triplet, nargs="+", default=[(5, 5, 20), (3, 3, 9)],
                        help="kind=lattice_size only: LxLyxCHI triplets, e.g. --sizes 5x5x20 3x3x9 9x9x9 "
                             "(each needs a prior 'trajectory --patch --independent' run at that lx/ly/chi)")
    p_cmp.add_argument("--patch", action="store_true",
                        help="kind=lattice_size only: compare the patched (default: unpatched) datasets")
    p_cmp.add_argument("--max-updates", type=int, default=None,
                        help="kind=lattice_size only: truncate the x-axis / all series to this many updates")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
