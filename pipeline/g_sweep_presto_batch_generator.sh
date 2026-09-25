#!/bin/bash
# g_sweep_presto_batch_generator.sh
# ------------------------------------
# Generates N SEPARATE, small PBS job scripts (one per g-batch) instead of
# one big multi-worker job requesting a whole node. Why separate jobs, not
# one job with internal numactl-pinned workers (see
# g_sweep_presto_parallel.pbs): presto's nodes report state "job-busy", not
# "job-exclusive", meaning the scheduler CAN pack additional smaller jobs
# onto a node that already has other jobs running on it. A single
# select=1:ncpus=96 request needs an entire node free at once (rare on a
# shared cluster); N separate select=1:ncpus=<small> requests can each slot
# into whatever partial capacity exists RIGHT NOW, on the same node or
# different ones -- almost always starts running much sooner.
#
# Each generated job is fully independent (own g-range slice, own log, own
# process) -- no numactl pinning needed, since PBS's own cpuset already
# restricts each job to its allocated cores; if you want to check whether
# those allocated cores land in a single NUMA domain, run
# `numactl --show` inside one of the running jobs once it starts.
#
# Usage:
#   bash pipeline/g_sweep_presto_batch_generator.sh
# Edit the CONFIG block below, then it prints the qsub commands to run --
# review them, then either paste them in or uncomment the auto-submit line
# at the bottom.

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG -- edit these
# ---------------------------------------------------------------------------
# Hexagon 5x9 vacuum production scan. N_BATCHES=N_G_TOTAL=15 -- one g-point
# per job, not a contiguous-range slice, per the confirmed 8-thread finding
# (thread_count_sweep_presto.pbs) and the job-busy packing model.
N_BATCHES=15
CORES_PER_BATCH=8             # matches THREADS_PER_BATCH -- confirmed optimum
                              # for the hexagon workload at chi=50; chi=81 not
                              # separately re-calibrated (time-constrained
                              # decision, not re-tested at this scale)
THREADS_PER_BATCH=8
WALLTIME="144:00:00"          # 6 days -- Mac took ~2.5 days for g=-1.500
                              # (the easiest point); presto's per-thread
                              # throughput at 8 threads vs the Mac's uncontrolled
                              # thread count is unverified, so this is a
                              # deliberate ~2.4x safety margin, not a measured bound
# ~90.6 GB RSS observed on the Mac for the REAL chi=81 hexagon 5x9 job
# (PID 48402, g=-1.400, sweep 6/40) -- this is a real data point, not a guess.
# 130gb leaves headroom above that peak; an OOM kill partway through a 6-day
# job is far worse than slightly reduced node-packing density. NOTE: at this
# size, memory -- not CORES_PER_BATCH -- is what actually limits how many
# jobs pack onto one node (384gb/node / 130gb =~ 3 jobs/node by memory, vs
# 96 cores/node / 8 cores =~ 12 jobs/node by cores). 15 jobs will really span
# ~5 nodes' worth of memory, not the ~1.25 nodes the core count alone suggests.
MEM_PER_BATCH="130gb"

G_MIN=0.1
G_MAX=1.5
N_G_TOTAL=15
CHIS="9 27 50 81"
SWEEPS_ASCEND=3
LANCZOS_TOL_ASCEND=1e-6        # matches the established hexagon recipe (patched-only)
SWEEPS_CONVERGE=40
SWEEPS_DESCEND="25 20 10"      # per descend stage, 81->50->27->9 (tapered, see g_sweep.py's docstring)
# Entropy convergence has repeatedly been the slower criterion (energy
# converges to ~1e-13 well before entanglement reaches 1e-10 -- see the
# g=1.5 log: max|d(ee)| already clears 1e-8 by sweep 6-9 but keeps grinding
# toward 1e-10). Loosened here to cut wasted sweeps; energy threshold left
# untouched since it was never the bottleneck. Points right at the vacuum
# transition may still be worth a rerun_nonconverged-style top-up look.
ENTANGLEMENT_CONVERGENCE_THRESHOLD=1e-8
LX=5
LY=9
SHAPE=hexagon
ENGINE_FLAG="--patched"
WARM_START_FLAG=""           # cold start every point -- matches the confined-phase-first reliability discipline

OUT_DIR="pipeline/generated_batches"
DRIVE_PATH="/home/fradm/projects/5_Z3"

# ---------------------------------------------------------------------------
mkdir -p "$OUT_DIR"

# Contiguous slice boundaries. Pure-stdlib python (no numpy import -- this
# must not depend on the ttn conda env being active, since it's cheap
# bookkeeping run from whatever `python3` is on PATH at generation time, and
# replicates np.linspace/np.array_split's own splitting rule exactly so the
# union of all batches' slices matches a single un-split run bit for bit).
SLICE_SPEC=$(python3 -c "
n, w = $N_G_TOTAL, $N_BATCHES
g = [$G_MIN + i * ($G_MAX - $G_MIN) / (n - 1) for i in range(n)] if n > 1 else [$G_MIN]
base, rem = divmod(n, w)
sizes = [base + 1 if i < rem else base for i in range(w)]
start = 0
for size in sizes:
    if size == 0:
        continue
    chunk = g[start:start + size]
    print(f'{chunk[0]:.6f} {chunk[-1]:.6f} {len(chunk)}')
    start += size
")
SLICE_MINS=()
SLICE_MAXS=()
SLICE_NS=()
while read -r gmin gmax n; do
    SLICE_MINS+=("$gmin")
    SLICE_MAXS+=("$gmax")
    SLICE_NS+=("$n")
done <<< "$SLICE_SPEC"

echo "Generating ${#SLICE_MINS[@]} batch scripts in $OUT_DIR/ ..."
submit_cmds=()
for i in "${!SLICE_MINS[@]}"; do
    gmin=${SLICE_MINS[$i]}
    gmax=${SLICE_MAXS[$i]}
    ng=${SLICE_NS[$i]}
    script="$OUT_DIR/batch${i}.pbs"

    cat > "$script" <<EOF
#!/bin/bash
#PBS -N g_sweep_b${i}
#PBS -l select=1:ncpus=${CORES_PER_BATCH}:mem=${MEM_PER_BATCH}
#PBS -l walltime=${WALLTIME}
#PBS -j oe
#PBS -o pbs_g_sweep_batch${i}.log

cd \$PBS_O_WORKDIR
PYTHON="\$HOME/.conda/envs/ttn/bin/python"
export OMP_NUM_THREADS=${THREADS_PER_BATCH}
export MKL_NUM_THREADS=${THREADS_PER_BATCH}
export OPENBLAS_NUM_THREADS=${THREADS_PER_BATCH}
mkdir -p ../5_Z3/logs

echo "batch ${i}: g in [${gmin}, ${gmax}] (${ng} points), threads=${THREADS_PER_BATCH}, node=\$(hostname)"
echo "cpuset actually assigned by PBS:"; numactl --show 2>/dev/null | grep -E "cpubind|nodebind" || true

"\$PYTHON" -u -m pipeline.g_sweep \\
    --lx ${LX} --ly ${LY} --shape ${SHAPE} \\
    --g-min ${gmin} --g-max ${gmax} --n-g ${ng} \\
    --chis ${CHIS} \\
    --sweeps-ascend ${SWEEPS_ASCEND} --lanczos-tol-ascend ${LANCZOS_TOL_ASCEND} \\
    --sweeps-converge ${SWEEPS_CONVERGE} --sweeps-descend ${SWEEPS_DESCEND} \\
    --entanglement-convergence-threshold ${ENTANGLEMENT_CONVERGENCE_THRESHOLD} \\
    ${ENGINE_FLAG} ${WARM_START_FLAG} \\
    --drive-path ${DRIVE_PATH} \\
    > ../5_Z3/logs/g_sweep_batch${i}.log 2>&1

echo "batch ${i} finished at \$(date)"
EOF
    chmod +x "$script"
    echo "  $script : g in [$gmin, $gmax] ($ng points), ncpus=$CORES_PER_BATCH"
    submit_cmds+=("qsub $script")
done

echo ""
echo "Review the generated scripts in $OUT_DIR/, then submit with:"
printf '%s\n' "${submit_cmds[@]}"
echo ""
echo "(all ${#submit_cmds[@]} jobs are independent -- submit them together;"
echo " each will start as soon as its own ${CORES_PER_BATCH} cores are free"
echo " somewhere, regardless of the others.)"

# Uncomment to auto-submit instead of just printing the commands:
# for cmd in "${submit_cmds[@]}"; do eval "$cmd"; done
