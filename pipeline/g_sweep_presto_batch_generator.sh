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
N_BATCHES=6                  # how many independent small jobs to generate
CORES_PER_BATCH=16           # ncpus requested per job -- keep modest so it
                              # fits into partial free capacity easily
THREADS_PER_BATCH=4           # BLAS/OMP thread count per job (separate axis
                              # from CORES_PER_BATCH -- see g_sweep_presto_
                              # parallel.pbs's comment on why more threads
                              # isn't automatically faster)
WALLTIME="48:00:00"
MEM_PER_BATCH="40gb"

G_MIN=0.1
G_MAX=1.5
N_G_TOTAL=15
CHIS="9 18 27"
SWEEPS_ASCEND=3
SWEEPS_CONVERGE=10
SWEEPS_DESCEND=10
LX=5
LY=5
SHAPE=parallelogram
ENGINE_FLAG="--patched"
WARM_START_FLAG=""           # "--warm-start-g" to warm-start within each batch's own slice

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
    --sweeps-ascend ${SWEEPS_ASCEND} --sweeps-converge ${SWEEPS_CONVERGE} --sweeps-descend ${SWEEPS_DESCEND} \\
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
