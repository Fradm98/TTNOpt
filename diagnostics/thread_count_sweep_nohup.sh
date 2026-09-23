#!/bin/bash
# thread_count_sweep_nohup.sh
# ------------------------------
# Same empirical thread-count calibration as thread_count_sweep_presto.pbs,
# adapted for a plain nohup-launched machine instead of PBS: no #PBS
# directives, no queueing, just run it directly.
#
# This machine's lscpu shows CPU(s)=8 (KVM-virtualized: Socket(s)=8,
# Core(s) per socket=1 is the hypervisor's fake per-vCPU topology, not the
# real Genoa chip's layout -- this VM is only allocated 8 vCPUs regardless
# of the physical host's real core count) and NUMA node(s)=1 (one flat
# domain, nothing to pin). So THREAD_COUNTS is capped at 8 here and there's
# no NUMA angle to test on this machine specifically.
#
# EDIT PYTHON and DRIVE_PATH below for this machine's actual paths before
# running -- they're placeholders, not copied from presto's layout.
#
# Usage:
#   nohup bash diagnostics/thread_count_sweep_nohup.sh > thread_sweep_driver.log 2>&1 &
#   # then check progress any time with:
#   tail -f <DRIVE_PATH>/logs/summary.txt

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THESE for this machine
# ---------------------------------------------------------------------------
PYTHON="$HOME/.conda/envs/ttn/bin/python"   # <-- confirm this path is right here
DRIVE_PATH="$HOME/projects/5_Z3_thread_sweep"  # <-- isolated, does not touch any real scan

THREAD_COUNTS="1 2 4 8"   # capped at 8 -- this VM only has 8 vCPUs
CHI_TEST=50
CHI_TEST_SWEEPS=3

cd "$(dirname "$0")/.."   # repo root, matching pipeline.g_sweep's expected cwd
mkdir -p "$DRIVE_PATH/logs"

echo "Job started on $(hostname) at $(date)"
echo "lscpu summary:"; lscpu | grep -E "Model name|CPU\(s\)|Socket|NUMA node" || true

# Step 1: build the shared ascend(9,27,50) checkpoint ONCE.
"$PYTHON" -u -m pipeline.g_sweep \
    --lx 5 --ly 9 --shape hexagon \
    --g-min 1.5 --g-max 1.5 --n-g 1 \
    --chis 9 27 "$CHI_TEST" \
    --sweeps-ascend 3 --lanczos-tol-ascend 1e-6 --sweeps-converge "$CHI_TEST_SWEEPS" --sweeps-descend 3 \
    --patched \
    --drive-path "$DRIVE_PATH" \
    > "$DRIVE_PATH/logs/build_checkpoint.log" 2>&1

echo "checkpoint built at $(date)"

# Step 2: for each thread count, reload the chi=CHI_TEST checkpoint (same-chi
# reload trick, see pipeline/g_descend_existing.py's docstring) and time
# CHI_TEST_SWEEPS fresh sweeps, sequentially, one thread count at a time.
: > "$DRIVE_PATH/logs/summary.txt"
for threads in $THREAD_COUNTS; do
    echo ""
    echo "=== threads=$threads ==="
    export OMP_NUM_THREADS=$threads
    export MKL_NUM_THREADS=$threads
    export OPENBLAS_NUM_THREADS=$threads

    log="$DRIVE_PATH/logs/threads_${threads}.log"
    start=$(date +%s)
    "$PYTHON" -u -m pipeline.g_descend_existing \
        --g-values -1.500 \
        --source-chi "$CHI_TEST" --descend-chis "$CHI_TEST" --sweeps-descend "$CHI_TEST_SWEEPS" \
        --lx 5 --ly 9 --shape hexagon \
        --patched \
        --drive-path "$DRIVE_PATH" \
        > "$log" 2>&1
    end=$(date +%s)
    total=$((end - start))
    per_sweep=$(python3 -c "print(f'{$total/$CHI_TEST_SWEEPS:.1f}')" 2>/dev/null || echo "n/a")
    line="threads=$threads: ${total}s total for $CHI_TEST_SWEEPS sweeps (${per_sweep}s/sweep)"
    echo "$line -- see $log"
    echo "$line" >> "$DRIVE_PATH/logs/summary.txt"
done

echo ""
echo "Job finished at $(date)"
echo "Summary:"
cat "$DRIVE_PATH/logs/summary.txt"
echo "DONE"
