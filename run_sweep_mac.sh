#!/bin/bash
# Wraps sweep.py with `caffeinate` to prevent macOS from sleeping during a
# long unattended run. Testing hypothesis: the "Bad file descriptor" /
# init_sys_streams crashes are caused by the Mac sleeping mid-run (the pty
# backing the parent's stdin/stdout/stderr can end up in a broken state on
# wake, which a freshly-spawned child's Python startup can't wrap) -- NOT a
# file-descriptor leak, since the fd-check instrumentation showed a flat,
# tiny open-fd count right up to the moment of the crash.
#
# Usage:
#   ./run_sweep_mac.sh path/to/output.log
# (defaults to a timestamped log under ../projects/5_Z3/logs/ if omitted)

set -euo pipefail

LOG_PATH="${1:-../projects/5_Z3/logs/sweep_caffeinated_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_PATH")"

echo "Running under caffeinate (macOS sleep disabled for this process tree)."
echo "Logging to: $LOG_PATH"

caffeinate -i python -u sweep.py > "$LOG_PATH" 2>&1 &
PID=$!
echo "Launched (PID $PID). tail -f \"$LOG_PATH\" to watch it."
wait "$PID"
