#!/bin/bash
# setup_ngt.sh
# --------------
# One-time setup of the TTNOpt repo + a working conda env on the ngt
# cluster, so diagnostics/thread_count_sweep_nohup.sh (and anything else in
# pipeline/) can actually run there. Run this directly on ngt (not via
# nohup -- it's interactive/quick, mostly network + conda I/O, not
# compute).
#
# Usage:
#   bash setup_ngt.sh
# or step through it by hand if you'd rather verify each stage.

set -euo pipefail

REPO_DIR="$HOME/TTNOpt"          # <-- adjust if you want it somewhere else
ENV_NAME="ttn"
PY_VERSION="3.14"

echo "=== 1. Clone the repo ==="
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already exists at $REPO_DIR, pulling latest instead."
    git -C "$REPO_DIR" pull
else
    # HTTPS avoids needing SSH keys set up on ngt for GitHub. If this repo
    # is private and HTTPS prompts for auth you don't have, use the SSH
    # form instead (requires an SSH key added to your GitHub account):
    #   git clone git@github.com:Fradm98/TTNOpt.git "$REPO_DIR"
    git clone https://github.com/Fradm98/TTNOpt.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo ""
echo "=== 2. Create the conda env ($ENV_NAME, python $PY_VERSION) ==="
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Env '$ENV_NAME' already exists, skipping creation (delete it first with"
    echo "'conda env remove -n $ENV_NAME' if you want a clean rebuild)."
else
    conda create -n "$ENV_NAME" -y python="$PY_VERSION" numpy scipy
fi

ENV_PYTHON="$(conda run -n "$ENV_NAME" which python)"
echo "env python: $ENV_PYTHON"

echo ""
echo "=== 3. Install the rest of the dependencies ==="
"$ENV_PYTHON" -m pip install -q \
    contourpy cycler dotmap fonttools graphviz h5py ImageIO kiwisolver \
    matplotlib networkx opt_einsum pandas pillow pydot pyparsing \
    python-dateutil PyYAML six tensornetwork threadpoolctl tqdm

echo ""
echo "=== 4. Install ttnopt itself (editable) ==="
"$ENV_PYTHON" -m pip install -q -e .

echo ""
echo "=== 5. Verify ==="
"$ENV_PYTHON" -c "
import numpy, scipy, ttnopt
from ttnopt.src.TTNLinearOperator import ttn_eigensolver
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('ttnopt + patched linop import: OK')
"
GSS_PATH="$(dirname "$ENV_PYTHON")/gss"
if [ -x "$GSS_PATH" ]; then
    echo "gss executable: $GSS_PATH -- OK"
else
    echo "WARNING: gss not found at $GSS_PATH -- check the pip install -e . output above for errors"
fi

echo ""
echo "=== Done ==="
echo "Repo:  $REPO_DIR"
echo "Env:   conda activate $ENV_NAME   (or use $ENV_PYTHON directly, matching this project's convention)"
echo ""
echo "Before running diagnostics/thread_count_sweep_nohup.sh, edit its PYTHON"
echo "and DRIVE_PATH variables to:"
echo "  PYTHON=\"$ENV_PYTHON\""
echo "  DRIVE_PATH=\"/eos/user/f/fdimarca/projects/5_Z3_thread_sweep\"   # matches pipeline.g_sweep's DEVICE_DRIVE_PATHS['ngt']"
