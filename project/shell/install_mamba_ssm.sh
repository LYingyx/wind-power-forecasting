#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
DEFAULT_WHEEL=https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
WHEEL_SOURCE=${1:-${MAMBA_WHEEL:-$DEFAULT_WHEEL}}

if "$PYTHON_BIN" -c 'from mamba_ssm import Mamba, Mamba2' 2>/dev/null; then
    "$PYTHON_BIN" -c 'import mamba_ssm; print(f"mamba-ssm already available: {mamba_ssm.__version__}")'
    exit 0
fi

echo "[mamba-install] source=$WHEEL_SOURCE"
"$PYTHON_BIN" -m pip install --no-deps "$WHEEL_SOURCE"
"$PYTHON_BIN" - <<'PY'
import torch
import mamba_ssm
from mamba_ssm import Mamba, Mamba2
print(f"[mamba-install] mamba_ssm={mamba_ssm.__version__} torch={torch.__version__}")
PY
