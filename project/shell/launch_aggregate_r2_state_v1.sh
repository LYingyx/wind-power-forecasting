#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 PHYSICAL_GPU_CSV [SLOTS_SPEC]" >&2
    echo "example: $0 0,1,2,3,4,5,6,7 auto" >&2
    exit 2
fi

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" scripts/launch_aggregate_r2_state_experiments.py \
    --gpus "$1" --slots "${2:-auto}"
