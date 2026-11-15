#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 PHYSICAL_GPU_CSV [SLOTS_SPEC]" >&2
    echo "examples:" >&2
    echo "  $0 0,1,2,3,4,5,6,7 auto" >&2
    echo "  $0 0,1 2" >&2
    echo "  $0 0,1,4 3,3,2" >&2
    exit 2
fi
SLOTS_SPEC=${2:-1}
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/prepare_aggregate_cluster_data.py \
    --source-cache data/model_cache_pad5_f8 --output data/aggregate_cluster_v1 \
    --verify-checksums
exec "$PYTHON_BIN" scripts/launch_aggregate_cluster_experiments.py \
    --gpus "$1" --slots "$SLOTS_SPEC"
