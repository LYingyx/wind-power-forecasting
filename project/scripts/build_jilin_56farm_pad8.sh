#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
WORKSPACE_ROOT=$(CDPATH= cd -- "$PROJECT_ROOT/.." && pwd)
SOURCE_PROJECT="$WORKSPACE_ROOT/roa_c2f_wind_forecasting"
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
CONFIG="$PROJECT_ROOT/configs/jilin_56farm_pad8.yaml"
OUTPUT="$PROJECT_ROOT/data/jilin_56farm_pad8.h5"

export PYTHONPATH="$SOURCE_PROJECT/src"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

test -x "$PYTHON_BIN" || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }

mkdir -p "$PROJECT_ROOT/data"

echo "[audit] config=$CONFIG"
"$PYTHON_BIN" "$SOURCE_PROJECT/scripts/build_jilin_cluster_dataset.py" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --audit-only

echo "[build] output=$OUTPUT"
"$PYTHON_BIN" "$SOURCE_PROJECT/scripts/build_jilin_cluster_dataset.py" \
    --config "$CONFIG" \
    --output "$OUTPUT"

"$PYTHON_BIN" "$SOURCE_PROJECT/scripts/inspect_jilin_cluster_dataset.py" \
    --dataset "$OUTPUT" \
    --split train \
    --sample-index 0 \
    --include-stations

echo "[done] $OUTPUT"
