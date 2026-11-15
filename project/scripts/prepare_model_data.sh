#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}

export PYTHONPATH="$PROJECT_ROOT/src"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/prepare_model_data.py \
  --config configs/forced_evolution_pad5.yaml
