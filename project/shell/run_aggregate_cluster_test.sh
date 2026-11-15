#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 PHYSICAL_GPU RUN_DIR" >&2
    exit 2
fi
GPU=$1
RUN_DIR=$(realpath "$2")
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}

exec {LOCK_FD}>"$RUN_DIR/.test.lock"
flock -x "$LOCK_FD"
if [[ -f "$RUN_DIR/TEST_COMPLETE" ]]; then
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_aggregate_run.py" \
        --run-dir "$RUN_DIR" --split test; then
        echo "[aggregate-test] audited-reuse run=$RUN_DIR"
        exit 0
    fi
fi
rm -f "$RUN_DIR/TEST_COMPLETE"
echo "[aggregate-test] start gpu=$GPU run=$RUN_DIR"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_aggregate_cluster.py" \
    --config "$RUN_DIR/config.yaml" --checkpoint "$RUN_DIR/best_model.pt" \
    --split test --output "$RUN_DIR/metrics_test.json" \
    --predictions-output "$RUN_DIR/predictions_test.npz" --device cuda \
    > >(tee -a "$RUN_DIR/test.log") \
    2> >(tee -a "$RUN_DIR/test.log" >&2)
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_aggregate_run.py" \
    --run-dir "$RUN_DIR" --split test
touch "$RUN_DIR/TEST_COMPLETE"
echo "[aggregate-test] complete gpu=$GPU run=$RUN_DIR"
