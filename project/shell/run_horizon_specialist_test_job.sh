#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then echo "usage: $0 PHYSICAL_GPU GROUP SEED" >&2; exit 2; fi
GPU=$1 GROUP=$2 SEED=$3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_HS_ROOT:-${PROJECT_ROOT}/artifacts/horizon_specialist_v1}"
RUN="$EXPERIMENT_ROOT/groups/$GROUP/seed$SEED"
[[ -s "$RUN/config.yaml" && -s "$RUN/best_model.pt" ]] || {
    echo "Missing frozen run for test: $RUN" >&2; exit 3;
}
if [[ -f "$RUN/TEST_COMPLETE" && -s "$RUN/metrics_test.json" && -s "$RUN/predictions_test.npz" ]]; then
    echo "[horizon-specialist-test] reuse group=$GROUP seed=$SEED"
    exit 0
fi
rm -f "$RUN/TEST_COMPLETE"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$RUN/config.yaml" --checkpoint "$RUN/best_model.pt" --split test \
    --output "$RUN/metrics_test.json" --predictions-output "$RUN/predictions_test.npz" \
    --device cuda 2> >(tee -a "$RUN/test.log" >&2) | tee -a "$RUN/test.log"
touch "$RUN/TEST_COMPLETE"
echo "[horizon-specialist-test] complete gpu=$GPU group=$GROUP seed=$SEED"
