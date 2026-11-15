#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 PHYSICAL_GPU LABEL GROUP_ROOT MODEL_SEED PERMUTATION_SEED SPLIT" >&2
    exit 2
fi
GPU=$1; LABEL=$2; GROUP_ROOT=$3; MODEL_SEED=$4; PERMUTATION_SEED=$5
SPLIT=$6
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
OUTPUT_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-${PROJECT_ROOT}/artifacts/paper_evidence_test_v1}"
RUN_DIR="${GROUP_ROOT}/seed${MODEL_SEED}"
OUTPUT_DIR="${OUTPUT_ROOT}/weather_robustness/${LABEL}/seed${MODEL_SEED}/perm${PERMUTATION_SEED}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"; flock -x "$LOCK_FD"
MARKER="$OUTPUT_DIR/${SPLIT^^}_COMPLETE"
if [[ -f "$MARKER" ]]; then echo "[weather-robustness] reuse $LABEL seed=$MODEL_SEED perm=$PERMUTATION_SEED split=$SPLIT"; exit 0; fi
if [[ "${WIND_TQDM_VISIBLE:-1}" == "1" ]]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
        --config "$RUN_DIR/config.yaml" --checkpoint "$RUN_DIR/best_model.pt" \
        --split "$SPLIT" --output "$OUTPUT_DIR/metrics_${SPLIT}.json" \
        --predictions-output "$OUTPUT_DIR/predictions_${SPLIT}.npz" \
        --weather-permutation-seed "$PERMUTATION_SEED" --device cuda \
        >"$OUTPUT_DIR/run.log" 2> >(tee -a "$OUTPUT_DIR/run.log" >&2)
else
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
        --config "$RUN_DIR/config.yaml" --checkpoint "$RUN_DIR/best_model.pt" \
        --split "$SPLIT" --output "$OUTPUT_DIR/metrics_${SPLIT}.json" \
        --predictions-output "$OUTPUT_DIR/predictions_${SPLIT}.npz" \
        --weather-permutation-seed "$PERMUTATION_SEED" --device cuda \
        >"$OUTPUT_DIR/run.log" 2>&1
fi
touch "$MARKER"
echo "[weather-robustness] complete physical_gpu=$GPU label=$LABEL seed=$MODEL_SEED perm=$PERMUTATION_SEED split=$SPLIT"
