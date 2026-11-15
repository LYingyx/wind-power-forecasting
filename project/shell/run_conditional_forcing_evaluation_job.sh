#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "usage: $0 GPU KIND LABEL SEED CONFIG CHECKPOINT [PERMUTATION_SEED]" >&2
    exit 2
fi

GPU=$1
KIND=$2
LABEL=$3
SEED=$4
CONFIG=$5
CHECKPOINT=$6
PERMUTATION_SEED=${7:-}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_CONDITIONAL_VALIDATION_ROOT:-${PROJECT_ROOT}/artifacts/conditional_forcing_validation_v1}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

case "$KIND" in
    component)
        OUTPUT_DIR="${EXPERIMENT_ROOT}/component_evaluations/${LABEL}/seed${SEED}"
        EXTRA_ARGS=()
        ;;
    shuffle)
        if [[ -z "$PERMUTATION_SEED" ]]; then
            echo "shuffle evaluation requires PERMUTATION_SEED" >&2
            exit 2
        fi
        OUTPUT_DIR="${EXPERIMENT_ROOT}/weather_shuffle/perm${PERMUTATION_SEED}/seed${SEED}"
        EXTRA_ARGS=(--weather-permutation-seed "$PERMUTATION_SEED" --weather-permutation-power-bins 10)
        ;;
    *)
        echo "Unsupported evaluation kind: $KIND" >&2
        exit 2
        ;;
esac

mkdir -p "$OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    echo "[conditional-v1] reuse evaluation kind=$KIND label=$LABEL seed=$SEED"
    exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split val \
    --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" \
    --device cuda \
    "${EXTRA_ARGS[@]}" \
    >"$OUTPUT_DIR/run.log" 2>&1

touch "$OUTPUT_DIR/COMPLETE"
echo "[conditional-v1] complete evaluation kind=$KIND label=$LABEL seed=$SEED permutation=$PERMUTATION_SEED"
