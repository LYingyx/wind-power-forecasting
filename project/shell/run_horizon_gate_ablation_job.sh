#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 GPU GROUP SEED" >&2
    exit 2
fi

GPU=$1
GROUP=$2
SEED=$3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_HORIZON_CONTROL_ROOT:-${PROJECT_ROOT}/artifacts/horizon_control_matrix_v1}"
RUN_DIR="${EXPERIMENT_ROOT}/groups/${GROUP}/seed${SEED}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

if [[ ! -f "$RUN_DIR/COMPLETE" ]]; then
    echo "Formal run is incomplete: ${RUN_DIR}" >&2
    exit 3
fi
if [[ -f "$RUN_DIR/GATE_ABLATION_COMPLETE" ]]; then
    echo "[horizon-gate-ablation-v1] reuse group=$GROUP seed=$SEED"
    exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$RUN_DIR/config.yaml" \
    --checkpoint "$RUN_DIR/best_model.pt" \
    --split val \
    --output "$RUN_DIR/metrics_gate_zero_val.json" \
    --predictions-output "$RUN_DIR/predictions_gate_zero_val.npz" \
    --zero-horizon-gate \
    --device cuda \
    >"$RUN_DIR/gate_zero.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$RUN_DIR/config.yaml" \
    --checkpoint "$RUN_DIR/best_model.pt" \
    --split val \
    --output "$RUN_DIR/metrics_gate_one_val.json" \
    --predictions-output "$RUN_DIR/predictions_gate_one_val.npz" \
    --unit-horizon-gate \
    --device cuda \
    >"$RUN_DIR/gate_one.log" 2>&1

touch "$RUN_DIR/GATE_ABLATION_COMPLETE"
echo "[horizon-gate-ablation-v1] complete group=$GROUP seed=$SEED"
