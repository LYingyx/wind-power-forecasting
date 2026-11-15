#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 GPU GROUP SEED CONTEXT_MODE" >&2
    exit 2
fi

GPU=$1
GROUP=$2
SEED=$3
CONTEXT_MODE=$4

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_CROSS_FARM_ROOT:-${PROJECT_ROOT}/artifacts/cross_farm_context_v1}"
EPOCHS="${WIND_CROSS_FARM_EPOCHS:-80}"
MAX_TRAIN_BATCHES="${WIND_CROSS_FARM_MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${WIND_CROSS_FARM_MAX_VAL_BATCHES:-}"
BASE_CONFIG="${PROJECT_ROOT}/configs/incremental_correction_pad5.yaml"
LEGACY_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only"
NEW_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/incremental_mechanism_v2/history_prior"
OUTPUT_DIR="${EXPERIMENT_ROOT}/groups/${GROUP}/seed${SEED}"

case "$CONTEXT_MODE" in
    self_duplicate|geographic|global_other|weather_coherent) ;;
    *)
        echo "Unsupported cross-farm context mode: $CONTEXT_MODE" >&2
        exit 2
        ;;
esac

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"
flock -x "$LOCK_FD"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    echo "[cross-farm-v1] reuse group=$GROUP seed=$SEED"
    exit 0
fi

if [[ -f "${LEGACY_HISTORY_ROOT}/seed${SEED}/best_model.pt" ]]; then
    HISTORY_CHECKPOINT="${LEGACY_HISTORY_ROOT}/seed${SEED}/best_model.pt"
else
    HISTORY_CHECKPOINT="${NEW_HISTORY_ROOT}/seed${SEED}/best_model.pt"
fi
if [[ ! -f "$HISTORY_CHECKPOINT" ]]; then
    echo "Missing history checkpoint for seed ${SEED}: ${HISTORY_CHECKPOINT}" >&2
    exit 3
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" \
    --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "${GROUP}/seed${SEED}" \
    "seed=${SEED}" \
    "model.freeze_history_baseline=false" \
    "model.initialize_history_baseline_from_checkpoint=true" \
    "model.correction_conditioning=history_weather" \
    "model.correction_history_mode=recent_1h" \
    "model.cross_farm_context_mode=${CONTEXT_MODE}" \
    "model.cross_farm_neighbors=4" \
    "model.correction_fusion_mode=blackbox" \
    "model.matched_conditioning_backbone=true" \
    "model.baseline_anchor_enabled=true" \
    "model.projection_mode=full" \
    "model.temporal_components=24" \
    "training.deterministic.baseline_auxiliary_loss_weight=0.0" \
    "training.deterministic.baseline_anchor_loss_weight=0.2" \
    "training.deterministic.distillation_weight=0.0" \
    "training.deterministic.ema_decay=0.999" \
    "training.deterministic.checkpoint_average_window=5" \
    >"$OUTPUT_DIR/config.log" 2>&1

TRAIN_LIMIT=()
EVAL_LIMIT=()
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
    TRAIN_LIMIT+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then
    TRAIN_LIMIT+=(--max-val-batches "$MAX_VAL_BATCHES")
    EVAL_LIMIT+=(--max-batches "$MAX_VAL_BATCHES")
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/train_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --epochs "$EPOCHS" \
    --history-checkpoint "$HISTORY_CHECKPOINT" \
    "${TRAIN_LIMIT[@]}" \
    >"$OUTPUT_DIR/run.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" \
    --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --model-state-key model_state \
    --split val \
    --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" \
    --device cuda \
    "${EVAL_LIMIT[@]}" \
    >>"$OUTPUT_DIR/run.log" 2>&1

touch "$OUTPUT_DIR/COMPLETE"
echo "[cross-farm-v1] complete group=$GROUP seed=$SEED mode=$CONTEXT_MODE"
