#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 GPU GROUP SEED HISTORY_MODE TEACHER_ARCHIVE TEACHER_LABEL" >&2
    exit 2
fi

GPU=$1
GROUP=$2
SEED=$3
HISTORY_MODE=$4
TEACHER_ARCHIVE=$5
TEACHER_LABEL=$6

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_CONSENSUS_ROOT:-${PROJECT_ROOT}/artifacts/consensus_controls_v1}"
EPOCHS="${WIND_CONSENSUS_EPOCHS:-80}"
BASE_CONFIG="${PROJECT_ROOT}/configs/incremental_correction_pad5.yaml"
LEGACY_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only"
NEW_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/incremental_mechanism_v2/history_prior"
OUTPUT_DIR="${EXPERIMENT_ROOT}/groups/${GROUP}/seed${SEED}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

mkdir -p "$OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    echo "[consensus-control] reuse group=$GROUP seed=$SEED"
    exit 0
fi
if [[ ! -f "$TEACHER_ARCHIVE" ]]; then
    echo "Missing teacher archive: $TEACHER_ARCHIVE" >&2
    exit 3
fi
case "$HISTORY_MODE" in
    cluster_recent_1h|full_history) ;;
    *)
        echo "HISTORY_MODE must be cluster_recent_1h or full_history" >&2
        exit 2
        ;;
esac

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
    "model.correction_history_mode=${HISTORY_MODE}" \
    "model.matched_conditioning_backbone=true" \
    "model.baseline_anchor_enabled=true" \
    "model.projection_mode=full" \
    "model.temporal_components=24" \
    "training.deterministic.baseline_auxiliary_loss_weight=0.0" \
    "training.deterministic.baseline_anchor_loss_weight=0.2" \
    "training.deterministic.distillation_weight=0.5" \
    "training.deterministic.distillation_uncertainty_aware=false" \
    "training.deterministic.distillation_station_scale=0.0" \
    "training.deterministic.distillation_cluster_scale=0.0" \
    "training.deterministic.ema_decay=0.0" \
    "training.deterministic.checkpoint_average_window=0" \
    >"$OUTPUT_DIR/config.log" 2>&1

printf '%s\n' \
    "teacher_label=${TEACHER_LABEL}" \
    "teacher_archive=$(realpath "$TEACHER_ARCHIVE")" \
    "history_mode=${HISTORY_MODE}" \
    "student_seed=${SEED}" \
    >"$OUTPUT_DIR/teacher_source.txt"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/train_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --epochs "$EPOCHS" \
    --history-checkpoint "$HISTORY_CHECKPOINT" \
    --teacher-archive "$TEACHER_ARCHIVE" \
    >"$OUTPUT_DIR/run.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" \
    --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --split val \
    --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" \
    --device cuda \
    >>"$OUTPUT_DIR/run.log" 2>&1

touch "$OUTPUT_DIR/COMPLETE"
echo "[consensus-control] complete group=$GROUP seed=$SEED teacher=$TEACHER_LABEL"
