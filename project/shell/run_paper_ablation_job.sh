#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 PHYSICAL_GPU GROUP SEED TEACHER_ARCHIVE_OR_NONE [CONFIG_OVERRIDE ...]" >&2
    exit 2
fi

GPU=$1
GROUP=$2
SEED=$3
TEACHER_ARCHIVE=$4
shift 4

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_PAPER_ABLATION_ROOT:-${PROJECT_ROOT}/artifacts/paper_ablation_completion_v1}"
BASE_CONFIG="${WIND_PAPER_ABLATION_CONFIG:-${PROJECT_ROOT}/configs/incremental_correction_pad5.yaml}"
EPOCHS="${WIND_PAPER_ABLATION_EPOCHS:-80}"
MAX_TRAIN_BATCHES="${WIND_PAPER_ABLATION_MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${WIND_PAPER_ABLATION_MAX_VAL_BATCHES:-}"
LEGACY_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only"
NEW_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/incremental_mechanism_v2/history_prior"
OUTPUT_DIR="${EXPERIMENT_ROOT}/groups/${GROUP}/seed${SEED}"

if [[ -f "${LEGACY_HISTORY_ROOT}/seed${SEED}/best_model.pt" ]]; then
    HISTORY_CHECKPOINT="${LEGACY_HISTORY_ROOT}/seed${SEED}/best_model.pt"
else
    HISTORY_CHECKPOINT="${NEW_HISTORY_ROOT}/seed${SEED}/best_model.pt"
fi
if [[ ! -f "$HISTORY_CHECKPOINT" ]]; then
    echo "Missing history checkpoint for seed ${SEED}: ${HISTORY_CHECKPOINT}" >&2
    exit 3
fi
if [[ "$TEACHER_ARCHIVE" != "NONE" && ! -f "$TEACHER_ARCHIVE" ]]; then
    echo "Missing teacher archive: $TEACHER_ARCHIVE" >&2
    exit 3
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"

run_logged() {
    local mode=$1
    local log_file=$2
    shift 2
    if [[ "${WIND_TQDM_VISIBLE:-1}" == "1" ]]; then
        if [[ "$mode" == "truncate" ]]; then
            "$@" >"$log_file" 2> >(tee -a "$log_file" >&2)
        else
            "$@" >>"$log_file" 2> >(tee -a "$log_file" >&2)
        fi
    elif [[ "$mode" == "truncate" ]]; then
        "$@" >"$log_file" 2>&1
    else
        "$@" >>"$log_file" 2>&1
    fi
}

mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"
flock -x "$LOCK_FD"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    echo "[paper-ablation] reuse group=$GROUP seed=$SEED"
    exit 0
fi

DISTILLATION_WEIGHT=0.0
if [[ "$TEACHER_ARCHIVE" != "NONE" ]]; then
    DISTILLATION_WEIGHT=0.5
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" \
    --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "${GROUP}/seed${SEED}" \
    "seed=${SEED}" \
    "model.freeze_history_baseline=false" \
    "model.initialize_history_baseline_from_checkpoint=true" \
    "model.correction_conditioning=history_weather" \
    "model.correction_history_mode=cluster_recent_1h" \
    "model.cross_farm_context_mode=none" \
    "model.correction_fusion_mode=blackbox" \
    "model.matched_conditioning_backbone=true" \
    "model.baseline_anchor_enabled=true" \
    "model.projection_mode=full" \
    "model.temporal_components=24" \
    "training.deterministic.baseline_auxiliary_loss_weight=0.0" \
    "training.deterministic.baseline_anchor_loss_weight=0.2" \
    "training.deterministic.distillation_weight=${DISTILLATION_WEIGHT}" \
    "training.deterministic.distillation_station_loss_weight=1.0" \
    "training.deterministic.distillation_cluster_loss_weight=2.0" \
    "training.deterministic.distillation_uncertainty_aware=false" \
    "training.deterministic.distillation_station_scale=0.0" \
    "training.deterministic.distillation_cluster_scale=0.0" \
    "training.deterministic.ema_decay=0.0" \
    "training.deterministic.checkpoint_average_window=0" \
    "$@" \
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
TEACHER_ARGUMENT=()
if [[ "$TEACHER_ARCHIVE" != "NONE" ]]; then
    TEACHER_ARGUMENT+=(--teacher-archive "$TEACHER_ARCHIVE")
fi

run_logged truncate "$OUTPUT_DIR/run.log" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/train_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --epochs "$EPOCHS" \
    --history-checkpoint "$HISTORY_CHECKPOINT" \
    "${TEACHER_ARGUMENT[@]}" \
    "${TRAIN_LIMIT[@]}"

run_logged append "$OUTPUT_DIR/run.log" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" \
    --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --split val \
    --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" \
    --device cuda \
    "${EVAL_LIMIT[@]}"

touch "$OUTPUT_DIR/COMPLETE"
echo "[paper-ablation] complete gpu=$GPU group=$GROUP seed=$SEED"
