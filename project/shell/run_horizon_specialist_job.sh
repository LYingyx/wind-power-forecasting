#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 PHYSICAL_GPU GROUP SEED [TEACHER_ARCHIVE]" >&2
    exit 2
fi

GPU=$1
GROUP=$2
SEED=$3
TEACHER_ARCHIVE="${4:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_HS_ROOT:-${PROJECT_ROOT}/artifacts/horizon_specialist_v1}"
BASE_CONFIG="${WIND_HS_CONFIG:-${PROJECT_ROOT}/configs/incremental_correction_pad5.yaml}"
EPOCHS="${WIND_HS_EPOCHS:-80}"
MAX_TRAIN_BATCHES="${WIND_HS_MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${WIND_HS_MAX_VAL_BATCHES:-}"
LEGACY_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only"
NEW_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/incremental_mechanism_v2/history_prior"
OUTPUT_DIR="${EXPERIMENT_ROOT}/groups/${GROUP}/seed${SEED}"

BASE_GROUP=${GROUP#kd_}
case "$BASE_GROUP" in
    l1_anchor_taper)
        OVERRIDES=(
            model.weather_view_mode=full model.weather_gate_mode=fixed
            model.specialist_mode=shared_head
            training.deterministic.anchor_schedule_mode=horizon_taper
        ) ;;
    l2_fixed_view_scalar)
        OVERRIDES=(
            model.weather_view_mode=horizon_mix model.weather_gate_mode=fixed
            model.specialist_mode=shared_head
            training.deterministic.anchor_schedule_mode=scalar
        ) ;;
    l3_fixed_view_taper)
        OVERRIDES=(
            model.weather_view_mode=horizon_mix model.weather_gate_mode=fixed
            model.specialist_mode=shared_head
            training.deterministic.anchor_schedule_mode=horizon_taper
        ) ;;
    l4_learned_view_taper)
        OVERRIDES=(
            model.weather_view_mode=horizon_mix model.weather_gate_mode=monotonic_sigmoid
            model.specialist_mode=shared_head
            training.deterministic.anchor_schedule_mode=horizon_taper
        ) ;;
    h0_dual_full_taper)
        OVERRIDES=(
            model.weather_view_mode=full model.weather_gate_mode=fixed
            model.specialist_mode=dual_head
            training.deterministic.anchor_schedule_mode=horizon_taper
        ) ;;
    h1_dual_mixed_taper)
        OVERRIDES=(
            model.weather_view_mode=horizon_mix model.weather_gate_mode=fixed
            model.specialist_mode=dual_head
            training.deterministic.anchor_schedule_mode=horizon_taper
        ) ;;
    h2_dual_learned_taper)
        OVERRIDES=(
            model.weather_view_mode=horizon_mix model.weather_gate_mode=monotonic_sigmoid
            model.specialist_mode=dual_head
            training.deterministic.anchor_schedule_mode=horizon_taper
        ) ;;
    *) echo "Unknown horizon-specialist group: $BASE_GROUP" >&2; exit 2 ;;
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
if [[ -n "$TEACHER_ARCHIVE" && ! -f "$TEACHER_ARCHIVE" ]]; then
    echo "Missing teacher archive: $TEACHER_ARCHIVE" >&2
    exit 3
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"

run_logged() {
    local mode=$1 log_file=$2
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

run_complete() {
    [[ -f "$OUTPUT_DIR/COMPLETE" && -s "$OUTPUT_DIR/best_model.pt" \
        && -s "$OUTPUT_DIR/metrics_val.json" && -s "$OUTPUT_DIR/predictions_val.npz" ]]
}

mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"
flock -x "$LOCK_FD"
if run_complete; then
    echo "[horizon-specialist] reuse group=$GROUP seed=$SEED"
    exit 0
fi
rm -f "$OUTPUT_DIR/COMPLETE"

DISTILLATION_WEIGHT=0.0
TEACHER_ARGUMENT=()
if [[ -n "$TEACHER_ARCHIVE" ]]; then
    DISTILLATION_WEIGHT=0.5
    TEACHER_ARGUMENT+=(--teacher-archive "$TEACHER_ARCHIVE")
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" \
    --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "horizon_specialist_v1/${GROUP}/seed${SEED}" \
    "seed=${SEED}" \
    model.freeze_history_baseline=false \
    model.initialize_history_baseline_from_checkpoint=true \
    model.correction_conditioning=history_weather \
    model.correction_history_mode=cluster_recent_1h \
    model.cross_farm_context_mode=none \
    model.correction_fusion_mode=blackbox \
    model.matched_conditioning_backbone=true \
    model.baseline_anchor_enabled=true \
    model.projection_mode=full \
    model.temporal_components=24 \
    training.deterministic.baseline_auxiliary_loss_weight=0.0 \
    training.deterministic.baseline_anchor_loss_weight=0.2 \
    "training.deterministic.distillation_weight=${DISTILLATION_WEIGHT}" \
    "${OVERRIDES[@]}" >"$OUTPUT_DIR/config.log" 2>&1

TRAIN_LIMIT=()
EVAL_LIMIT=()
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
    TRAIN_LIMIT+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then
    TRAIN_LIMIT+=(--max-val-batches "$MAX_VAL_BATCHES")
    EVAL_LIMIT+=(--max-batches "$MAX_VAL_BATCHES")
fi

run_logged truncate "$OUTPUT_DIR/run.log" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/train_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" --output-dir "$OUTPUT_DIR" \
    --device cuda --epochs "$EPOCHS" --history-checkpoint "$HISTORY_CHECKPOINT" \
    "${TEACHER_ARGUMENT[@]}" \
    "${TRAIN_LIMIT[@]}"
run_logged append "$OUTPUT_DIR/run.log" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --split val --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" --device cuda \
    "${EVAL_LIMIT[@]}"

touch "$OUTPUT_DIR/COMPLETE"
echo "[horizon-specialist] complete gpu=$GPU group=$GROUP seed=$SEED"
