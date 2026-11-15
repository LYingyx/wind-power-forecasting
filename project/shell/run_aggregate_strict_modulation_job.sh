#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: $0 PHYSICAL_GPU STAGE GROUP SEED BASE_R2W_CHECKPOINT TRAINING_STAGE [INITIAL_CHECKPOINT]" >&2
    exit 2
fi

GPU=$1
STAGE=$2
GROUP=$3
SEED=$4
BASE_CHECKPOINT=$5
TRAINING_STAGE=$6
INITIAL_CHECKPOINT=${7:-}
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
BASE_CONFIG=${WIND_AGG_CONFIG:-$PROJECT_ROOT/configs/aggregate_cluster_v1.yaml}
EXPERIMENT_ROOT=${WIND_STRICT_ROOT:-$PROJECT_ROOT/artifacts/aggregate_strict_modulation_v1}
OUTPUT_DIR=$EXPERIMENT_ROOT/$STAGE/$GROUP/seed$SEED
CYCLES=${WIND_AGG_CYCLES_PER_BATCH:-12}
ACCUMULATION=${WIND_AGG_GRAD_ACCUMULATION:-1}

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}

mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"
flock -x "$LOCK_FD"

if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_aggregate_run.py" --run-dir "$OUTPUT_DIR"; then
        echo "[strict-job] audited-reuse stage=$STAGE group=$GROUP seed=$SEED"
        exit 0
    fi
    echo "[strict-job] invalid completion marker; rerunning $OUTPUT_DIR" >&2
fi
rm -f "$OUTPUT_DIR/COMPLETE"

BASE_GROUP=${GROUP%_const}
MODE=none
STATE=baseline_latent
USE_REFERENCE=false
case "$BASE_GROUP" in
    a0_ft) MODE=none; STATE=baseline_latent ;;
    b1) MODE=effect_scale; STATE=current ;;
    b2) MODE=effect_scale; STATE=baseline_latent ;;
    c1) MODE=channel_scale; STATE=current ;;
    c2) MODE=channel_scale; STATE=baseline_latent ;;
    d1) MODE=low_rank; STATE=current ;;
    d2) MODE=low_rank; STATE=baseline_latent ;;
    e1) MODE=experts; STATE=baseline_latent ;;
    f1) MODE=ramp; STATE=baseline_latent ;;
    *) echo "Unknown strict modulation group: $GROUP" >&2; exit 2 ;;
esac
if [[ "$GROUP" == *_const ]]; then
    USE_REFERENCE=true
fi
if [[ "$TRAINING_STAGE" == adapter ]]; then
    EPOCHS=${WIND_STRICT_ADAPTER_EPOCHS:-30}
    PATIENCE=${WIND_STRICT_ADAPTER_PATIENCE:-6}
else
    EPOCHS=${WIND_STRICT_JOINT_EPOCHS:-20}
    PATIENCE=${WIND_STRICT_JOINT_PATIENCE:-5}
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "aggregate_strict_modulation_v1/$STAGE/$GROUP/seed$SEED" \
    "seed=$SEED" \
    "model.architecture=aggregate_strict_modulation" \
    "model.weather_route=r2_latent_interpolation" \
    "model.history_visible_steps=12" \
    "model.state_mode=current" \
    "model.correction_conditioning=weather" \
    "model.correction_fusion=blackbox" \
    "model.require_history_checkpoint=false" \
    "model.strict_modulation_mode=$MODE" \
    "model.modulation_state_source=$STATE" \
    "model.modulation_training_stage=$TRAINING_STAGE" \
    "model.modulation_use_reference_state=$USE_REFERENCE" \
    "model.modulation_amplitude=0.25" \
    "model.modulation_low_rank=4" \
    "model.modulation_expert_count=3" \
    "model.modulation_expert_coefficient=0.1" \
    "model.decoder_lr_multiplier=0.1" \
    "training.cycles_per_batch=$CYCLES" \
    "training.early_stopping_patience=$PATIENCE" \
    "training.deterministic.epochs=$EPOCHS" \
    "training.deterministic.learning_rate=0.0002" \
    "training.deterministic.gradient_accumulation_steps=$ACCUMULATION" \
    "training.deterministic.baseline_anchor_loss_weight=0.0" \
    "training.deterministic.ramp_loss_weight=0.0" \
    >"$OUTPUT_DIR/config.log" 2>&1

LIMIT_ARGS=()
if [[ -n ${WIND_AGG_MAX_TRAIN_BATCHES:-} ]]; then
    LIMIT_ARGS+=(--max-train-batches "$WIND_AGG_MAX_TRAIN_BATCHES")
fi
if [[ -n ${WIND_AGG_MAX_VAL_BATCHES:-} ]]; then
    LIMIT_ARGS+=(--max-val-batches "$WIND_AGG_MAX_VAL_BATCHES")
fi
INITIAL_ARGS=()
if [[ -n "$INITIAL_CHECKPOINT" ]]; then
    INITIAL_ARGS+=(--initial-checkpoint "$INITIAL_CHECKPOINT")
fi

echo "[strict-job] start gpu=$GPU stage=$STAGE group=$GROUP seed=$SEED mode=$MODE state=$STATE train_stage=$TRAINING_STAGE cycles=$CYCLES accumulation=$ACCUMULATION"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_aggregate_cluster.py" \
    --config "$OUTPUT_DIR/config.yaml" --output-dir "$OUTPUT_DIR" --device cuda \
    --base-checkpoint "$BASE_CHECKPOINT" "${INITIAL_ARGS[@]}" "${LIMIT_ARGS[@]}" \
    > >(tee -a "$OUTPUT_DIR/run.log") \
    2> >(tee -a "$OUTPUT_DIR/run.log" >&2)

EVAL_LIMIT=()
if [[ -n ${WIND_AGG_MAX_VAL_BATCHES:-} ]]; then
    EVAL_LIMIT+=(--max-batches "$WIND_AGG_MAX_VAL_BATCHES")
fi
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_aggregate_cluster.py" \
    --config "$OUTPUT_DIR/config.yaml" --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --split val --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" --device cuda \
    "${EVAL_LIMIT[@]}" \
    > >(tee -a "$OUTPUT_DIR/run.log") \
    2> >(tee -a "$OUTPUT_DIR/run.log" >&2)

AUDIT_ARGS=()
[[ -z ${WIND_AGG_MAX_VAL_BATCHES:-} ]] || AUDIT_ARGS+=(--allow-partial)
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_aggregate_run.py" \
    --run-dir "$OUTPUT_DIR" "${AUDIT_ARGS[@]}"
touch "$OUTPUT_DIR/COMPLETE"
echo "[strict-job] complete gpu=$GPU stage=$STAGE group=$GROUP seed=$SEED"
