#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
    echo "usage: $0 GPU STAGE GROUP SEED BASE_R2_CHECKPOINT TRAINING_STAGE [INITIAL_CHECKPOINT] [AUXILIARY_CHECKPOINT]" >&2
    exit 2
fi

GPU=$1
STAGE=$2
GROUP=$3
SEED=$4
BASE_CHECKPOINT=$5
TRAINING_STAGE=$6
INITIAL_CHECKPOINT=${7:-}
AUXILIARY_CHECKPOINT=${8:-}
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
BASE_CONFIG=${WIND_AGG_CONFIG:-$PROJECT_ROOT/configs/aggregate_cluster_v1.yaml}
EXPERIMENT_ROOT=${WIND_DYNAMICS_ROOT:-$PROJECT_ROOT/artifacts/aggregate_weather_dynamics_v1}
DYNAMICS_LATENT_ROOT=${WIND_DYNAMICS_LATENT_ROOT:-$PROJECT_ROOT/artifacts/aggregate_weather_dynamics_v1/latent_cache}
REALIZATION_CACHE_ROOT=${WIND_REALIZATION_CACHE_ROOT:-$PROJECT_ROOT/artifacts/aggregate_weather_dynamics_v1/realization_cache}
OUTPUT_DIR=$EXPERIMENT_ROOT/$STAGE/$GROUP/seed$SEED
CYCLES=${WIND_AGG_CYCLES_PER_BATCH:-6}
ACCUMULATION=${WIND_AGG_GRAD_ACCUMULATION:-2}

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3} MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}

mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"
flock -x "$LOCK_FD"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_aggregate_run.py" --run-dir "$OUTPUT_DIR"; then
        echo "[dynamics-job] audited-reuse stage=$STAGE group=$GROUP seed=$SEED"
        exit 0
    fi
fi
rm -f "$OUTPUT_DIR/COMPLETE"

MODE=
QUANTILE=0.0
CURRENT_WEIGHT=0.0
REALIZATION_ROOT=null
case "$GROUP" in
    weather_mean) MODE=weather_mean; CURRENT_WEIGHT=0.041666666666666664 ;;
    weather_envelope|weather_envelope_fold*) MODE=weather_envelope; QUANTILE=0.9; CURRENT_WEIGHT=0.041666666666666664 ;;
    m1) MODE=m1_baseline_headroom ;;
    m2) MODE=m2_anomaly_headroom ;;
    m3) MODE=m3_recursive_headroom ;;
    m4a) MODE=m4_fixed_equilibrium ;;
    m4b) MODE=m4_adaptive_equilibrium ;;
    m4_no_recurrence) MODE=m4_no_recurrence ;;
    m5) MODE=m5_envelope_realization; REALIZATION_ROOT="$REALIZATION_CACHE_ROOT" ;;
    m6a) MODE=m6_fixed_error_decay ;;
    m6b) MODE=m6_adaptive_error_decay ;;
    *) echo "Unknown dynamics group: $GROUP" >&2; exit 2 ;;
esac

if [[ "$TRAINING_STAGE" == head ]]; then
    EPOCHS=${WIND_DYNAMICS_HEAD_EPOCHS:-4}
    PATIENCE=${WIND_DYNAMICS_HEAD_PATIENCE:-4}
elif [[ "$TRAINING_STAGE" == joint ]]; then
    EPOCHS=${WIND_DYNAMICS_JOINT_EPOCHS:-16}
    PATIENCE=${WIND_DYNAMICS_JOINT_PATIENCE:-5}
else
    echo "Unknown dynamics stage: $TRAINING_STAGE" >&2
    exit 2
fi

OVERRIDES=(
    "seed=$SEED"
    "data.dynamics_time_cache_dir=$PROJECT_ROOT/data/aggregate_weather_dynamics_v1"
    "data.dynamics_latent_cache_root=$DYNAMICS_LATENT_ROOT"
    "data.realization_cache_root=$REALIZATION_ROOT"
    "data.train_exclude_origin_start_unix=${WIND_TRAIN_EXCLUDE_START_UNIX:-null}"
    "data.train_exclude_origin_stop_unix=${WIND_TRAIN_EXCLUDE_STOP_UNIX:-null}"
    "data.train_exclusion_embargo_seconds=${WIND_TRAIN_EXCLUSION_EMBARGO_SECONDS:-0}"
    "model.architecture=aggregate_weather_dynamics"
    "model.weather_route=r2_latent_interpolation"
    "model.history_visible_steps=12"
    "model.state_mode=current"
    "model.require_history_checkpoint=false"
    "model.weather_dynamics_mode=$MODE"
    "model.dynamics_training_stage=$TRAINING_STAGE"
    "model.dynamics_rate_min_per_hour=0.08333333333333333"
    "model.dynamics_rate_max_per_hour=4.0"
    "model.dynamics_initial_rate_per_hour=0.5"
    "model.realization_factor_max=1.2"
    "model.weather_quantile=$QUANTILE"
    "model.current_weather_loss_weight=$CURRENT_WEIGHT"
    "model.decoder_lr_multiplier=0.1"
    "training.cycles_per_batch=$CYCLES"
    "training.early_stopping_patience=$PATIENCE"
    "training.deterministic.epochs=$EPOCHS"
    "training.deterministic.learning_rate=0.0002"
    "training.deterministic.gradient_accumulation_steps=$ACCUMULATION"
    "training.deterministic.baseline_anchor_loss_weight=0.0"
    "training.deterministic.ramp_loss_weight=0.0"
    "training.deterministic.checkpoint_selection=${WIND_CHECKPOINT_SELECTION:-validation}"
)
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "aggregate_weather_dynamics_v1/$STAGE/$GROUP/seed$SEED" \
    "${OVERRIDES[@]}" >"$OUTPUT_DIR/config.log" 2>&1

EXTRA=()
[[ -z "$INITIAL_CHECKPOINT" ]] || EXTRA+=(--initial-checkpoint "$INITIAL_CHECKPOINT")
[[ -z "$AUXILIARY_CHECKPOINT" ]] || EXTRA+=(--auxiliary-checkpoint "$AUXILIARY_CHECKPOINT")
[[ -z ${WIND_AGG_MAX_TRAIN_BATCHES:-} ]] || EXTRA+=(--max-train-batches "$WIND_AGG_MAX_TRAIN_BATCHES")
[[ -z ${WIND_AGG_MAX_VAL_BATCHES:-} ]] || EXTRA+=(--max-val-batches "$WIND_AGG_MAX_VAL_BATCHES")

echo "[dynamics-job] start gpu=$GPU stage=$STAGE group=$GROUP seed=$SEED mode=$MODE train_stage=$TRAINING_STAGE"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_aggregate_cluster.py" \
    --config "$OUTPUT_DIR/config.yaml" --output-dir "$OUTPUT_DIR" --device cuda \
    --base-checkpoint "$BASE_CHECKPOINT" "${EXTRA[@]}" \
    > >(tee -a "$OUTPUT_DIR/run.log") 2> >(tee -a "$OUTPUT_DIR/run.log" >&2)

EVAL_LIMIT=()
[[ -z ${WIND_AGG_MAX_VAL_BATCHES:-} ]] || EVAL_LIMIT+=(--max-batches "$WIND_AGG_MAX_VAL_BATCHES")
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_aggregate_cluster.py" \
    --config "$OUTPUT_DIR/config.yaml" --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --split val --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" --device cuda \
    "${EVAL_LIMIT[@]}" \
    > >(tee -a "$OUTPUT_DIR/run.log") 2> >(tee -a "$OUTPUT_DIR/run.log" >&2)

AUDIT_ARGS=()
[[ -z ${WIND_AGG_MAX_VAL_BATCHES:-} ]] || AUDIT_ARGS+=(--allow-partial)
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_aggregate_run.py" --run-dir "$OUTPUT_DIR" "${AUDIT_ARGS[@]}"
touch "$OUTPUT_DIR/COMPLETE"
echo "[dynamics-job] complete gpu=$GPU stage=$STAGE group=$GROUP seed=$SEED"
