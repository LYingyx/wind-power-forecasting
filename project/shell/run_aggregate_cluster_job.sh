#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 PHYSICAL_GPU STAGE GROUP SEED [HISTORY_CHECKPOINT]" >&2
    exit 2
fi

GPU=$1
STAGE=$2
GROUP=$3
SEED=$4
HISTORY_CHECKPOINT=${5:-}
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
BASE_CONFIG=${WIND_AGG_CONFIG:-$PROJECT_ROOT/configs/aggregate_cluster_v1.yaml}
EXPERIMENT_ROOT=${WIND_AGG_ROOT:-$PROJECT_ROOT/artifacts/aggregate_cluster_v1}
OUTPUT_DIR=$EXPERIMENT_ROOT/$STAGE/$GROUP/seed$SEED
HISTORY_VISIBLE=${WIND_AGG_HISTORY_VISIBLE:-24}
STATE_MODE=${WIND_AGG_STATE_MODE:-recent_1h}
EPOCHS=${WIND_AGG_EPOCHS:-80}
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
        echo "[aggregate-job] audited-reuse stage=$STAGE group=$GROUP seed=$SEED"
        exit 0
    fi
    echo "[aggregate-job] invalid completion marker; rerunning $OUTPUT_DIR" >&2
fi
rm -f "$OUTPUT_DIR/COMPLETE"

ARCHITECTURE=aggregate_correction
WEATHER_ROUTE=r3_full_path
VISIBLE=$HISTORY_VISIBLE
STATE=$STATE_MODE
CONDITIONING=state_weather
CORRECTION_FUSION=blackbox
ANCHOR=0.2
REQUIRE_HISTORY=true

case "$GROUP" in
    hb_persistence) ARCHITECTURE=hb_persistence; VISIBLE=1; ANCHOR=0.0; REQUIRE_HISTORY=false; EPOCHS=1 ;;
    hb_dlinear) ARCHITECTURE=hb_dlinear; VISIBLE=24; ANCHOR=0.0; REQUIRE_HISTORY=false ;;
    hb_tcn_1h) ARCHITECTURE=hb_tcn; VISIBLE=4; ANCHOR=0.0; REQUIRE_HISTORY=false ;;
    hb_tcn_3h) ARCHITECTURE=hb_tcn; VISIBLE=12; ANCHOR=0.0; REQUIRE_HISTORY=false ;;
    hb_tcn_6h) ARCHITECTURE=hb_tcn; VISIBLE=24; ANCHOR=0.0; REQUIRE_HISTORY=false ;;
    hb_patchtst) ARCHITECTURE=hb_patchtst; VISIBLE=24; ANCHOR=0.0; REQUIRE_HISTORY=false ;;
    sc_current) STATE=current ;;
    sc_1h) STATE=recent_1h ;;
    sc_3h) STATE=recent_3h ;;
    sc_6h) STATE=full_6h ;;
    sc_multi) STATE=multiscale ;;
    sc_horizon) STATE=horizon ;;
    r1_raw_interpolation|r2_latent_interpolation|r3_full_path|r4_target_value|r4_no_time|r5_dynamic_pool|r6_regional_tokens|r7_patch_regional|r8_joint_spacetime|r8_factorized)
        WEATHER_ROUTE=$GROUP ;;
    r2_w) WEATHER_ROUTE=r2_latent_interpolation; CONDITIONING=weather ;;
    r2_add) WEATHER_ROUTE=r2_latent_interpolation; CONDITIONING=state_weather; CORRECTION_FUSION=additive ;;
    w_only|f_early_concat|f_target_concat|f_lead_gate|f_prediction_blend|f_cross_attention|f_bidirectional|f_tft|f_astafn|f_timexer)
        ARCHITECTURE=$GROUP; ANCHOR=0.0; REQUIRE_HISTORY=false ;;
    m0) CONDITIONING=none ;;
    ms) CONDITIONING=state ;;
    mw) CONDITIONING=weather ;;
    mfull) CONDITIONING=state_weather ;;
    madd) CONDITIONING=state_weather; CORRECTION_FUSION=additive ;;
    *) echo "Unknown aggregate group: $GROUP" >&2; exit 2 ;;
esac

if [[ "$ARCHITECTURE" == aggregate_correction && -z "$HISTORY_CHECKPOINT" ]]; then
    echo "Missing history checkpoint for group=$GROUP seed=$SEED" >&2
    exit 3
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "${WIND_AGG_EXPERIMENT_NAMESPACE:-aggregate_cluster_v1}/$STAGE/$GROUP/seed$SEED" \
    "seed=$SEED" \
    "model.architecture=$ARCHITECTURE" \
    "model.weather_route=$WEATHER_ROUTE" \
    "model.history_visible_steps=$VISIBLE" \
    "model.state_mode=$STATE" \
    "model.correction_conditioning=$CONDITIONING" \
    "model.correction_fusion=$CORRECTION_FUSION" \
    "model.require_history_checkpoint=$REQUIRE_HISTORY" \
    "training.cycles_per_batch=$CYCLES" \
    "training.deterministic.gradient_accumulation_steps=$ACCUMULATION" \
    "training.deterministic.baseline_anchor_loss_weight=$ANCHOR" \
    >"$OUTPUT_DIR/config.log" 2>&1

TRAIN_LIMIT=()
EVAL_LIMIT=()
if [[ -n ${WIND_AGG_MAX_TRAIN_BATCHES:-} ]]; then
    TRAIN_LIMIT+=(--max-train-batches "$WIND_AGG_MAX_TRAIN_BATCHES")
fi
if [[ -n ${WIND_AGG_MAX_VAL_BATCHES:-} ]]; then
    TRAIN_LIMIT+=(--max-val-batches "$WIND_AGG_MAX_VAL_BATCHES")
    EVAL_LIMIT+=(--max-batches "$WIND_AGG_MAX_VAL_BATCHES")
fi
HISTORY_ARGS=()
if [[ -n "$HISTORY_CHECKPOINT" ]]; then
    HISTORY_ARGS+=(--history-checkpoint "$HISTORY_CHECKPOINT")
fi

echo "[aggregate-job] start gpu=$GPU stage=$STAGE group=$GROUP seed=$SEED cycles=$CYCLES accumulation=$ACCUMULATION"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_aggregate_cluster.py" \
    --config "$OUTPUT_DIR/config.yaml" --output-dir "$OUTPUT_DIR" --device cuda \
    --epochs "$EPOCHS" "${HISTORY_ARGS[@]}" "${TRAIN_LIMIT[@]}" \
    > >(tee -a "$OUTPUT_DIR/run.log") \
    2> >(tee -a "$OUTPUT_DIR/run.log" >&2)

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
echo "[aggregate-job] complete gpu=$GPU stage=$STAGE group=$GROUP seed=$SEED"
