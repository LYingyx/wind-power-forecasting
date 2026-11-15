#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PHYSICAL_GPU GROUP SEED" >&2
    exit 2
fi
GPU=$1 GROUP=$2 SEED=$3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
EXPERIMENT_ROOT="${WIND_CR_ROOT:-${PROJECT_ROOT}/artifacts/cross_resolution_temporal_v1}"
BASE_CONFIG="${WIND_CR_CONFIG:-${PROJECT_ROOT}/configs/incremental_correction_pad5.yaml}"
EPOCHS="${WIND_CR_EPOCHS:-80}"
MAX_TRAIN_BATCHES="${WIND_CR_MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${WIND_CR_MAX_VAL_BATCHES:-}"
OUTPUT_DIR="$EXPERIMENT_ROOT/groups/$GROUP/seed$SEED"

QUERY_MODE="${WIND_CR_TIME_QUERY_MODE:-bracketing_mlp}"
BRACKET_TIME_MODE="${WIND_CR_BRACKET_TIME_FEATURE_MODE:-relative_distances}"
BRACKET_ACTIVATION="${WIND_CR_BRACKET_MAPPER_ACTIVATION:-gelu}"
DIFFERENCE_INPUT="${WIND_CR_FORCING_DIFFERENCE_INPUT:-actual}"
STATION_QUERY_MODE=learned_cells
STATION_CELLS=3
WIND_GEOMETRY=true
case "$GROUP" in
    t0_raw_linear) QUERY_MODE=raw_linear ;;
    t1_latent_linear) QUERY_MODE=latent_linear ;;
    t2_bracket_nonlinear) QUERY_MODE=bracketing_mlp ;;
    t3_nearest) QUERY_MODE=nearest ;;
    t4_native_query) QUERY_MODE=full_attention ;;
    t2_phase_off) QUERY_MODE=bracketing_mlp; BRACKET_TIME_MODE=zero ;;
    t2_linear) QUERY_MODE=bracketing_mlp; BRACKET_ACTIVATION=identity ;;
    t2_no_diff) QUERY_MODE=bracketing_mlp; DIFFERENCE_INPUT=zero ;;
    s0_raw_bilinear) STATION_QUERY_MODE=raw_bilinear; STATION_CELLS=1; WIND_GEOMETRY=false ;;
    s1_latent_bilinear) STATION_QUERY_MODE=bilinear; STATION_CELLS=1; WIND_GEOMETRY=false ;;
    s2_learned_2x2) STATION_QUERY_MODE=learned_cells; STATION_CELLS=1; WIND_GEOMETRY=false ;;
    s3_fixed_gaussian_4x4) STATION_QUERY_MODE=fixed_gaussian; STATION_CELLS=3; WIND_GEOMETRY=false ;;
    s4_learned_4x4_nogeom) STATION_QUERY_MODE=learned_cells; STATION_CELLS=3; WIND_GEOMETRY=false ;;
    s5_learned_4x4_wind) STATION_QUERY_MODE=learned_cells; STATION_CELLS=3; WIND_GEOMETRY=true ;;
    *) echo "Unknown cross-resolution group: $GROUP" >&2; exit 2 ;;
esac

LEGACY_HISTORY_ROOT="$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only"
NEW_HISTORY_ROOT="$PROJECT_ROOT/artifacts/incremental_mechanism_v2/history_prior"
if [[ -f "$LEGACY_HISTORY_ROOT/seed$SEED/best_model.pt" ]]; then
    HISTORY_CHECKPOINT="$LEGACY_HISTORY_ROOT/seed$SEED/best_model.pt"
else
    HISTORY_CHECKPOINT="$NEW_HISTORY_ROOT/seed$SEED/best_model.pt"
fi
[[ -f "$HISTORY_CHECKPOINT" ]] || {
    echo "Missing history checkpoint: $HISTORY_CHECKPOINT" >&2; exit 3;
}

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"
flock -x "$LOCK_FD"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    if "$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_cross_resolution_run.py" \
        --run-dir "$OUTPUT_DIR" --group "$GROUP" --seed "$SEED"; then
        echo "[cross-resolution] audited-reuse group=$GROUP seed=$SEED"
        exit 0
    fi
    echo "[cross-resolution] prior completion marker failed audit; rerunning" >&2
fi
rm -f "$OUTPUT_DIR/COMPLETE"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "${WIND_CR_EXPERIMENT_NAMESPACE:-cross_resolution_temporal_v2}/${GROUP}/seed${SEED}" \
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
    model.weather_view_mode=legacy \
    model.weather_gate_mode=fixed \
    model.specialist_mode=shared_head \
    "model.time_query_mode=${QUERY_MODE}" \
    "model.bracket_time_feature_mode=${BRACKET_TIME_MODE}" \
    "model.bracket_mapper_activation=${BRACKET_ACTIVATION}" \
    "model.forcing_difference_input=${DIFFERENCE_INPUT}" \
    "model.station_query_mode=${STATION_QUERY_MODE}" \
    "model.station_neighborhood_cells=${STATION_CELLS}" \
    "model.include_wind_geometry=${WIND_GEOMETRY}" \
    training.deterministic.baseline_auxiliary_loss_weight=0.0 \
    training.deterministic.baseline_anchor_loss_weight=0.2 \
    training.deterministic.anchor_schedule_mode=horizon_taper \
    training.deterministic.distillation_weight=0.0 \
    >"$OUTPUT_DIR/config.log" 2>&1

TRAIN_LIMIT=(); EVAL_LIMIT=()
[[ -z "$MAX_TRAIN_BATCHES" ]] || TRAIN_LIMIT+=(--max-train-batches "$MAX_TRAIN_BATCHES")
if [[ -n "$MAX_VAL_BATCHES" ]]; then
    TRAIN_LIMIT+=(--max-val-batches "$MAX_VAL_BATCHES")
    EVAL_LIMIT+=(--max-batches "$MAX_VAL_BATCHES")
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" --output-dir "$OUTPUT_DIR" \
    --device cuda --epochs "$EPOCHS" --history-checkpoint "$HISTORY_CHECKPOINT" \
    "${TRAIN_LIMIT[@]}" >"$OUTPUT_DIR/run.log" \
    2> >(tee -a "$OUTPUT_DIR/run.log" >&2)
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --split val --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" --device cuda \
    "${EVAL_LIMIT[@]}" >>"$OUTPUT_DIR/run.log" \
    2> >(tee -a "$OUTPUT_DIR/run.log" >&2)
touch "$OUTPUT_DIR/COMPLETE"
echo "[cross-resolution] complete gpu=$GPU group=$GROUP seed=$SEED"
