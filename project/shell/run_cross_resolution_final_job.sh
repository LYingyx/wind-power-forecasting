#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PHYSICAL_GPU normal|kd SEED" >&2
    exit 2
fi
GPU=$1 GROUP=$2 SEED=$3
[[ "$GROUP" == normal || "$GROUP" == kd ]] || { echo "group must be normal or kd" >&2; exit 2; }
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
BASE_CONFIG="${WIND_CR_CONFIG:-${PROJECT_ROOT}/configs/incremental_correction_pad5.yaml}"
EPOCHS="${WIND_CR_EPOCHS:-80}"
ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2"
TEMPORAL="$ROOT/temporal_selection.json"
SPATIAL="$ROOT/spatial/analysis_formal/decision.json"
OUTPUT_DIR="$ROOT/final/groups/$GROUP/seed$SEED"
TEACHER="$ROOT/final/teacher/train_teacher_normal.npz"

readarray -t VALUES < <("$PYTHON_BIN" - "$TEMPORAL" "$SPATIAL" <<'PY'
import json, pathlib, sys
t=json.loads(pathlib.Path(sys.argv[1]).read_text())
s=json.loads(pathlib.Path(sys.argv[2]).read_text())
if not t.get('temporal_structure_frozen') or not s.get('spatial_structure_frozen'):
    raise SystemExit('Final architecture is not frozen')
for value in (
    t['selected_time_query_mode'], t['selected_bracket_time_feature_mode'],
    t['selected_bracket_mapper_activation'], t['selected_forcing_difference_input'],
    s['selected_station_query_mode'], s['selected_station_neighborhood_cells'],
    str(bool(s['selected_include_wind_geometry'])).lower(),
): print(value)
PY
)
[[ ${#VALUES[@]} -eq 7 ]] || { echo "Could not resolve the frozen final architecture" >&2; exit 3; }
QUERY_MODE=${VALUES[0]} BRACKET_TIME_MODE=${VALUES[1]} BRACKET_ACTIVATION=${VALUES[2]}
DIFFERENCE_INPUT=${VALUES[3]} STATION_QUERY_MODE=${VALUES[4]} STATION_CELLS=${VALUES[5]}
WIND_GEOMETRY=${VALUES[6]}
LEGACY_HISTORY_ROOT="$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only"
NEW_HISTORY_ROOT="$PROJECT_ROOT/artifacts/incremental_mechanism_v2/history_prior"
if [[ -f "$LEGACY_HISTORY_ROOT/seed$SEED/best_model.pt" ]]; then
    HISTORY_CHECKPOINT="$LEGACY_HISTORY_ROOT/seed$SEED/best_model.pt"
else
    HISTORY_CHECKPOINT="$NEW_HISTORY_ROOT/seed$SEED/best_model.pt"
fi
[[ -f "$HISTORY_CHECKPOINT" ]] || { echo "Missing history checkpoint: $HISTORY_CHECKPOINT" >&2; exit 3; }
if [[ "$GROUP" == kd && ! -s "$TEACHER" ]]; then
    echo "Missing final ensemble teacher: $TEACHER" >&2; exit 3
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"
flock -x "$LOCK_FD"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]] && "$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_cross_resolution_final_run.py" \
    --project-root "$PROJECT_ROOT" --run-dir "$OUTPUT_DIR" --group "$GROUP" --seed "$SEED"; then
    echo "[cross-resolution-final] audited-reuse group=$GROUP seed=$SEED"
    exit 0
fi
rm -f "$OUTPUT_DIR/COMPLETE"
KD_WEIGHT=0.0; TEACHER_ARGS=()
if [[ "$GROUP" == kd ]]; then KD_WEIGHT=0.5; TEACHER_ARGS=(--teacher-archive "$TEACHER"); fi
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "cross_resolution_temporal_v2/final/${GROUP}/seed${SEED}" \
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
    "training.deterministic.distillation_weight=${KD_WEIGHT}" \
    training.deterministic.distillation_uncertainty_aware=false \
    training.deterministic.distillation_station_scale=0.0 \
    training.deterministic.distillation_cluster_scale=0.0 \
    >"$OUTPUT_DIR/config.log" 2>&1
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" --output-dir "$OUTPUT_DIR" --device cuda \
    --epochs "$EPOCHS" --history-checkpoint "$HISTORY_CHECKPOINT" "${TEACHER_ARGS[@]}" \
    >"$OUTPUT_DIR/run.log" 2> >(tee -a "$OUTPUT_DIR/run.log" >&2)
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --split val --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" --device cuda \
    >>"$OUTPUT_DIR/run.log" 2> >(tee -a "$OUTPUT_DIR/run.log" >&2)
touch "$OUTPUT_DIR/COMPLETE"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_cross_resolution_final_run.py" \
    --project-root "$PROJECT_ROOT" --run-dir "$OUTPUT_DIR" --group "$GROUP" --seed "$SEED"
echo "[cross-resolution-final] complete gpu=$GPU group=$GROUP seed=$SEED"
