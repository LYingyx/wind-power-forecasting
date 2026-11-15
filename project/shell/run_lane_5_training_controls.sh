#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/experiment_common.sh"
initialize_lane "${1:?Usage: $0 GPU_ID}"
ensure_temporal_selection

SELECTED_CONFIG="$EXPERIMENT_ROOT/selected/temporal/config.yaml"
SPATIAL_CHECKPOINT="$EXPERIMENT_ROOT/selected/spatial/best_model.pt"

run_forecast_experiment controls scratch "$SELECTED_CONFIG" ""
run_forecast_experiment controls no_aux "$SELECTED_CONFIG" "$SPATIAL_CHECKPOINT" \
    training.deterministic.auxiliary_weather_loss_weight=0.0

mark_lane_complete lane_5
