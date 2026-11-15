#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/experiment_common.sh"
initialize_lane "${1:?Usage: $0 GPU_ID}"
ensure_spatial_selection

SPATIAL_CONFIG="$EXPERIMENT_ROOT/selected/spatial/config.yaml"
SPATIAL_CHECKPOINT="$EXPERIMENT_ROOT/selected/spatial/best_model.pt"

run_forecast_experiment temporal f2_tcn "$SPATIAL_CONFIG" "$SPATIAL_CHECKPOINT" \
    model.forcing_path_mode=tcn
run_forecast_experiment temporal f3_transformer "$SPATIAL_CONFIG" "$SPATIAL_CHECKPOINT" \
    model.forcing_path_mode=transformer

mark_lane_complete lane_4
