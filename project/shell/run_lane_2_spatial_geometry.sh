#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/experiment_common.sh"
initialize_lane "${1:?Usage: $0 GPU_ID}"
ensure_local_selection

LOCAL_CONFIG="$EXPERIMENT_ROOT/selected/local/config.yaml"

run_spatial_experiment_from_base s4_selected_local_wind "$LOCAL_CONFIG" \
    model.include_wind_geometry=true \
    model.include_region_context=false \
    model.forcing_path_mode=independent

run_spatial_experiment_from_base s5_selected_local_wind_context "$LOCAL_CONFIG" \
    model.include_wind_geometry=true \
    model.include_region_context=true \
    model.forcing_path_mode=independent

mark_lane_complete lane_2
