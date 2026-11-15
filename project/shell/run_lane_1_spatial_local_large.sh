#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/experiment_common.sh"
initialize_lane "${1:?Usage: $0 GPU_ID}"

run_spatial_experiment s2_learned_cell3 \
    model.station_query_mode=learned_cells \
    model.station_neighborhood_cells=3 \
    model.include_wind_geometry=false \
    model.include_region_context=false \
    model.forcing_path_mode=independent

run_spatial_experiment s3_learned_cell5 \
    model.station_query_mode=learned_cells \
    model.station_neighborhood_cells=5 \
    model.include_wind_geometry=false \
    model.include_region_context=false \
    model.forcing_path_mode=independent

mark_lane_complete lane_1
