#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/experiment_common.sh"
initialize_lane "${1:?Usage: $0 GPU_ID}"

run_spatial_experiment s0_nearest_point \
    model.station_query_mode=nearest \
    model.station_neighborhood_cells=1 \
    model.include_wind_geometry=false \
    model.include_region_context=false \
    model.forcing_path_mode=independent

run_spatial_experiment s0b_bilinear_cell1 \
    model.station_query_mode=bilinear \
    model.station_neighborhood_cells=1 \
    model.include_wind_geometry=false \
    model.include_region_context=false \
    model.forcing_path_mode=independent

run_spatial_experiment s1_learned_cell1 \
    model.station_query_mode=learned_cells \
    model.station_neighborhood_cells=1 \
    model.include_wind_geometry=false \
    model.include_region_context=false \
    model.forcing_path_mode=independent

mark_lane_complete lane_0
