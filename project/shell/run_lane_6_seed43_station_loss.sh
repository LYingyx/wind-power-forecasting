#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/experiment_common.sh"
initialize_lane "${1:?Usage: $0 GPU_ID}"
ensure_temporal_selection

promote_selected_seed42
run_final_seed 43
run_forecast_experiment controls station_only \
    "$EXPERIMENT_ROOT/selected/temporal/config.yaml" \
    "$EXPERIMENT_ROOT/selected/spatial/best_model.pt" \
    training.deterministic.cluster_loss_weight=0.0

mark_lane_complete lane_6
