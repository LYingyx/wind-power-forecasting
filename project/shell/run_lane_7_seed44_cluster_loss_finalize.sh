#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/experiment_common.sh"
initialize_lane "${1:?Usage: $0 GPU_ID}"
ensure_temporal_selection

run_final_seed 44
run_forecast_experiment controls cluster_only \
    "$EXPERIMENT_ROOT/selected/temporal/config.yaml" \
    "$EXPERIMENT_ROOT/selected/spatial/best_model.pt" \
    training.deterministic.station_loss_weight=0.0

for target in \
    "$EXPERIMENT_ROOT/controls/scratch/COMPLETE" \
    "$EXPERIMENT_ROOT/controls/no_aux/COMPLETE" \
    "$EXPERIMENT_ROOT/controls/station_only/COMPLETE" \
    "$EXPERIMENT_ROOT/final/seed42/TEST_COMPLETE" \
    "$EXPERIMENT_ROOT/final/seed43/TEST_COMPLETE"; do
    wait_for_file "$target" "final prerequisite $(basename "$(dirname "$target")")"
done

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/summarize_experiments.py" \
    --root "$EXPERIMENT_ROOT" \
    --output-dir "$EXPERIMENT_ROOT/summary"
touch "$EXPERIMENT_ROOT/ALL_COMPLETE"
mark_lane_complete lane_7
