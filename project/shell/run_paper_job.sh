#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "$0")/experiment_common.sh"

GPU_ID=${1:?Usage: $0 GPU_ID KIND CATEGORY ID BASE_CONFIG CHECKPOINT [OVERRIDES...]}
KIND=${2:?Missing job kind}
CATEGORY=${3:?Missing category}
EXPERIMENT_ID=${4:?Missing experiment id}
BASE=${5:-}
CHECKPOINT=${6:-}
shift 6
if [[ "$BASE" == "NONE" ]]; then BASE=""; fi
if [[ "$CHECKPOINT" == "NONE" ]]; then CHECKPOINT=""; fi

initialize_lane "$GPU_ID"
case "$KIND" in
    spatial)
        run_spatial_experiment_from_base "$EXPERIMENT_ID" "$BASE" "$@"
        ;;
    forecast)
        run_forecast_experiment \
            "$CATEGORY" "$EXPERIMENT_ID" "$BASE" "$CHECKPOINT" "$@"
        ;;
    test)
        evaluate_test_once "$EXPERIMENT_ROOT/$CATEGORY/$EXPERIMENT_ID" "${1:-0}"
        ;;
    *)
        echo "Unsupported paper job kind: $KIND" >&2
        exit 2
        ;;
esac
