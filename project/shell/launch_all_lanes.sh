#!/usr/bin/env bash
set -euo pipefail

SHELL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SHELL_DIR/.." && pwd)
EXPERIMENT_ROOT=${WIND_EXPERIMENT_ROOT:-$PROJECT_ROOT/artifacts/experiments_pad5_cells}
GPU_CSV=${1:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
if [[ ${#GPUS[@]} -ne 8 ]]; then
    echo "Usage: $0 gpu0,gpu1,gpu2,gpu3,gpu4,gpu5,gpu6,gpu7" >&2
    exit 2
fi

LANES=(
    run_lane_0_spatial_local_small.sh
    run_lane_1_spatial_local_large.sh
    run_lane_2_spatial_geometry.sh
    run_lane_3_forcing_path_local.sh
    run_lane_4_forcing_path_deep.sh
    run_lane_5_training_controls.sh
    run_lane_6_seed43_station_loss.sh
    run_lane_7_seed44_cluster_loss_finalize.sh
)

mkdir -p "$EXPERIMENT_ROOT/logs"
JOBS_FILE="$EXPERIMENT_ROOT/logs/jobs.tsv"
: > "$JOBS_FILE"
printf 'lane\tgpu\tpid\texperiment_root\n' >> "$JOBS_FILE"

PIDS=()
terminate_children() {
    trap - INT TERM
    echo
    echo "[launcher] stopping ${#PIDS[@]} lane processes"
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap terminate_children INT TERM

for index in "${!LANES[@]}"; do
    lane=${LANES[$index]}
    gpu=${GPUS[$index]}
    WIND_LANE_ID="$index" \
    WIND_PROGRESS_PREFIX="[lane-$index]" \
    WIND_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
    bash "$SHELL_DIR/$lane" "$gpu" &
    pid=$!
    PIDS+=("$pid")
    printf '%s\t%s\t%s\t%s\n' \
        "$index" "$gpu" "$pid" "$EXPERIMENT_ROOT" >> "$JOBS_FILE"
    echo "[launcher] started lane=$index gpu=$gpu pid=$pid"
done
echo "[launch] Job table: $JOBS_FILE"
echo "[launcher] all lanes are attached to this terminal; Ctrl-C stops this launch"

failed=0
for index in "${!PIDS[@]}"; do
    pid=${PIDS[$index]}
    if wait "$pid"; then
        echo "[launcher] finished lane=$index pid=$pid status=0"
    else
        status=$?
        echo "[launcher] failed lane=$index pid=$pid status=$status" >&2
        failed=1
    fi
done
trap - INT TERM
if (( failed != 0 )); then
    exit 1
fi
echo "[launcher] all lanes completed"
