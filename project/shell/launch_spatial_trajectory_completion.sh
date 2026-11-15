#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_STATE_LADDER_ROOT:-${PROJECT_ROOT}/artifacts/state_forcing_ladder_v1}"
SEED_CSV="${WIND_STATE_LADDER_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
SLOT_CSV="${1:-2,2,2,2}"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
IFS=',' read -r -a SLOTS <<< "$SLOT_CSV"

if [[ ${#SEEDS[@]} -ne 10 ]]; then
    echo "WIND_STATE_LADDER_SEEDS must contain exactly ten seeds" >&2
    exit 2
fi
if [[ ${#SLOTS[@]} -lt 1 ]]; then
    echo "At least one GPU slot is required" >&2
    exit 2
fi

export WIND_STATE_LADDER_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/logs"
rm -f "$EXPERIMENT_ROOT/SPATIAL_TRAJECTORY_COMPLETE"
JOBS_LOG="$EXPERIMENT_ROOT/logs/spatial_trajectory_jobs.tsv"
printf 'index\tgpu\tpid\tseed\n' > "$JOBS_LOG"

ACTIVE_PIDS=()
stop_active() {
    trap - INT TERM
    for pid in "${ACTIVE_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap stop_active INT TERM

next=0
available_slots=("${SLOTS[@]}")
declare -A pid_to_slot=()
declare -A pid_to_seed=()
echo "[spatial-trajectory-v1] stage-start jobs=${#SEEDS[@]} slots=${#SLOTS[@]}"
while (( next < ${#SEEDS[@]} || ${#ACTIVE_PIDS[@]} > 0 )); do
    while (( next < ${#SEEDS[@]} && ${#available_slots[@]} > 0 )); do
        slot_index=$((${#available_slots[@]} - 1))
        gpu=${available_slots[$slot_index]}
        unset "available_slots[$slot_index]"
        seed=${SEEDS[$next]}
        bash "$PROJECT_ROOT/shell/run_state_forcing_ladder_job.sh" \
            "$gpu" frozen_cluster_recent_1h "$seed" history_weather cluster_recent_1h &
        pid=$!
        ACTIVE_PIDS+=("$pid")
        pid_to_slot[$pid]=$gpu
        pid_to_seed[$pid]=$seed
        printf '%s\t%s\t%s\t%s\n' "$next" "$gpu" "$pid" "$seed" >> "$JOBS_LOG"
        next=$((next + 1))
    done
    if (( ${#ACTIVE_PIDS[@]} > 0 )); then
        status=0
        wait -n -p finished_pid "${ACTIVE_PIDS[@]}" || status=$?
        finished_seed=${pid_to_seed[$finished_pid]}
        available_slots+=("${pid_to_slot[$finished_pid]}")
        unset "pid_to_slot[$finished_pid]" "pid_to_seed[$finished_pid]"
        remaining=()
        for active_pid in "${ACTIVE_PIDS[@]}"; do
            if [[ "$active_pid" != "$finished_pid" ]]; then
                remaining+=("$active_pid")
            fi
        done
        ACTIVE_PIDS=("${remaining[@]}")
        if (( status != 0 )); then
            echo "[spatial-trajectory-v1] failed seed=${finished_seed}" >&2
            for active_pid in "${ACTIVE_PIDS[@]}"; do
                kill -TERM "$active_pid" 2>/dev/null || true
            done
            wait 2>/dev/null || true
            exit "$status"
        fi
    fi
done

touch "$EXPERIMENT_ROOT/SPATIAL_TRAJECTORY_COMPLETE"
echo "[spatial-trajectory-v1] all complete root=$EXPERIMENT_ROOT"
