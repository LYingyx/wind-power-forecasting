#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_STATE_LADDER_ROOT:-${PROJECT_ROOT}/artifacts/state_forcing_ladder_v1}"
SEED_CSV="${WIND_STATE_LADDER_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
SLOT_CSV="${1:-0,0,0,1,1,1,3,3,3,4,5,6,7}"
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
mkdir -p "$EXPERIMENT_ROOT/groups" "$EXPERIMENT_ROOT/logs"
rm -f "$EXPERIMENT_ROOT/ALL_COMPLETE"
JOBS_LOG="$EXPERIMENT_ROOT/logs/jobs.tsv"
printf 'index\tgpu\tpid\tjob\n' > "$JOBS_LOG"

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

JOBS=()
for seed in "${SEEDS[@]}"; do
    JOBS+=(
        "frozen_weather_only|${seed}|weather_only|full_history"
        "frozen_cluster_current|${seed}|history_weather|cluster_current"
        "frozen_station_current|${seed}|history_weather|station_current"
        "frozen_cluster_ramp15m|${seed}|history_weather|cluster_current_ramp15m"
        "frozen_recent_1h|${seed}|history_weather|recent_1h"
        "frozen_full_history|${seed}|history_weather|full_history"
    )
done

total=${#JOBS[@]}
next=0
available_slots=("${SLOTS[@]}")
declare -A pid_to_slot=()
declare -A pid_to_label=()
echo "[state-ladder-v1] stage-start jobs=$total slots=${#SLOTS[@]}"
while (( next < total || ${#ACTIVE_PIDS[@]} > 0 )); do
    while (( next < total && ${#available_slots[@]} > 0 )); do
        slot_index=$((${#available_slots[@]} - 1))
        gpu=${available_slots[$slot_index]}
        unset "available_slots[$slot_index]"
        IFS='|' read -r -a fields <<< "${JOBS[$next]}"
        bash "$PROJECT_ROOT/shell/run_state_forcing_ladder_job.sh" \
            "$gpu" "${fields[@]}" &
        pid=$!
        ACTIVE_PIDS+=("$pid")
        pid_to_slot[$pid]=$gpu
        pid_to_label[$pid]="${JOBS[$next]}"
        printf '%s\t%s\t%s\t%s\n' \
            "$next" "$gpu" "$pid" "${JOBS[$next]}" >> "$JOBS_LOG"
        next=$((next + 1))
    done
    if (( ${#ACTIVE_PIDS[@]} > 0 )); then
        status=0
        wait -n -p finished_pid "${ACTIVE_PIDS[@]}" || status=$?
        finished_label=${pid_to_label[$finished_pid]}
        available_slots+=("${pid_to_slot[$finished_pid]}")
        unset "pid_to_slot[$finished_pid]" "pid_to_label[$finished_pid]"
        remaining=()
        for active_pid in "${ACTIVE_PIDS[@]}"; do
            if [[ "$active_pid" != "$finished_pid" ]]; then
                remaining+=("$active_pid")
            fi
        done
        ACTIVE_PIDS=("${remaining[@]}")
        if (( status != 0 )); then
            echo "[state-ladder-v1] failed job=${finished_label}" >&2
            for active_pid in "${ACTIVE_PIDS[@]}"; do
                kill -TERM "$active_pid" 2>/dev/null || true
            done
            wait 2>/dev/null || true
            ACTIVE_PIDS=()
            exit "$status"
        fi
    fi
done

touch "$EXPERIMENT_ROOT/ALL_COMPLETE"
echo "[state-ladder-v1] all complete root=$EXPERIMENT_ROOT"
