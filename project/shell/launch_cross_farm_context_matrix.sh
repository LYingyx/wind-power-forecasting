#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_CROSS_FARM_ROOT:-${PROJECT_ROOT}/artifacts/cross_farm_context_v1}"
SEED_CSV="${WIND_CROSS_FARM_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
SLOT_CSV="${1:-0,1,2,3}"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
IFS=',' read -r -a SLOTS <<< "$SLOT_CSV"

if [[ ${#SEEDS[@]} -ne 10 ]]; then
    echo "WIND_CROSS_FARM_SEEDS must contain exactly ten seeds" >&2
    exit 2
fi
if [[ ${#SLOTS[@]} -lt 1 ]]; then
    echo "At least one GPU slot is required" >&2
    exit 2
fi

export WIND_CROSS_FARM_ROOT="$EXPERIMENT_ROOT"
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
        "g1_self_duplicate|${seed}|self_duplicate"
        "g2_geographic_k4|${seed}|geographic"
        "g3_global_other|${seed}|global_other"
        "g4_weather_coherent_k4|${seed}|weather_coherent"
    )
done

total=${#JOBS[@]}
next=0
available_slots=("${SLOTS[@]}")
declare -A pid_to_slot=()
declare -A pid_to_label=()
echo "[cross-farm-v1] stage-start jobs=$total slots=${#SLOTS[@]}"
while (( next < total || ${#ACTIVE_PIDS[@]} > 0 )); do
    while (( next < total && ${#available_slots[@]} > 0 )); do
        slot_index=$((${#available_slots[@]} - 1))
        gpu=${available_slots[$slot_index]}
        unset "available_slots[$slot_index]"
        IFS='|' read -r group seed mode <<< "${JOBS[$next]}"
        bash "$PROJECT_ROOT/shell/run_cross_farm_context_job.sh" \
            "$gpu" "$group" "$seed" "$mode" &
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
            echo "[cross-farm-v1] failed job=${finished_label}" >&2
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
echo "[cross-farm-v1] all complete root=$EXPERIMENT_ROOT"
