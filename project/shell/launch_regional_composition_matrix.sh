#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_COMPOSITION_ROOT:-${PROJECT_ROOT}/artifacts/regional_composition_v1}"
SEED_CSV="${WIND_COMPOSITION_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
SLOT_CSV="${1:-0,1,2}"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
IFS=',' read -r -a SLOTS <<< "$SLOT_CSV"

if [[ ${#SEEDS[@]} -ne 10 ]]; then
    echo "WIND_COMPOSITION_SEEDS must contain exactly ten seeds" >&2
    exit 2
fi
if [[ ${#SLOTS[@]} -lt 1 ]]; then
    echo "At least one GPU slot is required" >&2
    exit 2
fi

export WIND_COMPOSITION_ROOT="$EXPERIMENT_ROOT"
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
        "r0_matched_control|${seed}|composition_control"
        "r1_handcrafted_statistics|${seed}|composition_statistics"
        "r2_tiny_deepsets|${seed}|composition_deepsets"
    )
done

total=${#JOBS[@]}
next=0
available_slots=("${SLOTS[@]}")
declare -A pid_to_slot=()
declare -A pid_to_label=()
echo "[regional-composition-v1] stage-start jobs=$total slots=${#SLOTS[@]}"
while (( next < total || ${#ACTIVE_PIDS[@]} > 0 )); do
    while (( next < total && ${#available_slots[@]} > 0 )); do
        slot_index=$((${#available_slots[@]} - 1))
        gpu=${available_slots[$slot_index]}
        unset "available_slots[$slot_index]"
        IFS='|' read -r group seed mode <<< "${JOBS[$next]}"
        bash "$PROJECT_ROOT/shell/run_regional_composition_job.sh" \
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
            echo "[regional-composition-v1] failed job=${finished_label}" >&2
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
echo "[regional-composition-v1] all complete root=$EXPERIMENT_ROOT"
