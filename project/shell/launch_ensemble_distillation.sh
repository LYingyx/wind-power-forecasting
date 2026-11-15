#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plain}"
SLOT_CSV="${2:-5}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_STABILIZATION_ROOT:-${PROJECT_ROOT}/artifacts/ensemble_stabilization_v1}"
SEED_CSV="${WIND_STABILIZATION_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
IFS=',' read -r -a SLOTS <<< "$SLOT_CSV"

if [[ ${#SEEDS[@]} -ne 10 ]]; then
    echo "WIND_STABILIZATION_SEEDS must contain exactly ten seeds" >&2
    exit 2
fi

case "$MODE" in
    plain)
        METHODS=("kd_025|0.25|false" "kd_050|0.50|false")
        ;;
    aware025)
        METHODS=("kd_aware_025|0.25|true")
        ;;
    aware050)
        METHODS=("kd_aware_050|0.50|true")
        ;;
    *)
        echo "mode must be plain, aware025, or aware050" >&2
        exit 2
        ;;
esac

export WIND_STABILIZATION_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/groups" "$EXPERIMENT_ROOT/logs"
JOBS_LOG="$EXPERIMENT_ROOT/logs/${MODE}_jobs.tsv"
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
for method in "${METHODS[@]}"; do
    IFS='|' read -r group weight aware <<< "$method"
    for seed in "${SEEDS[@]}"; do
        JOBS+=("${group}|${seed}|${weight}|${aware}")
    done
done

total=${#JOBS[@]}
next=0
available_slots=("${SLOTS[@]}")
declare -A pid_to_slot=()
declare -A pid_to_label=()
echo "[distillation-v1] mode=$MODE jobs=$total slots=${#SLOTS[@]}"
while (( next < total || ${#ACTIVE_PIDS[@]} > 0 )); do
    while (( next < total && ${#available_slots[@]} > 0 )); do
        slot_index=$((${#available_slots[@]} - 1))
        gpu=${available_slots[$slot_index]}
        unset "available_slots[$slot_index]"
        IFS='|' read -r -a fields <<< "${JOBS[$next]}"
        bash "$PROJECT_ROOT/shell/run_ensemble_distillation_job.sh" \
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
            echo "[distillation-v1] failed job=${finished_label}" >&2
            for active_pid in "${ACTIVE_PIDS[@]}"; do
                kill -TERM "$active_pid" 2>/dev/null || true
            done
            wait 2>/dev/null || true
            exit "$status"
        fi
    fi
done

touch "$EXPERIMENT_ROOT/${MODE^^}_COMPLETE"
echo "[distillation-v1] mode=$MODE complete root=$EXPERIMENT_ROOT"
