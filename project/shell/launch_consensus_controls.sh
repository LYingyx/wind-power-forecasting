#!/usr/bin/env bash
set -euo pipefail

SLOT_CSV="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_CONSENSUS_ROOT:-${PROJECT_ROOT}/artifacts/consensus_controls_v1}"
TEACHER_ROOT="${EXPERIMENT_ROOT}/teacher"
SEEDS=(42 43 44 45 46 47 48 49 50 51)
TEACHER_SEEDS=(43 44 45 46 47 48 49 50 51 42)
IFS=',' read -r -a SLOTS <<< "$SLOT_CSV"

if [[ ${#SLOTS[@]} -lt 1 ]]; then
    echo "At least one GPU slot is required" >&2
    exit 2
fi
if [[ ! -f "$TEACHER_ROOT/COMPLETE" ]]; then
    echo "Teacher archives are incomplete: run prepare_consensus_control_teachers.sh" >&2
    exit 3
fi

export WIND_CONSENSUS_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/groups" "$EXPERIMENT_ROOT/logs"
JOBS_LOG="$EXPERIMENT_ROOT/logs/jobs.tsv"
printf 'index\tgpu\tpid\tgroup\tstudent_seed\tteacher\n' >"$JOBS_LOG"

JOBS=()
for index in "${!SEEDS[@]}"; do
    student_seed=${SEEDS[$index]}
    teacher_seed=${TEACHER_SEEDS[$index]}
    JOBS+=(
        "recent_single_teacher_kd050|${student_seed}|cluster_recent_1h|${TEACHER_ROOT}/single_recent/train_teacher_seed${teacher_seed}.npz|single_recent_seed${teacher_seed}"
    )
done
for seed in "${SEEDS[@]}"; do
    JOBS+=(
        "full_history_ensemble_kd050|${seed}|full_history|${TEACHER_ROOT}/train_teacher_full_history_ensemble.npz|full_history_ensemble_42_51"
    )
done

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

total=${#JOBS[@]}
next=0
available_slots=("${SLOTS[@]}")
declare -A pid_to_slot=()
declare -A pid_to_label=()
echo "[consensus-control] jobs=$total slots=${#SLOTS[@]}"
while (( next < total || ${#ACTIVE_PIDS[@]} > 0 )); do
    while (( next < total && ${#available_slots[@]} > 0 )); do
        slot_index=$((${#available_slots[@]} - 1))
        gpu=${available_slots[$slot_index]}
        unset "available_slots[$slot_index]"
        IFS='|' read -r group seed history_mode teacher_archive teacher_label <<<"${JOBS[$next]}"
        bash "$PROJECT_ROOT/shell/run_consensus_control_job.sh" \
            "$gpu" "$group" "$seed" "$history_mode" "$teacher_archive" "$teacher_label" &
        pid=$!
        ACTIVE_PIDS+=("$pid")
        pid_to_slot[$pid]=$gpu
        pid_to_label[$pid]="${JOBS[$next]}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$next" "$gpu" "$pid" "$group" "$seed" "$teacher_label" >>"$JOBS_LOG"
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
            echo "[consensus-control] failed job=${finished_label}" >&2
            for active_pid in "${ACTIVE_PIDS[@]}"; do
                kill -TERM "$active_pid" 2>/dev/null || true
            done
            wait 2>/dev/null || true
            exit "$status"
        fi
    fi
done

touch "$EXPERIMENT_ROOT/TRAINING_COMPLETE"
echo "[consensus-control] all training complete: $EXPERIMENT_ROOT"
