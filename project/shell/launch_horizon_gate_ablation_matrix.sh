#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_HORIZON_CONTROL_ROOT:-${PROJECT_ROOT}/artifacts/horizon_control_matrix_v1}"
SEED_CSV="${WIND_HORIZON_CONTROL_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
SLOT_CSV="${1:-2,2,2,2}"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
IFS=',' read -r -a SLOTS <<< "$SLOT_CSV"
MODEL_GROUPS=(
    horizon_film
    horizon_gated_multiplicative
    horizon_conditioned_query
    horizon_attention_bias
    horizon_conditional_lora
)

if [[ ${#SEEDS[@]} -ne 10 ]]; then
    echo "WIND_HORIZON_CONTROL_SEEDS must contain exactly ten seeds" >&2
    exit 2
fi
if [[ ! -f "$EXPERIMENT_ROOT/HORIZON_CONTROL_COMPLETE" ]]; then
    echo "Formal horizon-control matrix is incomplete" >&2
    exit 3
fi
mkdir -p "$EXPERIMENT_ROOT/logs"
if [[ -f "$EXPERIMENT_ROOT/HORIZON_GATE_ABLATION_COMPLETE" ]]; then
    unlink "$EXPERIMENT_ROOT/HORIZON_GATE_ABLATION_COMPLETE"
fi

JOB_SEEDS=()
JOB_GROUPS=()
for seed in "${SEEDS[@]}"; do
    for group in "${MODEL_GROUPS[@]}"; do
        JOB_SEEDS+=("$seed")
        JOB_GROUPS+=("$group")
    done
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

next=0
available_slots=("${SLOTS[@]}")
declare -A pid_to_slot=()
declare -A pid_to_label=()
echo "[horizon-gate-ablation-v1] stage-start jobs=${#JOB_SEEDS[@]} slots=${#SLOTS[@]}"
while (( next < ${#JOB_SEEDS[@]} || ${#ACTIVE_PIDS[@]} > 0 )); do
    while (( next < ${#JOB_SEEDS[@]} && ${#available_slots[@]} > 0 )); do
        slot_index=$((${#available_slots[@]} - 1))
        gpu=${available_slots[$slot_index]}
        unset "available_slots[$slot_index]"
        seed=${JOB_SEEDS[$next]}
        group=${JOB_GROUPS[$next]}
        bash "$PROJECT_ROOT/shell/run_horizon_gate_ablation_job.sh" "$gpu" "$group" "$seed" &
        pid=$!
        ACTIVE_PIDS+=("$pid")
        pid_to_slot[$pid]=$gpu
        pid_to_label[$pid]="${group}/seed${seed}"
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
            echo "[horizon-gate-ablation-v1] failed ${finished_label}" >&2
            for active_pid in "${ACTIVE_PIDS[@]}"; do
                kill -TERM "$active_pid" 2>/dev/null || true
            done
            wait 2>/dev/null || true
            exit "$status"
        fi
    fi
done

touch "$EXPERIMENT_ROOT/HORIZON_GATE_ABLATION_COMPLETE"
echo "[horizon-gate-ablation-v1] all complete root=$EXPERIMENT_ROOT"
