#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_CONDITIONAL_VALIDATION_ROOT:-${PROJECT_ROOT}/artifacts/conditional_forcing_validation_v1}"
SEED_CSV="${WIND_CONDITIONAL_VALIDATION_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
PERMUTATION_CSV="${WIND_WEATHER_PERMUTATION_SEEDS:-7301,7302,7303,7304,7305}"
SLOT_CSV="${1:-0,0,0,1,1,1,3,3,3,4,5,6,7}"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
IFS=',' read -r -a PERMUTATION_SEEDS <<< "$PERMUTATION_CSV"
IFS=',' read -r -a SLOTS <<< "$SLOT_CSV"

if [[ ${#SEEDS[@]} -ne 10 ]]; then
    echo "WIND_CONDITIONAL_VALIDATION_SEEDS must contain exactly ten seeds" >&2
    exit 2
fi
if [[ ${#PERMUTATION_SEEDS[@]} -lt 1 ]]; then
    echo "At least one controlled weather-permutation seed is required" >&2
    exit 2
fi
if [[ ${#SLOTS[@]} -lt 1 ]]; then
    echo "At least one GPU slot is required" >&2
    exit 2
fi

export WIND_CONDITIONAL_VALIDATION_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/groups" "$EXPERIMENT_ROOT/logs"
rm -f "$EXPERIMENT_ROOT/ALL_COMPLETE"
JOBS_LOG="$EXPERIMENT_ROOT/logs/jobs.tsv"
printf 'stage\tindex\tgpu\tpid\tjob\n' > "$JOBS_LOG"

link_group() {
    local label=$1
    local source=$2
    local destination="$EXPERIMENT_ROOT/groups/$label"
    if [[ -L "$destination" ]]; then
        if [[ "$(readlink -f "$destination")" != "$(readlink -f "$source")" ]]; then
            echo "Existing group link has a different target: $destination" >&2
            exit 3
        fi
    elif [[ -e "$destination" ]]; then
        echo "Cannot create group link over existing path: $destination" >&2
        exit 3
    else
        ln -s "$(readlink -f "$source")" "$destination"
    fi
}

V2_ROOT="$PROJECT_ROOT/artifacts/incremental_mechanism_v2"
link_group factor_history_weather "$V2_ROOT/groups/joint_full"
link_group reference_supervised_aux "$V2_ROOT/groups/joint_full"
link_group anchor_000 "$V2_ROOT/groups/joint_no_baseline_aux"
link_group reference_direct_d64 "$V2_ROOT/groups/direct_d64"

HISTORY_GROUP="$EXPERIMENT_ROOT/groups/history_baseline"
mkdir -p "$HISTORY_GROUP"
for seed in "${SEEDS[@]}"; do
    if [[ -d "$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed${seed}" ]]; then
        source="$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed${seed}"
    else
        source="$V2_ROOT/history_prior/seed${seed}"
    fi
    destination="$HISTORY_GROUP/seed${seed}"
    if [[ ! -e "$destination" && ! -L "$destination" ]]; then
        ln -s "$(readlink -f "$source")" "$destination"
    fi
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

run_stage() {
    local stage=$1
    local runner=$2
    shift 2
    local jobs=("$@")
    local total=${#jobs[@]}
    local next=0
    local -a available_slots=("${SLOTS[@]}")
    local -A pid_to_slot=()
    local -A pid_to_label=()
    echo "[conditional-v1] stage-start name=$stage jobs=$total slots=${#SLOTS[@]}"
    ACTIVE_PIDS=()
    while (( next < total || ${#ACTIVE_PIDS[@]} > 0 )); do
        while (( next < total && ${#available_slots[@]} > 0 )); do
            local slot_index=$((${#available_slots[@]} - 1))
            local gpu=${available_slots[$slot_index]}
            unset "available_slots[$slot_index]"
            local fields=()
            IFS='|' read -r -a fields <<< "${jobs[$next]}"
            bash "$runner" "$gpu" "${fields[@]}" &
            local pid=$!
            ACTIVE_PIDS+=("$pid")
            pid_to_slot[$pid]=$gpu
            pid_to_label[$pid]="${jobs[$next]}"
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$stage" "$next" "$gpu" "$pid" "${jobs[$next]}" >> "$JOBS_LOG"
            next=$((next + 1))
        done
        if (( ${#ACTIVE_PIDS[@]} > 0 )); then
            local finished_pid
            local status=0
            wait -n -p finished_pid "${ACTIVE_PIDS[@]}" || status=$?
            local finished_label=${pid_to_label[$finished_pid]}
            available_slots+=("${pid_to_slot[$finished_pid]}")
            unset "pid_to_slot[$finished_pid]" "pid_to_label[$finished_pid]"
            local -a remaining=()
            local active_pid
            for active_pid in "${ACTIVE_PIDS[@]}"; do
                if [[ "$active_pid" != "$finished_pid" ]]; then
                    remaining+=("$active_pid")
                fi
            done
            ACTIVE_PIDS=("${remaining[@]}")
            if (( status != 0 )); then
                echo "[conditional-v1] failed job=${finished_label}" >&2
                for active_pid in "${ACTIVE_PIDS[@]}"; do
                    kill -TERM "$active_pid" 2>/dev/null || true
                done
                wait 2>/dev/null || true
                ACTIVE_PIDS=()
                return "$status"
            fi
        fi
    done
    echo "[conditional-v1] stage-complete name=$stage"
}

TRAIN_JOBS=()
for seed in "${SEEDS[@]}"; do
    TRAIN_JOBS+=(
        "factor_context_only|${seed}|context_only|0.5|0.0|false"
        "factor_history_only|${seed}|history_only|0.5|0.0|false"
        "factor_weather_only|${seed}|weather_only|0.5|0.0|false"
        "anchor_005|${seed}|history_weather|0.0|0.05|true"
        "anchor_010|${seed}|history_weather|0.0|0.1|true"
        "anchor_020|${seed}|history_weather|0.0|0.2|true"
        "initialization_random|${seed}|history_weather|0.5|0.0|false|false"
    )
done
run_stage train "$PROJECT_ROOT/shell/run_conditional_forcing_validation_job.sh" "${TRAIN_JOBS[@]}"

EVAL_JOBS=()
FULL_ROOT="$V2_ROOT/groups/joint_full"
NO_AUX_ROOT="$V2_ROOT/groups/joint_no_baseline_aux"
for seed in "${SEEDS[@]}"; do
    EVAL_JOBS+=(
        "component|factor_history_weather|${seed}|${FULL_ROOT}/seed${seed}/config.yaml|${FULL_ROOT}/seed${seed}/best_model.pt"
        "component|anchor_000|${seed}|${NO_AUX_ROOT}/seed${seed}/config.yaml|${NO_AUX_ROOT}/seed${seed}/best_model.pt"
    )
done
run_stage components "$PROJECT_ROOT/shell/run_conditional_forcing_evaluation_job.sh" "${EVAL_JOBS[@]}"

SHUFFLE_JOBS=()
for permutation_seed in "${PERMUTATION_SEEDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        SHUFFLE_JOBS+=(
            "shuffle|factor_history_weather|${seed}|${FULL_ROOT}/seed${seed}/config.yaml|${FULL_ROOT}/seed${seed}/best_model.pt|${permutation_seed}"
        )
    done
done
run_stage weather_shuffle "$PROJECT_ROOT/shell/run_conditional_forcing_evaluation_job.sh" "${SHUFFLE_JOBS[@]}"

touch "$EXPERIMENT_ROOT/ALL_COMPLETE"
echo "[conditional-v1] all complete root=$EXPERIMENT_ROOT"
