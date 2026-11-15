#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_INCREMENTAL_V2_ROOT:-${PROJECT_ROOT}/artifacts/incremental_mechanism_v2}"
SEED_CSV="${WIND_INCREMENTAL_V2_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
SLOT_CSV="${1:-0,0,0,1,1,1,3,3,3,4,5,6,7}"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
IFS=',' read -r -a SLOTS <<< "$SLOT_CSV"

if [[ ${#SEEDS[@]} -ne 10 ]]; then
    echo "WIND_INCREMENTAL_V2_SEEDS must contain exactly ten seeds" >&2
    exit 2
fi
if [[ ${#SLOTS[@]} -lt 1 ]]; then
    echo "At least one GPU slot is required" >&2
    exit 2
fi

export WIND_INCREMENTAL_V2_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/logs"
JOBS_LOG="$EXPERIMENT_ROOT/logs/jobs.tsv"
printf 'stage\tindex\tgpu\tpid\tjob\n' > "$JOBS_LOG"

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
    shift
    local jobs=("$@")
    local total=${#jobs[@]}
    local start=0
    echo "[incremental-v2] stage-start name=$stage jobs=$total slots=${#SLOTS[@]}"
    while (( start < total )); do
        ACTIVE_PIDS=()
        local labels=()
        local end=$((start + ${#SLOTS[@]}))
        if (( end > total )); then end=$total; fi
        local index
        for ((index=start; index<end; index++)); do
            local gpu=${SLOTS[$((index - start))]}
            local fields=()
            IFS='|' read -r -a fields <<< "${jobs[$index]}"
            bash "$PROJECT_ROOT/shell/run_incremental_mechanism_v2_job.sh" \
                "$gpu" "${fields[@]}" &
            local pid=$!
            ACTIVE_PIDS+=("$pid")
            labels+=("${jobs[$index]}")
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$stage" "$index" "$gpu" "$pid" "${jobs[$index]}" >> "$JOBS_LOG"
        done
        local failed=0
        local offset
        for offset in "${!ACTIVE_PIDS[@]}"; do
            if ! wait "${ACTIVE_PIDS[$offset]}"; then
                echo "[incremental-v2] failed job=${labels[$offset]}" >&2
                failed=1
            fi
        done
        ACTIVE_PIDS=()
        if (( failed != 0 )); then return 1; fi
        start=$end
    done
    echo "[incremental-v2] stage-complete name=$stage"
}

# Seven additional history priors are required by the new seeds. The legacy
# checkpoints for 42--44 are deliberately reused.
HISTORY_JOBS=()
for seed in "${SEEDS[@]}"; do
    if [[ ! -f "$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed${seed}/best_model.pt" ]]; then
        HISTORY_JOBS+=("history|history_prior|${seed}")
    fi
done
run_stage history "${HISTORY_JOBS[@]}"

MODEL_JOBS=()
for seed in "${SEEDS[@]}"; do
    MODEL_JOBS+=(
        "direct|direct_d64|${seed}"
        "direct|direct_d72_capacity|${seed}|model.d_model=72"
        "incremental|joint_full|${seed}|model.freeze_history_baseline=false|model.projection_mode=full|model.temporal_components=24|model.correction_conditioning=history_weather|training.deterministic.baseline_auxiliary_loss_weight=0.5"
        "incremental|joint_dct4|${seed}|model.freeze_history_baseline=false|model.projection_mode=temporal|model.temporal_components=4|model.correction_conditioning=history_weather|training.deterministic.baseline_auxiliary_loss_weight=0.5"
        "incremental|joint_dct7|${seed}|model.freeze_history_baseline=false|model.projection_mode=temporal|model.temporal_components=7|model.correction_conditioning=history_weather|training.deterministic.baseline_auxiliary_loss_weight=0.5"
        "incremental|joint_dct8|${seed}|model.freeze_history_baseline=false|model.projection_mode=temporal|model.temporal_components=8|model.correction_conditioning=history_weather|training.deterministic.baseline_auxiliary_loss_weight=0.5"
        "incremental|joint_dct12|${seed}|model.freeze_history_baseline=false|model.projection_mode=temporal|model.temporal_components=12|model.correction_conditioning=history_weather|training.deterministic.baseline_auxiliary_loss_weight=0.5"
        "incremental|joint_weather_only|${seed}|model.freeze_history_baseline=false|model.projection_mode=full|model.temporal_components=24|model.correction_conditioning=weather_only|training.deterministic.baseline_auxiliary_loss_weight=0.5"
        "incremental|joint_no_baseline_aux|${seed}|model.freeze_history_baseline=false|model.projection_mode=full|model.temporal_components=24|model.correction_conditioning=history_weather|training.deterministic.baseline_auxiliary_loss_weight=0.0"
    )
done
run_stage models "${MODEL_JOBS[@]}"
touch "$EXPERIMENT_ROOT/ALL_COMPLETE"
echo "[incremental-v2] all complete root=$EXPERIMENT_ROOT"
