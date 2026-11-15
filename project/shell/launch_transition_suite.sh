#!/usr/bin/env bash
set -euo pipefail

SHELL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SHELL_DIR/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
TEMPLATE_CONFIG=${WIND_TRANSITION_TEMPLATE:-$PROJECT_ROOT/configs/controlled_transition_pad5.yaml}
SOURCE_PAPER_ROOT=${WIND_TRANSITION_SOURCE_ROOT:-$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1}
EXPERIMENT_ROOT=${WIND_TRANSITION_ROOT:-$PROJECT_ROOT/artifacts/controlled_transition_pad5_v1}
GPU_CSV=${1:-0,1,2,3,4,5,6,7}
SEED_CSV=${WIND_TRANSITION_SEEDS:-42,43,44}
WAIT_TIMEOUT_SECONDS=${WIND_WAIT_TIMEOUT_SECONDS:-259200}
WAIT_POLL_SECONDS=${WIND_WAIT_POLL_SECONDS:-30}
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"

if [[ ${#GPUS[@]} -lt 1 ]]; then
    echo "At least one GPU index is required" >&2
    exit 2
fi
if [[ ${#SEEDS[@]} -ne 3 ]]; then
    echo "WIND_TRANSITION_SEEDS must contain exactly three seeds" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python interpreter: $PYTHON_BIN" >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WIND_EXPERIMENT_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/base" "$EXPERIMENT_ROOT/selected"
JOBS_FILE="$EXPERIMENT_ROOT/logs/transition_jobs.tsv"
printf 'stage\tindex\tgpu\tpid\tjob\n' > "$JOBS_FILE"

wait_for_source() {
    local target=$1
    local started=$SECONDS
    while [[ ! -f "$target" ]]; do
        if (( SECONDS - started >= WAIT_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for selected time-alignment config: $target" >&2
            exit 3
        fi
        echo "[transition] waiting elapsed=$((SECONDS - started))s target=$target"
        sleep "$WAIT_POLL_SECONDS"
    done
}

source_config_for_seed() {
    local seed=$1
    if [[ -n ${WIND_TRANSITION_SOURCE_CONFIG:-} ]]; then
        printf '%s\n' "$WIND_TRANSITION_SOURCE_CONFIG"
    else
        printf '%s\n' "$SOURCE_PAPER_ROOT/selected/time_alignment/seed$seed/config.yaml"
    fi
}

if [[ -z ${WIND_TRANSITION_SOURCE_CONFIG:-} ]]; then
    wait_for_source "$SOURCE_PAPER_ROOT/selected/time_alignment/COMPLETE"
fi
for seed in "${SEEDS[@]}"; do
    source_config=$(source_config_for_seed "$seed")
    test -f "$source_config" || {
        echo "Missing source config for seed $seed: $source_config" >&2
        exit 2
    }
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_transition_base.py" \
        --source "$source_config" --template "$TEMPLATE_CONFIG" --seed "$seed" \
        --output "$EXPERIMENT_ROOT/base/seed$seed.yaml"
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

JOBS=()
add_job() {
    local IFS='|'
    JOBS+=("$*")
}

run_stage() {
    local stage=$1
    shift
    local jobs=("$@")
    local total=${#jobs[@]}
    local start=0
    echo "[transition] stage-start name=$stage jobs=$total"
    while (( start < total )); do
        ACTIVE_PIDS=()
        local labels=()
        local end=$((start + ${#GPUS[@]}))
        if (( end > total )); then end=$total; fi
        local index
        for ((index=start; index<end; index++)); do
            local gpu=${GPUS[$((index - start))]}
            local fields=()
            IFS='|' read -r -a fields <<< "${jobs[$index]}"
            WIND_LANE_ID="transition-$stage-$index" \
            WIND_PROGRESS_PREFIX="[$stage:$index]" \
            WIND_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
            bash "$SHELL_DIR/run_paper_job.sh" "$gpu" "${fields[@]}" &
            local pid=$!
            ACTIVE_PIDS+=("$pid")
            labels+=("${jobs[$index]}")
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$stage" "$index" "$gpu" "$pid" "${jobs[$index]}" >> "$JOBS_FILE"
        done
        local failed=0
        local offset
        for offset in "${!ACTIVE_PIDS[@]}"; do
            if ! wait "${ACTIVE_PIDS[$offset]}"; then
                echo "[transition] failed job=${labels[$offset]}" >&2
                failed=1
            fi
        done
        ACTIVE_PIDS=()
        if (( failed != 0 )); then return 1; fi
        start=$end
    done
    echo "[transition] stage-complete name=$stage"
}

candidate_ids=(
    r0_current_parallel
    r1_controlled_gru
    r2_selective_ssm
    r3_controlled_koopman
    r4_selective_ssm_multirate
)
JOBS=()
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/base/seed$seed.yaml"
    add_job forecast transition "r0_current_parallel/seed$seed" "$base" NONE \
        "seed=$seed" model.architecture=forced_evolution \
        model.fusion_mode=state_conditioned model.multirate_enabled=false \
        training.deterministic.multirate_target_loss_weight=0.0 \
        training.deterministic.multirate_consistency_loss_weight=0.0
    add_job forecast transition "r1_controlled_gru/seed$seed" "$base" NONE \
        "seed=$seed" model.architecture=controlled_transition \
        model.transition_type=gru model.multirate_enabled=false \
        training.deterministic.multirate_target_loss_weight=0.0 \
        training.deterministic.multirate_consistency_loss_weight=0.0
    add_job forecast transition "r2_selective_ssm/seed$seed" "$base" NONE \
        "seed=$seed" model.architecture=controlled_transition \
        model.transition_type=selective_ssm model.multirate_enabled=false \
        training.deterministic.multirate_target_loss_weight=0.0 \
        training.deterministic.multirate_consistency_loss_weight=0.0
    add_job forecast transition "r3_controlled_koopman/seed$seed" "$base" NONE \
        "seed=$seed" model.architecture=controlled_transition \
        model.transition_type=controlled_koopman model.multirate_enabled=false \
        training.deterministic.multirate_target_loss_weight=0.0 \
        training.deterministic.multirate_consistency_loss_weight=0.0
    add_job forecast transition "r4_selective_ssm_multirate/seed$seed" "$base" NONE \
        "seed=$seed" model.architecture=controlled_transition \
        model.transition_type=selective_ssm model.multirate_enabled=true \
        training.deterministic.multirate_target_loss_weight=0.25 \
        training.deterministic.multirate_consistency_loss_weight=0.1
done
run_stage backbone_screen "${JOBS[@]}"

candidates=()
for id in "${candidate_ids[@]}"; do
    candidates+=("$EXPERIMENT_ROOT/transition/$id")
done
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_group.py" \
    --output-dir "$EXPERIMENT_ROOT/selected/model" \
    --metric cluster_mae_mw --seeds "$SEED_CSV" "${candidates[@]}"

JOBS=()
for seed in "${SEEDS[@]}"; do
    add_job test selected/model "seed$seed" NONE NONE 1
done
run_stage selected_test "${JOBS[@]}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/summarize_experiments.py" \
    --root "$EXPERIMENT_ROOT" --output-dir "$EXPERIMENT_ROOT/summary"
touch "$EXPERIMENT_ROOT/ALL_COMPLETE"
echo "[transition] all complete root=$EXPERIMENT_ROOT"
