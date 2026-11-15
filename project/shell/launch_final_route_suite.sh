#!/usr/bin/env bash
set -euo pipefail

SHELL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SHELL_DIR/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
TEMPLATE_CONFIG=${WIND_FINAL_ROUTE_TEMPLATE:-$PROJECT_ROOT/configs/controlled_transition_pad5.yaml}
PAPER_ROOT=${WIND_FINAL_ROUTE_PAPER_ROOT:-$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1}
TRANSITION_ROOT=${WIND_FINAL_ROUTE_TRANSITION_ROOT:-$PROJECT_ROOT/artifacts/controlled_transition_pad5_v1}
EXPERIMENT_ROOT=${WIND_FINAL_ROUTE_ROOT:-$PROJECT_ROOT/artifacts/final_route_pad5_v1}
GPU_CSV=${1:-5,6,7}
SEED_CSV=${WIND_FINAL_ROUTE_SEEDS:-42,43,44}
WAIT_TIMEOUT_SECONDS=${WIND_WAIT_TIMEOUT_SECONDS:-259200}
WAIT_POLL_SECONDS=${WIND_WAIT_POLL_SECONDS:-30}
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"

if [[ ${#GPUS[@]} -lt 1 ]]; then
    echo "At least one GPU index is required" >&2
    exit 2
fi
if [[ ${#SEEDS[@]} -ne 3 ]]; then
    echo "WIND_FINAL_ROUTE_SEEDS must contain exactly three seeds" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python interpreter: $PYTHON_BIN" >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WIND_EXPERIMENT_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/base" \
    "$EXPERIMENT_ROOT/selected" "$EXPERIMENT_ROOT/analysis"
JOBS_FILE="$EXPERIMENT_ROOT/logs/final_route_jobs.tsv"
printf 'stage\tindex\tgpu\tpid\tjob\n' > "$JOBS_FILE"

wait_for_file() {
    local target=$1
    local label=$2
    local started=$SECONDS
    while [[ ! -f "$target" ]]; do
        if (( SECONDS - started >= WAIT_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for $label: $target" >&2
            exit 3
        fi
        echo "[final-route] waiting label=$label elapsed=$((SECONDS - started))s"
        sleep "$WAIT_POLL_SECONDS"
    done
}

wait_for_file "$PAPER_ROOT/ALL_COMPLETE" "formal paper suite"
wait_for_file "$TRANSITION_ROOT/ALL_COMPLETE" "controlled-transition suite"
for seed in "${SEEDS[@]}"; do
    source_config="$PAPER_ROOT/selected/training/seed$seed/config.yaml"
    test -f "$source_config" || {
        echo "Missing validation-selected training config: $source_config" >&2
        exit 2
    }
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_transition_base.py" \
        --source "$source_config" \
        --template "$TEMPLATE_CONFIG" \
        --seed "$seed" \
        --auxiliary-weather-loss-weight 0.2 \
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
    echo "[final-route] stage-start name=$stage jobs=$total"
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
            WIND_LANE_ID="final-route-$stage-$index" \
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
                echo "[final-route] failed job=${labels[$offset]}" >&2
                failed=1
            fi
        done
        ACTIVE_PIDS=()
        if (( failed != 0 )); then return 1; fi
        start=$end
    done
    echo "[final-route] stage-complete name=$stage"
}

common_overrides=(
    model.architecture=controlled_transition
    model.transition_type=gru
    model.multirate_enabled=false
    model.output_level=hierarchical
    training.deterministic.auxiliary_weather_loss_weight=0.2
    training.deterministic.multirate_target_loss_weight=0.0
    training.deterministic.multirate_consistency_loss_weight=0.0
)

JOBS=()
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/base/seed$seed.yaml"
    add_job forecast candidates "m0_controlled_full/seed$seed" "$base" NONE \
        "seed=$seed" "${common_overrides[@]}"
    add_job forecast candidates "s0_no_wind_geometry/seed$seed" "$base" NONE \
        "seed=$seed" "${common_overrides[@]}" \
        model.include_wind_geometry=false
    add_job forecast candidates "s1_fixed_bilinear/seed$seed" "$base" NONE \
        "seed=$seed" "${common_overrides[@]}" \
        model.station_query_mode=bilinear \
        model.station_neighborhood_cells=1 \
        model.include_wind_geometry=false
    add_job forecast candidates "t0_independent_weather/seed$seed" "$base" NONE \
        "seed=$seed" "${common_overrides[@]}" \
        model.forcing_path_mode=independent
    add_job forecast candidates "t1_nearest_hour/seed$seed" "$base" NONE \
        "seed=$seed" "${common_overrides[@]}" \
        model.time_query_mode=nearest
done
run_stage route_training "${JOBS[@]}"

# Only the two complete main-model candidates participate in model selection.
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_group.py" \
    --output-dir "$EXPERIMENT_ROOT/selected/model" \
    --metric cluster_mae_mw \
    --seeds "$SEED_CSV" \
    "$PAPER_ROOT/controls/scratch_aux" \
    "$EXPERIMENT_ROOT/candidates/m0_controlled_full"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_final_route.py" \
    --final-root "$EXPERIMENT_ROOT" \
    --paper-root "$PAPER_ROOT" \
    --transition-root "$TRANSITION_ROOT" \
    --output-dir "$EXPERIMENT_ROOT/analysis" \
    --seeds "$SEED_CSV"

JOBS=()
for seed in "${SEEDS[@]}"; do
    add_job test selected/model "seed$seed" NONE NONE 1
done
run_stage selected_test "${JOBS[@]}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/summarize_experiments.py" \
    --root "$EXPERIMENT_ROOT" --output-dir "$EXPERIMENT_ROOT/summary"
touch "$EXPERIMENT_ROOT/ALL_COMPLETE"
echo "[final-route] all complete root=$EXPERIMENT_ROOT"
