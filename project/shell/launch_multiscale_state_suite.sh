#!/usr/bin/env bash
set -euo pipefail

SHELL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SHELL_DIR/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
BASE_CONFIG=${WIND_MULTISCALE_STATE_CONFIG:-$PROJECT_ROOT/configs/multiscale_state_pad5.yaml}
PAPER_ROOT=${WIND_MULTISCALE_STATE_PAPER_ROOT:-$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1}
FINAL_ROUTE_ROOT=${WIND_MULTISCALE_STATE_FINAL_ROOT:-$PROJECT_ROOT/artifacts/final_route_pad5_v1}
EXPERIMENT_ROOT=${WIND_MULTISCALE_STATE_ROOT:-$PROJECT_ROOT/artifacts/multiscale_state_pad5_v1}
GPU_CSV=${1:-3,4,5,6,7}
SEED_CSV=${WIND_MULTISCALE_STATE_SEEDS:-42,43,44}
PILOT_SEED=${WIND_MULTISCALE_STATE_PILOT_SEED:-42}
WAIT_TIMEOUT_SECONDS=${WIND_WAIT_TIMEOUT_SECONDS:-259200}
WAIT_POLL_SECONDS=${WIND_WAIT_POLL_SECONDS:-30}
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"

if [[ ${#GPUS[@]} -lt 1 ]]; then
    echo "At least one GPU index is required" >&2
    exit 2
fi
if [[ ${#SEEDS[@]} -ne 3 ]]; then
    echo "WIND_MULTISCALE_STATE_SEEDS must contain exactly three seeds" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python interpreter: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Missing base config: $BASE_CONFIG" >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WIND_EXPERIMENT_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/selected" \
    "$EXPERIMENT_ROOT/analysis" "$EXPERIMENT_ROOT/summary"
JOBS_FILE="$EXPERIMENT_ROOT/logs/multiscale_state_jobs.tsv"
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
        echo "[multiscale-state] waiting label=$label elapsed=$((SECONDS - started))s"
        sleep "$WAIT_POLL_SECONDS"
    done
}

wait_for_file "$PAPER_ROOT/ALL_COMPLETE" "formal paper suite"
wait_for_file "$FINAL_ROUTE_ROOT/ALL_COMPLETE" "final-route suite"

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
    echo "[multiscale-state] stage-start name=$stage jobs=$total"
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
            WIND_LANE_ID="multiscale-state-$stage-$index" \
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
                echo "[multiscale-state] failed job=${labels[$offset]}" >&2
                failed=1
            fi
        done
        ACTIVE_PIDS=()
        if (( failed != 0 )); then return 1; fi
        start=$end
    done
    echo "[multiscale-state] stage-complete name=$stage"
}

common_overrides=(
    model.architecture=controlled_transition
    model.transition_type=gru
    model.multirate_enabled=false
    model.output_level=hierarchical
    model.include_wind_geometry=true
    model.include_region_context=false
    model.forcing_path_mode=difference
    model.time_query_mode=full_attention
    training.deterministic.auxiliary_weather_loss_weight=0.2
    training.deterministic.multirate_target_loss_weight=0.0
    training.deterministic.multirate_consistency_loss_weight=0.0
)

# Stage 1: one-seed spatial-scale screening. The local-only model remains eligible.
JOBS=()
add_job forecast pilot_spatial s0_local "$BASE_CONFIG" NONE \
    "seed=$PILOT_SEED" "${common_overrides[@]}" \
    model.regional_pool_size=1 model.regional_neighborhood_cells=1 \
    training.deterministic.ramp_loss_weight=0.0
add_job forecast pilot_spatial s1_pool2_cell1 "$BASE_CONFIG" NONE \
    "seed=$PILOT_SEED" "${common_overrides[@]}" \
    model.regional_pool_size=2 model.regional_neighborhood_cells=1 \
    training.deterministic.ramp_loss_weight=0.0
add_job forecast pilot_spatial s2_pool2_cell3 "$BASE_CONFIG" NONE \
    "seed=$PILOT_SEED" "${common_overrides[@]}" \
    model.regional_pool_size=2 model.regional_neighborhood_cells=3 \
    training.deterministic.ramp_loss_weight=0.0
add_job forecast pilot_spatial s3_pool3_cell1 "$BASE_CONFIG" NONE \
    "seed=$PILOT_SEED" "${common_overrides[@]}" \
    model.regional_pool_size=3 model.regional_neighborhood_cells=1 \
    training.deterministic.ramp_loss_weight=0.0
run_stage pilot_spatial "${JOBS[@]}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_experiment.py" \
    --output-dir "$EXPERIMENT_ROOT/selected/spatial_pilot" \
    --metric cluster_mae_mw \
    "$EXPERIMENT_ROOT/pilot_spatial/s0_local" \
    "$EXPERIMENT_ROOT/pilot_spatial/s1_pool2_cell1" \
    "$EXPERIMENT_ROOT/pilot_spatial/s2_pool2_cell3" \
    "$EXPERIMENT_ROOT/pilot_spatial/s3_pool3_cell1"

# Stage 2: select quarter-hour change supervision after the spatial choice is frozen.
RAMP_BASE="$EXPERIMENT_ROOT/selected/spatial_pilot/config.yaml"
JOBS=()
for weight in 0.1 0.2 0.3; do
    label=${weight/./p}
    add_job forecast pilot_ramp "r${label}" "$RAMP_BASE" NONE \
        "seed=$PILOT_SEED" "training.deterministic.ramp_loss_weight=$weight"
done
run_stage pilot_ramp "${JOBS[@]}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_experiment.py" \
    --output-dir "$EXPERIMENT_ROOT/selected/ramp_pilot" \
    --metric cluster_mae_mw \
    "$EXPERIMENT_ROOT/selected/spatial_pilot" \
    "$EXPERIMENT_ROOT/pilot_ramp/r0p1" \
    "$EXPERIMENT_ROOT/pilot_ramp/r0p2" \
    "$EXPERIMENT_ROOT/pilot_ramp/r0p3"

# Stage 3: compact capacity/dropout screening around the selected mechanism.
TUNING_BASE="$EXPERIMENT_ROOT/selected/ramp_pilot/config.yaml"
JOBS=()
add_job forecast pilot_capacity d64_drop005 "$TUNING_BASE" NONE \
    "seed=$PILOT_SEED" model.d_model=64 model.dropout=0.05
add_job forecast pilot_capacity d96_drop005 "$TUNING_BASE" NONE \
    "seed=$PILOT_SEED" model.d_model=96 model.dropout=0.05
add_job forecast pilot_capacity d96_drop010 "$TUNING_BASE" NONE \
    "seed=$PILOT_SEED" model.d_model=96 model.dropout=0.1
add_job forecast pilot_capacity d128_drop005 "$TUNING_BASE" NONE \
    "seed=$PILOT_SEED" model.d_model=128 model.dropout=0.05
add_job forecast pilot_capacity d128_drop010 "$TUNING_BASE" NONE \
    "seed=$PILOT_SEED" model.d_model=128 model.dropout=0.1
run_stage pilot_capacity "${JOBS[@]}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_experiment.py" \
    --output-dir "$EXPERIMENT_ROOT/selected/tuning_pilot" \
    --metric cluster_mae_mw \
    "$EXPERIMENT_ROOT/selected/ramp_pilot" \
    "$EXPERIMENT_ROOT/pilot_capacity/d64_drop005" \
    "$EXPERIMENT_ROOT/pilot_capacity/d96_drop005" \
    "$EXPERIMENT_ROOT/pilot_capacity/d96_drop010" \
    "$EXPERIMENT_ROOT/pilot_capacity/d128_drop005" \
    "$EXPERIMENT_ROOT/pilot_capacity/d128_drop010"

# Stage 4: three-seed confirmation and matched removal controls.
CONFIRM_BASE="$EXPERIMENT_ROOT/selected/tuning_pilot/config.yaml"
JOBS=()
for seed in "${SEEDS[@]}"; do
    add_job forecast confirmation "main_state/seed$seed" "$CONFIRM_BASE" NONE "seed=$seed"
    add_job forecast confirmation "matched_parallel/seed$seed" "$CONFIRM_BASE" NONE \
        "seed=$seed" model.architecture=forced_evolution
    add_job forecast confirmation "no_multiscale/seed$seed" "$CONFIRM_BASE" NONE \
        "seed=$seed" model.regional_pool_size=1 model.regional_neighborhood_cells=1
    add_job forecast confirmation "no_ramp_loss/seed$seed" "$CONFIRM_BASE" NONE \
        "seed=$seed" training.deterministic.ramp_loss_weight=0.0
done
run_stage confirmation "${JOBS[@]}"

# State recurrence is not forced to win: validation mean chooses the frozen test model.
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_group.py" \
    --output-dir "$EXPERIMENT_ROOT/selected/model" \
    --metric cluster_mae_mw \
    --seeds "$SEED_CSV" \
    "$EXPERIMENT_ROOT/confirmation/main_state" \
    "$EXPERIMENT_ROOT/confirmation/matched_parallel"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_multiscale_state.py" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --final-route-root "$FINAL_ROUTE_ROOT" \
    --paper-root "$PAPER_ROOT" \
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
echo "[multiscale-state] all complete root=$EXPERIMENT_ROOT"
