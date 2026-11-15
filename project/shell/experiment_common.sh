#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$COMMON_DIR/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
BASE_CONFIG=${WIND_BASE_CONFIG:-$PROJECT_ROOT/configs/forced_evolution_pad5.yaml}
EXPERIMENT_ROOT=${WIND_EXPERIMENT_ROOT:-$PROJECT_ROOT/artifacts/experiments_pad5_cells}
WAIT_TIMEOUT_SECONDS=${WIND_WAIT_TIMEOUT_SECONDS:-259200}
WAIT_POLL_SECONDS=${WIND_WAIT_POLL_SECONDS:-30}
SMOKE_MODE=${WIND_SMOKE:-0}
LANE_ID=${WIND_LANE_ID:-manual}
LANE_PREFIX="[lane-$LANE_ID]"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

mkdir -p "$EXPERIMENT_ROOT" "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/lanes"

initialize_lane() {
    local gpu_id=$1
    if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
        echo "A lane requires exactly one physical GPU index, got: $gpu_id" >&2
        exit 2
    fi
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "Missing Python interpreter: $PYTHON_BIN" >&2
        exit 2
    fi
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    echo "$LANE_PREFIX start gpu=$gpu_id root=$EXPERIMENT_ROOT smoke=$SMOKE_MODE"
    "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this lane")
print(f"[gpu] {torch.cuda.get_device_name(0)}")
PY
    if [[ ! -f "$PROJECT_ROOT/data/model_cache_pad5_f8/manifest.json" ]]; then
        echo "Prepared pad5 cache is missing; run scripts/prepare_model_data.sh first" >&2
        exit 2
    fi
}

wait_for_file() {
    local target=$1
    local label=$2
    local started=$SECONDS
    while [[ ! -f "$target" ]]; do
        if (( SECONDS - started >= WAIT_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for $label: $target" >&2
            exit 3
        fi
        echo "$LANE_PREFIX wait label=$label elapsed=$((SECONDS - started))s target=$target"
        sleep "$WAIT_POLL_SECONDS"
    done
}

materialize_config() {
    local experiment_id=$1
    local base_config=$2
    local output_config=$3
    shift 3
    local extra_overrides=()
    if [[ "$SMOKE_MODE" == "1" ]]; then
        extra_overrides=(data.preload_weather=mmap)
    fi
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
        --base "$base_config" \
        --output "$output_config" \
        --experiment-id "$experiment_id" \
        "$@" \
        "${extra_overrides[@]}"
}

training_limit_args() {
    if [[ "$SMOKE_MODE" == "1" ]]; then
        printf '%s\n' --epochs 1 --max-train-batches 1 --max-val-batches 1
    fi
}

evaluation_limit_args() {
    if [[ "$SMOKE_MODE" == "1" ]]; then
        printf '%s\n' --max-batches 1
    fi
}

run_spatial_experiment_from_base() {
    local experiment_id=$1
    local base_config=$2
    shift 2
    local run_dir="$EXPERIMENT_ROOT/spatial/$experiment_id"
    local complete="$run_dir/COMPLETE"
    mkdir -p "$run_dir"
    local lock_fd
    exec {lock_fd}>"$run_dir/.run.lock"
    flock -x "$lock_fd"
    if ! materialize_config "$experiment_id" "$base_config" "$run_dir/config.yaml" "$@"; then
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        return 1
    fi
    if [[ -f "$complete" ]]; then
        echo "[skip] spatial/$experiment_id"
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        return
    fi
    echo "$LANE_PREFIX experiment-start stage=spatial id=$experiment_id output=$run_dir"
    mapfile -t limit_args < <(training_limit_args)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_spatial_pretrain.py" \
        --config "$run_dir/config.yaml" \
        --output-dir "$run_dir" \
        --device cuda \
        "${limit_args[@]}" 2>&1 | tee -a "$run_dir/run.log"
    mapfile -t eval_args < <(evaluation_limit_args)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_spatial_pretrain.py" \
        --config "$run_dir/config.yaml" \
        --checkpoint "$run_dir/best_model.pt" \
        --split val \
        --output "$run_dir/metrics_val.json" \
        --device cuda \
        "${eval_args[@]}" 2>&1 | tee -a "$run_dir/run.log"
    touch "$complete"
    echo "$LANE_PREFIX experiment-complete stage=spatial id=$experiment_id"
    flock -u "$lock_fd"
    exec {lock_fd}>&-
}

run_spatial_experiment() {
    local experiment_id=$1
    shift
    run_spatial_experiment_from_base "$experiment_id" "$BASE_CONFIG" "$@"
}

ensure_local_selection() {
    local candidates=(s1_learned_cell1 s2_learned_cell3 s3_learned_cell5)
    local candidate
    for candidate in "${candidates[@]}"; do
        wait_for_file "$EXPERIMENT_ROOT/spatial/$candidate/COMPLETE" "spatial/$candidate"
    done
    local selected="$EXPERIMENT_ROOT/selected/local"
    mkdir -p "$selected"
    (
        flock -x 9
        if [[ ! -f "$selected/COMPLETE" ]]; then
            "$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_experiment.py" \
                --output-dir "$selected" \
                --metric cluster_mae_mw \
                "${candidates[@]/#/$EXPERIMENT_ROOT/spatial/}"
        fi
    ) 9>"$selected/.selection.lock"
}

ensure_spatial_selection() {
    ensure_local_selection
    local candidates=(
        "$EXPERIMENT_ROOT/selected/local"
        "$EXPERIMENT_ROOT/spatial/s4_selected_local_wind"
        "$EXPERIMENT_ROOT/spatial/s5_selected_local_wind_context"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        wait_for_file "$candidate/COMPLETE" "spatial candidate $(basename "$candidate")"
    done
    local selected="$EXPERIMENT_ROOT/selected/spatial"
    mkdir -p "$selected"
    (
        flock -x 9
        if [[ ! -f "$selected/COMPLETE" ]]; then
            "$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_experiment.py" \
                --output-dir "$selected" \
                --metric cluster_mae_mw \
                "${candidates[@]}"
        fi
    ) 9>"$selected/.selection.lock"
}

run_forecast_experiment() {
    local category=$1
    local experiment_id=$2
    local base_config=$3
    local spatial_checkpoint=$4
    shift 4
    local run_dir="$EXPERIMENT_ROOT/$category/$experiment_id"
    local complete="$run_dir/COMPLETE"
    mkdir -p "$run_dir"
    local lock_fd
    exec {lock_fd}>"$run_dir/.run.lock"
    flock -x "$lock_fd"
    if ! materialize_config "$experiment_id" "$base_config" "$run_dir/config.yaml" "$@"; then
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        return 1
    fi
    if [[ -f "$complete" ]]; then
        echo "[skip] $category/$experiment_id"
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        return
    fi
    echo "$LANE_PREFIX experiment-start stage=$category id=$experiment_id output=$run_dir"
    mapfile -t limit_args < <(training_limit_args)
    local checkpoint_args=()
    if [[ -n "$spatial_checkpoint" ]]; then
        checkpoint_args=(
            --spatial-checkpoint "$spatial_checkpoint"
            --spatial-load-mode spatial_only
        )
    fi
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_forced_evolution.py" \
        --config "$run_dir/config.yaml" \
        --output-dir "$run_dir" \
        --device cuda \
        "${checkpoint_args[@]}" \
        "${limit_args[@]}" 2>&1 | tee -a "$run_dir/run.log"
    mapfile -t eval_args < <(evaluation_limit_args)
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
        --config "$run_dir/config.yaml" \
        --checkpoint "$run_dir/best_model.pt" \
        --split val \
        --output "$run_dir/metrics_val.json" \
        --predictions-output "$run_dir/predictions_val.npz" \
        --device cuda \
        "${eval_args[@]}" 2>&1 | tee -a "$run_dir/run.log"
    touch "$complete"
    echo "$LANE_PREFIX experiment-complete stage=$category id=$experiment_id"
    flock -u "$lock_fd"
    exec {lock_fd}>&-
}

ensure_temporal_selection() {
    ensure_spatial_selection
    local candidates=(f0_independent f1_difference f2_tcn f3_transformer)
    local candidate
    for candidate in "${candidates[@]}"; do
        wait_for_file "$EXPERIMENT_ROOT/temporal/$candidate/COMPLETE" "temporal/$candidate"
    done
    local selected="$EXPERIMENT_ROOT/selected/temporal"
    mkdir -p "$selected"
    (
        flock -x 9
        if [[ ! -f "$selected/COMPLETE" ]]; then
            "$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_experiment.py" \
                --output-dir "$selected" \
                --metric cluster_mae_mw \
                "${candidates[@]/#/$EXPERIMENT_ROOT/temporal/}"
        fi
    ) 9>"$selected/.selection.lock"
}

evaluate_test_once() {
    local run_dir=$1
    local export_stations=${2:-0}
    local lock_fd
    exec {lock_fd}>"$run_dir/.test.lock"
    flock -x "$lock_fd"
    if [[ -f "$run_dir/TEST_COMPLETE" ]]; then
        echo "[skip] test $(basename "$run_dir")"
        flock -u "$lock_fd"
        exec {lock_fd}>&-
        return
    fi
    mapfile -t eval_args < <(evaluation_limit_args)
    local station_args=()
    if [[ "$export_stations" == "1" ]]; then
        station_args=(--export-station-predictions)
    fi
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
        --config "$run_dir/config.yaml" \
        --checkpoint "$run_dir/best_model.pt" \
        --split test \
        --output "$run_dir/metrics_test.json" \
        --predictions-output "$run_dir/predictions_test.npz" \
        --device cuda \
        "${station_args[@]}" \
        "${eval_args[@]}" 2>&1 | tee -a "$run_dir/run.log"
    touch "$run_dir/TEST_COMPLETE"
    flock -u "$lock_fd"
    exec {lock_fd}>&-
}

promote_selected_seed42() {
    ensure_temporal_selection
    local source="$EXPERIMENT_ROOT/selected/temporal"
    local run_dir="$EXPERIMENT_ROOT/final/seed42"
    mkdir -p "$run_dir"
    if [[ ! -f "$run_dir/COMPLETE" ]]; then
        materialize_config final_seed42 "$source/config.yaml" "$run_dir/config.yaml" "seed=42"
        cp "$source/best_model.pt" "$run_dir/best_model.pt"
        cp "$source/history.json" "$run_dir/history.json"
        cp "$source/metrics_val.json" "$run_dir/metrics_val.json"
        touch "$run_dir/COMPLETE"
    fi
    evaluate_test_once "$run_dir"
}

run_final_seed() {
    local seed=$1
    ensure_temporal_selection
    local spatial="$EXPERIMENT_ROOT/selected/spatial/best_model.pt"
    local base="$EXPERIMENT_ROOT/selected/temporal/config.yaml"
    run_forecast_experiment final "seed$seed" "$base" "$spatial" "seed=$seed"
    evaluate_test_once "$EXPERIMENT_ROOT/final/seed$seed"
}

mark_lane_complete() {
    local lane=$1
    touch "$EXPERIMENT_ROOT/lanes/$lane.COMPLETE"
    echo "$LANE_PREFIX complete $lane"
}
