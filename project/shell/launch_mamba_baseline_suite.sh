#!/usr/bin/env bash
set -euo pipefail

SHELL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SHELL_DIR/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
TEMPLATE_CONFIG=${WIND_MAMBA_TEMPLATE:-$PROJECT_ROOT/configs/controlled_transition_pad5.yaml}
PAPER_ROOT=${WIND_MAMBA_PAPER_ROOT:-$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1}
FINAL_ROUTE_ROOT=${WIND_MAMBA_FINAL_ROUTE_ROOT:-$PROJECT_ROOT/artifacts/final_route_pad5_v1}
EXPERIMENT_ROOT=${WIND_MAMBA_ROOT:-$PROJECT_ROOT/artifacts/mamba_baselines_pad5_v1}
GPU_CSV=${1:-5,6,7}
SEED_CSV=${WIND_MAMBA_SEEDS:-42,43,44}
WAIT_TIMEOUT_SECONDS=${WIND_WAIT_TIMEOUT_SECONDS:-259200}
WAIT_POLL_SECONDS=${WIND_WAIT_POLL_SECONDS:-30}
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"

if [[ ${#GPUS[@]} -lt 1 || ${#SEEDS[@]} -ne 3 ]]; then
    echo "Provide at least one GPU and exactly three seeds" >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c 'from mamba_ssm import Mamba, Mamba2' 2>/dev/null; then
    echo "Official mamba-ssm with Mamba and Mamba2 is unavailable." >&2
    echo "Run shell/install_mamba_ssm.sh [LOCAL_WHEEL_OR_URL] first." >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WIND_EXPERIMENT_ROOT="$EXPERIMENT_ROOT"
mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/base" "$EXPERIMENT_ROOT/analysis"
JOBS_FILE="$EXPERIMENT_ROOT/logs/mamba_jobs.tsv"
printf 'stage\tindex\tgpu\tpid\tjob\n' > "$JOBS_FILE"

wait_for_file() {
    local target=$1
    local started=$SECONDS
    while [[ ! -f "$target" ]]; do
        if (( SECONDS - started >= WAIT_TIMEOUT_SECONDS )); then
            echo "Timed out waiting for $target" >&2
            exit 3
        fi
        echo "[mamba] waiting elapsed=$((SECONDS - started))s target=$target"
        sleep "$WAIT_POLL_SECONDS"
    done
}

wait_for_file "$PAPER_ROOT/ALL_COMPLETE"
wait_for_file "$FINAL_ROUTE_ROOT/ALL_COMPLETE"
for seed in "${SEEDS[@]}"; do
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_transition_base.py" \
        --source "$PAPER_ROOT/selected/training/seed$seed/config.yaml" \
        --template "$TEMPLATE_CONFIG" \
        --seed "$seed" \
        --auxiliary-weather-loss-weight 0.2 \
        --include-mamba-settings \
        --output "$EXPERIMENT_ROOT/base/seed$seed.yaml"
done

ACTIVE_PIDS=()
stop_active() {
    trap - INT TERM
    for pid in "${ACTIVE_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    exit 130
}
trap stop_active INT TERM

JOBS=()
add_job() { local IFS='|'; JOBS+=("$*"); }
run_stage() {
    local stage=$1
    shift
    local jobs=("$@")
    local total=${#jobs[@]}
    local start=0
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
            WIND_LANE_ID="mamba-$stage-$index" \
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
                echo "[mamba] failed job=${labels[$offset]}" >&2
                failed=1
            fi
        done
        ACTIVE_PIDS=()
        if (( failed != 0 )); then return 1; fi
        start=$end
    done
}

JOBS=()
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/base/seed$seed.yaml"
    add_job forecast mamba "m0_official_mamba/seed$seed" "$base" NONE \
        "seed=$seed" model.transition_type=mamba model.multirate_enabled=false \
        model.mamba_d_state=16 \
        training.deterministic.multirate_target_loss_weight=0.0 \
        training.deterministic.multirate_consistency_loss_weight=0.0
    add_job forecast mamba "m1_official_mamba2/seed$seed" "$base" NONE \
        "seed=$seed" model.transition_type=mamba2 model.multirate_enabled=false \
        model.mamba_d_state=64 \
        training.deterministic.multirate_target_loss_weight=0.0 \
        training.deterministic.multirate_consistency_loss_weight=0.0
done
run_stage training "${JOBS[@]}"

JOBS=()
for id in m0_official_mamba m1_official_mamba2; do
    for seed in "${SEEDS[@]}"; do
        add_job test mamba "$id/seed$seed" NONE NONE 1
    done
done
run_stage test "${JOBS[@]}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/summarize_experiments.py" \
    --root "$EXPERIMENT_ROOT" --output-dir "$EXPERIMENT_ROOT/summary"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_mamba_baselines.py" \
    --root "$EXPERIMENT_ROOT" \
    --final-route-root "$FINAL_ROUTE_ROOT" \
    --paper-root "$PAPER_ROOT" \
    --output-dir "$EXPERIMENT_ROOT/analysis"
touch "$EXPERIMENT_ROOT/ALL_COMPLETE"
echo "[mamba] all complete root=$EXPERIMENT_ROOT"
