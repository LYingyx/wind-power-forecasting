#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
EXPERIMENT_ROOT="${WIND_CR_ROOT:-${PROJECT_ROOT}/artifacts/cross_resolution_temporal_v1}"
SEED_CSV="${WIND_CR_SEEDS:-42,46,50}"
GROUP_CSV="${WIND_CR_GROUPS:-t0_raw_linear,t1_latent_linear,t2_bracket_nonlinear,t3_nearest}"
IFS=',' read -r -a GPUS <<<"$GPU_CSV"
IFS=',' read -r -a SEEDS <<<"$SEED_CSV"
IFS=',' read -r -a EXP_GROUPS <<<"$GROUP_CSV"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "At least one physical GPU is required" >&2; exit 2; }
declare -A SEEN POSITION
for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}
    [[ -z "${SEEN[$gpu]+x}" ]] || { echo "Duplicate physical GPU: $gpu" >&2; exit 2; }
    SEEN[$gpu]=1; POSITION[$gpu]=$((index + 1))
done
JOBS=(); for group in "${EXP_GROUPS[@]}"; do for seed in "${SEEDS[@]}"; do JOBS+=("$group|$seed"); done; done
mkdir -p "$EXPERIMENT_ROOT/logs"
printf 'index\tphysical_gpu\tpid\tgroup\tseed\n' >"$EXPERIMENT_ROOT/logs/jobs.tsv"
printf '[cross-resolution] physical_gpus=%s groups=%s seeds=%s runs=%s\n' \
    "$GPU_CSV" "$GROUP_CSV" "$SEED_CSV" "${#JOBS[@]}"
paper_tqdm_start "${#JOBS[@]}" "Cross-resolution T0-T3" "run"
ACTIVE=(); AVAILABLE=("${GPUS[@]}"); next=0
declare -A PID_GPU PID_JOB
abort_launcher() {
    for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    paper_tqdm_abort; wait 2>/dev/null || true; exit 130
}
trap abort_launcher INT TERM
while (( next < ${#JOBS[@]} || ${#ACTIVE[@]} > 0 )); do
    while (( next < ${#JOBS[@]} && ${#AVAILABLE[@]} > 0 )); do
        slot=$((${#AVAILABLE[@]} - 1)); gpu=${AVAILABLE[$slot]}; unset 'AVAILABLE[$slot]'
        IFS='|' read -r group seed <<<"${JOBS[$next]}"
        WIND_CR_ROOT="$EXPERIMENT_ROOT" \
            WIND_PROGRESS_PREFIX="[gpu=$gpu $group seed=$seed]" \
            WIND_PROGRESS_POSITION="${POSITION[$gpu]}" WIND_PROGRESS_LEAVE=0 \
            bash "$PROJECT_ROOT/shell/run_cross_resolution_temporal_job.sh" \
            "$gpu" "$group" "$seed" &
        pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="${JOBS[$next]}"
        printf '%s\t%s\t%s\t%s\t%s\n' "$next" "$gpu" "$pid" "$group" "$seed" \
            >>"$EXPERIMENT_ROOT/logs/jobs.tsv"
        next=$((next + 1))
    done
    status=0; wait -n -p finished "${ACTIVE[@]}" || status=$?
    AVAILABLE+=("${PID_GPU[$finished]}")
    remaining=(); for pid in "${ACTIVE[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done
    ACTIVE=("${remaining[@]}")
    if (( status != 0 )); then
        echo "Failed: ${PID_JOB[$finished]}" >&2
        for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
        paper_tqdm_abort; wait 2>/dev/null || true; exit "$status"
    fi
    paper_tqdm_update "${PID_JOB[$finished]} gpu=${PID_GPU[$finished]}"
done
paper_tqdm_finish
touch "$EXPERIMENT_ROOT/TRAINING_COMPLETE"
echo "[cross-resolution] T0-T3 training complete"
