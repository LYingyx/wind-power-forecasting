#!/usr/bin/env bash
set -euo pipefail

GPU_CSV=${1:-0}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
DECISION="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/final/analysis_val/decision.json"
[[ -f "$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/final/analysis_val/VALIDATION_SELECTION_FROZEN" ]] || { echo "2024 gate is closed" >&2; exit 3; }
IFS=',' read -r -a GPUS <<<"$GPU_CSV"; declare -A SEEN POSITION
for index in "${!GPUS[@]}"; do gpu=${GPUS[$index]}; [[ -z "${SEEN[$gpu]+x}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }; SEEN[$gpu]=1; POSITION[$gpu]=$((index+1)); done
JOBS=(); for group in normal kd; do for seed in {42..51}; do JOBS+=("$group|$seed"); done; done
paper_tqdm_start "${#JOBS[@]}" "Frozen 2024 evaluation" "checkpoint"
ACTIVE=(); AVAILABLE=("${GPUS[@]}"); next=0; declare -A PID_GPU PID_JOB
abort_launcher(){ for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit 130; }; trap abort_launcher INT TERM
while (( next < ${#JOBS[@]} || ${#ACTIVE[@]} > 0 )); do
    while (( next < ${#JOBS[@]} && ${#AVAILABLE[@]} > 0 )); do
        slot=$((${#AVAILABLE[@]}-1)); gpu=${AVAILABLE[$slot]}; unset 'AVAILABLE[$slot]'; IFS='|' read -r group seed <<<"${JOBS[$next]}"
        WIND_PROGRESS_PREFIX="[gpu=$gpu test-$group seed=$seed]" WIND_PROGRESS_POSITION="${POSITION[$gpu]}" WIND_PROGRESS_LEAVE=0 \
            bash "$PROJECT_ROOT/shell/run_cross_resolution_final_test_job.sh" "$gpu" "$group" "$seed" &
        pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="${JOBS[$next]}"; next=$((next+1))
    done
    status=0; wait -n -p finished "${ACTIVE[@]}" || status=$?; AVAILABLE+=("${PID_GPU[$finished]}")
    remaining=(); for pid in "${ACTIVE[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done; ACTIVE=("${remaining[@]}")
    if (( status != 0 )); then echo "Failed: ${PID_JOB[$finished]}" >&2; abort_launcher; fi
    paper_tqdm_update "${PID_JOB[$finished]} gpu=${PID_GPU[$finished]}"
done
paper_tqdm_finish
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_cross_resolution_final_test.py" --project-root "$PROJECT_ROOT"
