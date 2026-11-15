#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then echo "usage: $0 normal|kd [GPU_CSV]" >&2; exit 2; fi
GROUP=$1 GPU_CSV=${2:-0}
[[ "$GROUP" == normal || "$GROUP" == kd ]] || { echo "group must be normal or kd" >&2; exit 2; }
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
[[ -f "$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/spatial/SPATIAL_STRUCTURE_FROZEN" ]] || {
    echo "Final training gate is closed: spatial structure is not frozen" >&2; exit 3;
}
IFS=',' read -r -a GPUS <<<"$GPU_CSV"
SEEDS=(42 43 44 45 46 47 48 49 50 51)
declare -A SEEN POSITION
for index in "${!GPUS[@]}"; do gpu=${GPUS[$index]}; [[ -z "${SEEN[$gpu]+x}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }; SEEN[$gpu]=1; POSITION[$gpu]=$((index+1)); done
ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/final"
mkdir -p "$ROOT/logs"
if [[ "$GROUP" == normal ]]; then
    export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/reuse_cross_resolution_final_normal.py" \
        --project-root "$PROJECT_ROOT"
fi
printf '[cross-resolution-final] group=%s physical_gpus=%s runs=10\n' "$GROUP" "$GPU_CSV"
paper_tqdm_start 10 "Final $GROUP checkpoints" "run"
ACTIVE=(); AVAILABLE=("${GPUS[@]}"); next=0; declare -A PID_GPU PID_JOB
abort_launcher(){ for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit 130; }
trap abort_launcher INT TERM
while (( next < 10 || ${#ACTIVE[@]} > 0 )); do
    while (( next < 10 && ${#AVAILABLE[@]} > 0 )); do
        slot=$((${#AVAILABLE[@]}-1)); gpu=${AVAILABLE[$slot]}; unset 'AVAILABLE[$slot]'; seed=${SEEDS[$next]}
        WIND_PROGRESS_PREFIX="[gpu=$gpu final-$GROUP seed=$seed]" WIND_PROGRESS_POSITION="${POSITION[$gpu]}" WIND_PROGRESS_LEAVE=0 \
            bash "$PROJECT_ROOT/shell/run_cross_resolution_final_job.sh" "$gpu" "$GROUP" "$seed" &
        pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="$GROUP|$seed"; next=$((next+1))
    done
    status=0; wait -n -p finished "${ACTIVE[@]}" || status=$?
    AVAILABLE+=("${PID_GPU[$finished]}"); remaining=(); for pid in "${ACTIVE[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done; ACTIVE=("${remaining[@]}")
    if (( status != 0 )); then echo "Failed: ${PID_JOB[$finished]}" >&2; abort_launcher; fi
    paper_tqdm_update "${PID_JOB[$finished]} gpu=${PID_GPU[$finished]}"
done
paper_tqdm_finish
touch "$ROOT/${GROUP^^}_COMPLETE"
