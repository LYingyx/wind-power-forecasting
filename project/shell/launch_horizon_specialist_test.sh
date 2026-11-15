#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
EXPERIMENT_ROOT="${WIND_HS_ROOT:-${PROJECT_ROOT}/artifacts/horizon_specialist_v1}"
SCREEN_DECISION="$EXPERIMENT_ROOT/analysis_screening/decision.json"
[[ -f "$EXPERIMENT_ROOT/analysis_val/analysis.json" ]] || {
    echo "Formal validation analysis must be frozen before test" >&2; exit 3;
}
[[ -f "$SCREEN_DECISION" ]] || { echo "Missing screening decision" >&2; exit 3; }
LIGHT=$(jq -r '.selected_lightweight // empty' "$SCREEN_DECISION")
HIGH=$(jq -r '.selected_high_cost // empty' "$SCREEN_DECISION")
EXP_GROUPS=()
[[ -z "$LIGHT" ]] || EXP_GROUPS+=("$LIGHT")
[[ -z "$HIGH" ]] || EXP_GROUPS+=(h0_dual_full_taper "$HIGH")
[[ ! -d "$EXPERIMENT_ROOT/groups/kd_${LIGHT}" ]] || EXP_GROUPS+=("kd_${LIGHT}")
[[ ${#EXP_GROUPS[@]} -gt 0 ]] || { echo "No frozen candidate groups" >&2; exit 3; }
IFS=',' read -r -a GPUS <<<"$GPU_CSV"
SEEDS=(42 43 44 45 46 47 48 49 50 51)
declare -A POSITION SEEN
for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}; [[ -z "${SEEN[$gpu]+x}" ]] || { echo "Duplicate GPU $gpu" >&2; exit 2; }
    SEEN[$gpu]=1 POSITION[$gpu]=$((index + 1))
done
JOBS=(); for group in "${EXP_GROUPS[@]}"; do for seed in "${SEEDS[@]}"; do JOBS+=("$group|$seed"); done; done
mkdir -p "$EXPERIMENT_ROOT/logs"
paper_tqdm_start "${#JOBS[@]}" "Frozen 2024 test evaluation" "run"
ACTIVE=() AVAILABLE=("${GPUS[@]}"); next=0
declare -A PID_GPU PID_JOB
trap 'for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit 130' INT TERM
while (( next < ${#JOBS[@]} || ${#ACTIVE[@]} > 0 )); do
    while (( next < ${#JOBS[@]} && ${#AVAILABLE[@]} > 0 )); do
        slot=$((${#AVAILABLE[@]} - 1)); gpu=${AVAILABLE[$slot]}; unset 'AVAILABLE[$slot]'
        IFS='|' read -r group seed <<<"${JOBS[$next]}"
        WIND_HS_ROOT="$EXPERIMENT_ROOT" WIND_PROGRESS_POSITION="${POSITION[$gpu]}" \
            bash "$PROJECT_ROOT/shell/run_horizon_specialist_test_job.sh" "$gpu" "$group" "$seed" &
        pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="${JOBS[$next]}"; next=$((next + 1))
    done
    status=0; wait -n -p finished "${ACTIVE[@]}" || status=$?
    AVAILABLE+=("${PID_GPU[$finished]}")
    remaining=(); for pid in "${ACTIVE[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done; ACTIVE=("${remaining[@]}")
    if (( status != 0 )); then
        for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
        paper_tqdm_abort; wait 2>/dev/null || true; exit "$status"
    fi
    paper_tqdm_update "${PID_JOB[$finished]} gpu=${PID_GPU[$finished]}"
done
paper_tqdm_finish
touch "$EXPERIMENT_ROOT/TEST_COMPLETE"
echo "[horizon-specialist] frozen test evaluation complete"
