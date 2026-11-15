#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ $# -gt 1 ]]; then
    echo "usage: $0 PHYSICAL_GPU_CSV [--dry-run]" >&2
    exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
EXPERIMENT_ROOT="${WIND_HS_ROOT:-${PROJECT_ROOT}/artifacts/horizon_specialist_v1}"
SEED_CSV="${WIND_HS_SCREEN_SEEDS:-42,46,50}"
GROUP_CSV="${WIND_HS_SCREEN_GROUPS:-l1_anchor_taper,l2_fixed_view_scalar,l3_fixed_view_taper}"
REQUIRED_SEED_COUNT="${WIND_HS_REQUIRED_SEED_COUNT:-3}"
STAGE_LABEL="${WIND_HS_STAGE_LABEL:-screening}"
APPEND_HIGH_COST="${WIND_HS_APPEND_HIGH_COST:-1}"
JOB_TEACHER_ARCHIVE="${WIND_HS_JOB_TEACHER_ARCHIVE:-}"
DECISION="$EXPERIMENT_ROOT/stage0/decision.json"
if [[ "$APPEND_HIGH_COST" == "1" && -f "$DECISION" ]] && [[ "$(jq -r '.go_high_cost_specialists' "$DECISION")" == "true" ]]; then
    GROUP_CSV="${GROUP_CSV},h0_dual_full_taper,h1_dual_mixed_taper"
fi
IFS=',' read -r -a GPUS <<<"$GPU_CSV"
IFS=',' read -r -a SEEDS <<<"$SEED_CSV"
IFS=',' read -r -a EXP_GROUPS <<<"$GROUP_CSV"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "At least one physical GPU is required" >&2; exit 2; }
[[ ${#SEEDS[@]} -eq "$REQUIRED_SEED_COUNT" ]] || { echo "$STAGE_LABEL requires exactly $REQUIRED_SEED_COUNT seeds" >&2; exit 2; }
declare -A SEEN POSITION
for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}
    [[ -z "${SEEN[$gpu]+x}" ]] || { echo "Duplicate physical GPU: $gpu" >&2; exit 2; }
    SEEN[$gpu]=1 POSITION[$gpu]=$((index + 1))
done
JOBS=()
for group in "${EXP_GROUPS[@]}"; do
    for seed in "${SEEDS[@]}"; do JOBS+=("${group}|${seed}"); done
done
printf '[horizon-specialist] stage=%s root=%s\n' "$STAGE_LABEL" "$EXPERIMENT_ROOT"
printf '[horizon-specialist] physical_gpus=%s groups=%s seeds=%s runs=%s\n' \
    "$GPU_CSV" "$GROUP_CSV" "$SEED_CSV" "${#JOBS[@]}"
if (( DRY_RUN )); then exit 0; fi
mkdir -p "$EXPERIMENT_ROOT/logs"
printf 'index\tphysical_gpu\tpid\tgroup\tseed\n' >"$EXPERIMENT_ROOT/logs/${STAGE_LABEL}_jobs.tsv"
paper_tqdm_start "${#JOBS[@]}" "Horizon specialist ${STAGE_LABEL}" "run"
ACTIVE=() AVAILABLE=("${GPUS[@]}")
declare -A PID_GPU PID_JOB
next=0
abort_launcher() {
    for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    paper_tqdm_abort
    wait 2>/dev/null || true
    exit 130
}
trap abort_launcher INT TERM
while (( next < ${#JOBS[@]} || ${#ACTIVE[@]} > 0 )); do
    while (( next < ${#JOBS[@]} && ${#AVAILABLE[@]} > 0 )); do
        slot=$((${#AVAILABLE[@]} - 1)); gpu=${AVAILABLE[$slot]}; unset 'AVAILABLE[$slot]'
        IFS='|' read -r group seed <<<"${JOBS[$next]}"
        printf '[horizon-specialist] launch %s/%s gpu=%s group=%s seed=%s\n' \
            "$((next + 1))" "${#JOBS[@]}" "$gpu" "$group" "$seed"
        JOB_ARGS=("$gpu" "$group" "$seed")
        [[ -z "$JOB_TEACHER_ARCHIVE" ]] || JOB_ARGS+=("$JOB_TEACHER_ARCHIVE")
        WIND_HS_ROOT="$EXPERIMENT_ROOT" \
            WIND_PROGRESS_PREFIX="[gpu=$gpu $group seed=$seed]" \
            WIND_PROGRESS_POSITION="${POSITION[$gpu]}" WIND_PROGRESS_LEAVE=0 \
            bash "$PROJECT_ROOT/shell/run_horizon_specialist_job.sh" \
            "${JOB_ARGS[@]}" &
        pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="${JOBS[$next]}"
        printf '%s\t%s\t%s\t%s\t%s\n' "$next" "$gpu" "$pid" "$group" "$seed" \
            >>"$EXPERIMENT_ROOT/logs/${STAGE_LABEL}_jobs.tsv"
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
MARKER=$(printf '%s_COMPLETE' "$STAGE_LABEL" | tr '[:lower:]-' '[:upper:]_')
touch "$EXPERIMENT_ROOT/$MARKER"
echo "[horizon-specialist] $STAGE_LABEL complete"
