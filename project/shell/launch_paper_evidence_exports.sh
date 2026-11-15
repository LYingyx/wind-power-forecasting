#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
OUTPUT_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-${PROJECT_ROOT}/artifacts/paper_evidence_test_v1}"
SPLIT="${WIND_PAPER_EVIDENCE_SPLIT:-test}"
SEED_CSV="${WIND_PAPER_EVIDENCE_EXPORT_SEEDS:-42}"
IFS=',' read -r -a GPUS <<<"$GPU_CSV"
IFS=',' read -r -a SEEDS <<<"$SEED_CSV"
if [[ ${#GPUS[@]} -lt 1 ]]; then echo "At least one physical GPU is required" >&2; exit 2; fi
declare -A GPU_POSITION
for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}
    if [[ -n "${GPU_POSITION[$gpu]+x}" ]]; then echo "Duplicate physical GPU: $gpu" >&2; exit 2; fi
    GPU_POSITION[$gpu]=$((index + 1))
done

SPECS=(
    "Direct|${PROJECT_ROOT}/artifacts/incremental_mechanism_v2/groups/direct_d64"
    "Normal|${PROJECT_ROOT}/artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h"
    "KD|${PROJECT_ROOT}/artifacts/ensemble_stabilization_v1/groups/kd_050"
)
JOBS=()
for spec in "${SPECS[@]}"; do
    IFS='|' read -r label root <<<"$spec"
    for seed in "${SEEDS[@]}"; do JOBS+=("$label|$root|$seed"); done
done
mkdir -p "$OUTPUT_ROOT/logs"
printf 'index\tphysical_gpu\tpid\tlabel\tseed\tsplit\n' >"$OUTPUT_ROOT/logs/station_${SPLIT}_export_jobs.tsv"
paper_tqdm_start "${#JOBS[@]}" "Station exports" "run"

ACTIVE=(); AVAILABLE=("${GPUS[@]}"); next=0
declare -A PID_GPU PID_JOB
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
        IFS='|' read -r label root seed <<<"${JOBS[$next]}"
        WIND_PAPER_EVIDENCE_ROOT="$OUTPUT_ROOT" \
            WIND_PROGRESS_PREFIX="[gpu=$gpu $label seed=$seed]" \
            WIND_PROGRESS_POSITION="${GPU_POSITION[$gpu]}" \
            WIND_PROGRESS_LEAVE=0 \
            bash "$PROJECT_ROOT/shell/run_paper_evidence_export_job.sh" "$gpu" "$label" "$root" "$seed" "$SPLIT" &
        pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="${JOBS[$next]}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$next" "$gpu" "$pid" "$label" "$seed" "$SPLIT" >>"$OUTPUT_ROOT/logs/station_${SPLIT}_export_jobs.tsv"
        next=$((next + 1))
    done
    status=0; wait -n -p finished "${ACTIVE[@]}" || status=$?
    AVAILABLE+=("${PID_GPU[$finished]}")
    remaining=(); for pid in "${ACTIVE[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done; ACTIVE=("${remaining[@]}")
    if (( status != 0 )); then
        echo "Failed: ${PID_JOB[$finished]}" >&2
        for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
        paper_tqdm_abort
        wait 2>/dev/null || true
        exit "$status"
    fi
    paper_tqdm_update "${PID_JOB[$finished]} gpu=${PID_GPU[$finished]}"
done
paper_tqdm_finish
touch "$OUTPUT_ROOT/STATION_${SPLIT^^}_EXPORTS_COMPLETE"
echo "[evidence-export] all complete: $OUTPUT_ROOT"
