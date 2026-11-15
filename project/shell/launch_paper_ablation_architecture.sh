#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then DRY_RUN=1; elif [[ $# -gt 1 ]]; then echo "usage: $0 PHYSICAL_GPU_CSV [--dry-run]" >&2; exit 2; fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
EXPERIMENT_ROOT="${WIND_PAPER_ABLATION_ROOT:-${PROJECT_ROOT}/artifacts/paper_ablation_completion_v1}"
SEED_CSV="${WIND_PAPER_ABLATION_SEEDS:-42,43,44,45,46,47,48,49,50,51}"
GROUP_CSV="${WIND_PAPER_ARCH_GROUPS:-nearest_hour,independent_weather_path,no_wind_geometry,fixed_bilinear,no_aux_weather,no_anchor,frozen_baseline}"
IFS=',' read -r -a GPUS <<<"$GPU_CSV"
IFS=',' read -r -a SEEDS <<<"$SEED_CSV"
IFS=',' read -r -a ABLATION_GROUPS <<<"$GROUP_CSV"

if [[ ${#GPUS[@]} -lt 1 ]]; then echo "At least one physical GPU is required" >&2; exit 2; fi
if [[ ${#SEEDS[@]} -ne 10 ]]; then echo "Formal architecture ablation requires 10 seeds" >&2; exit 2; fi
declare -A GPU_POSITION
for index in "${!GPUS[@]}"; do
    gpu=${GPUS[$index]}
    if [[ -n "${GPU_POSITION[$gpu]+x}" ]]; then echo "Duplicate physical GPU: $gpu" >&2; exit 2; fi
    GPU_POSITION[$gpu]=$((index + 1))
done

overrides_for_group() {
    case "$1" in
        nearest_hour) echo "model.time_query_mode=nearest" ;;
        independent_weather_path) echo "model.forcing_path_mode=independent" ;;
        no_wind_geometry) echo "model.include_wind_geometry=false" ;;
        fixed_bilinear) echo "model.station_query_mode=bilinear model.station_neighborhood_cells=1 model.include_wind_geometry=false" ;;
        no_aux_weather) echo "training.deterministic.auxiliary_weather_loss_weight=0.0" ;;
        no_anchor) echo "model.baseline_anchor_enabled=false training.deterministic.baseline_anchor_loss_weight=0.0" ;;
        frozen_baseline) echo "model.freeze_history_baseline=true" ;;
        *) echo "Unknown architecture group: $1" >&2; return 2 ;;
    esac
}

JOBS=()
for group in "${ABLATION_GROUPS[@]}"; do
    overrides_for_group "$group" >/dev/null
    for seed in "${SEEDS[@]}"; do JOBS+=("${group}|${seed}"); done
done
printf '[paper-ablation] family=architecture\n'
printf '[paper-ablation] project_root=%s\n' "$PROJECT_ROOT"
printf '[paper-ablation] output_root=%s\n' "$EXPERIMENT_ROOT"
printf '[paper-ablation] physical_gpus=%s slots=%s\n' "$GPU_CSV" "${#GPUS[@]}"
printf '[paper-ablation] groups=%s group_count=%s\n' "$GROUP_CSV" "${#ABLATION_GROUPS[@]}"
printf '[paper-ablation] seeds=%s runs=%s\n' "$SEED_CSV" "${#JOBS[@]}"
if (( DRY_RUN )); then
    printf '[paper-ablation] dry-run only; no training was started\n'
    exit 0
fi
mkdir -p "$EXPERIMENT_ROOT/logs"
printf 'index\tphysical_gpu\tpid\tgroup\tseed\n' >"$EXPERIMENT_ROOT/logs/architecture_jobs.tsv"
paper_tqdm_start "${#JOBS[@]}" "Architecture ablations" "run"

ACTIVE=()
AVAILABLE=("${GPUS[@]}")
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
        read -r -a overrides <<<"$(overrides_for_group "$group")"
        printf '[paper-ablation] launch run=%s/%s gpu=%s group=%s seed=%s\n' "$((next + 1))" "${#JOBS[@]}" "$gpu" "$group" "$seed"
        WIND_PAPER_ABLATION_ROOT="$EXPERIMENT_ROOT" \
            WIND_PROGRESS_PREFIX="[gpu=$gpu $group seed=$seed]" \
            WIND_PROGRESS_POSITION="${GPU_POSITION[$gpu]}" \
            WIND_PROGRESS_LEAVE=0 \
            bash "$PROJECT_ROOT/shell/run_paper_ablation_job.sh" \
            "$gpu" "$group" "$seed" NONE "${overrides[@]}" &
        pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="${JOBS[$next]}"
        printf '%s\t%s\t%s\t%s\t%s\n' "$next" "$gpu" "$pid" "$group" "$seed" >>"$EXPERIMENT_ROOT/logs/architecture_jobs.tsv"
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
touch "$EXPERIMENT_ROOT/ARCHITECTURE_COMPLETE"
echo "[paper-ablation] architecture matrix complete: $EXPERIMENT_ROOT"
