#!/usr/bin/env bash
set -euo pipefail
GPU_CSV="${1:-0}"; SPLIT="${WIND_PAPER_EVIDENCE_SPLIT:-test}"; PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"; source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
ROOT="${WIND_PAPER_ABLATION_ROOT:-${PROJECT_ROOT}/artifacts/paper_ablation_completion_v1}"; SEED_CSV="${WIND_PAPER_ABLATION_SEEDS:-42,43,44,45,46,47,48,49,50,51}"; GROUP_CSV="${WIND_PAPER_TEST_GROUPS:-nearest_hour,independent_weather_path,no_wind_geometry,fixed_bilinear,no_aux_weather,no_anchor,frozen_baseline,kd_station_only,kd_cluster_only,kd_no_aux_weather,kd_no_anchor}"
IFS=',' read -r -a GPUS <<<"$GPU_CSV"; IFS=',' read -r -a SEEDS <<<"$SEED_CSV"; IFS=',' read -r -a TEST_GROUPS <<<"$GROUP_CSV"
if [[ ${#SEEDS[@]} -ne 10 ]]; then echo "Formal test evaluation requires 10 seeds" >&2; exit 2; fi
declare -A GPU_POSITION; for index in "${!GPUS[@]}"; do gpu=${GPUS[$index]}; if [[ -n "${GPU_POSITION[$gpu]+x}" ]]; then echo "Duplicate physical GPU: $gpu" >&2; exit 2; fi; GPU_POSITION[$gpu]=$((index + 1)); done
JOBS=(); missing=(); for group in "${TEST_GROUPS[@]}"; do for seed in "${SEEDS[@]}"; do JOBS+=("$group|$seed"); [[ -f "$ROOT/groups/$group/seed$seed/COMPLETE" ]] || missing+=("$group/seed$seed"); done; done
if (( ${#missing[@]} > 0 )); then printf 'Training incomplete; run architecture/KD launcher first:\n' >&2; printf '  %s\n' "${missing[@]}" >&2; exit 3; fi
mkdir -p "$ROOT/logs"; printf 'index\tphysical_gpu\tpid\tgroup\tseed\tsplit\n' >"$ROOT/logs/ablation_${SPLIT}_evaluation_jobs.tsv"; printf '[ablation-%s] gpus=%s runs=%s\n' "$SPLIT" "$GPU_CSV" "${#JOBS[@]}"; paper_tqdm_start "${#JOBS[@]}" "Ablation ${SPLIT} evaluation" eval
ACTIVE=(); AVAILABLE=("${GPUS[@]}"); next=0; declare -A PID_GPU PID_JOB
abort_launcher() { for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit 130; }; trap abort_launcher INT TERM
while (( next < ${#JOBS[@]} || ${#ACTIVE[@]} > 0 )); do
 while (( next < ${#JOBS[@]} && ${#AVAILABLE[@]} > 0 )); do slot=$((${#AVAILABLE[@]} - 1)); gpu=${AVAILABLE[$slot]}; unset 'AVAILABLE[$slot]'; IFS='|' read -r group seed <<<"${JOBS[$next]}"; WIND_PAPER_ABLATION_ROOT="$ROOT" WIND_PROGRESS_PREFIX="[gpu=$gpu $group seed=$seed $SPLIT]" WIND_PROGRESS_POSITION="${GPU_POSITION[$gpu]}" WIND_PROGRESS_LEAVE=0 bash "$PROJECT_ROOT/shell/run_paper_ablation_test_job.sh" "$gpu" "$group" "$seed" "$SPLIT" & pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="${JOBS[$next]}"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$next" "$gpu" "$pid" "$group" "$seed" "$SPLIT" >>"$ROOT/logs/ablation_${SPLIT}_evaluation_jobs.tsv"; next=$((next + 1)); done
 status=0; wait -n -p finished "${ACTIVE[@]}" || status=$?; AVAILABLE+=("${PID_GPU[$finished]}"); remaining=(); for pid in "${ACTIVE[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done; ACTIVE=("${remaining[@]}"); if (( status != 0 )); then echo "Failed: ${PID_JOB[$finished]}" >&2; for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit "$status"; fi; paper_tqdm_update "${PID_JOB[$finished]} gpu=${PID_GPU[$finished]}"
done
paper_tqdm_finish; touch "$ROOT/ABLATION_${SPLIT^^}_EVALUATION_COMPLETE"; echo "[ablation-$SPLIT] matrix complete"
