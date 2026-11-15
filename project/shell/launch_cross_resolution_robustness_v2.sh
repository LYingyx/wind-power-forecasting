#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
SELECTION="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/temporal_selection.json"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/resolve_cross_resolution_selection.py" \
    --project-root "$PROJECT_ROOT" --output "$SELECTION" >/dev/null
"$PYTHON_BIN" - "$SELECTION" <<'PY'
import json, pathlib, sys
selection = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not selection.get("temporal_structure_frozen", False):
    raise SystemExit("Robustness gate is closed: temporal mechanism selection is incomplete")
if selection.get("selected_variant") not in {"T2_bracket_nonlinear", "t2_bracket_nonlinear", "t2_no_diff"}:
    raise SystemExit("Robustness T2 claim is not run because Stage A did not select T2")
PY
IFS=',' read -r -a GPUS <<<"$GPU_CSV"
GROUPS=(T0_raw_linear T1_latent_linear T2_final T4_native_query)
SEEDS=(42 43 44 45 46 47 48 49 50 51)
JOBS=(); for group in "${GROUPS[@]}"; do for seed in "${SEEDS[@]}"; do JOBS+=("$group|$seed"); done; done
declare -A SEEN POSITION
for index in "${!GPUS[@]}"; do gpu=${GPUS[$index]}; [[ -z "${SEEN[$gpu]+x}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }; SEEN[$gpu]=1; POSITION[$gpu]=$((index + 1)); done
ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/robustness"
mkdir -p "$ROOT/logs"
printf '[robustness] physical_gpus=%s runs=%s\n' "$GPU_CSV" "${#JOBS[@]}"
paper_tqdm_start "${#JOBS[@]}" "Cross-resolution robustness" "eval"
ACTIVE=(); AVAILABLE=("${GPUS[@]}"); next=0; declare -A PID_GPU PID_JOB
abort_launcher() { for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit 130; }
trap abort_launcher INT TERM
while (( next < ${#JOBS[@]} || ${#ACTIVE[@]} > 0 )); do
    while (( next < ${#JOBS[@]} && ${#AVAILABLE[@]} > 0 )); do
        slot=$((${#AVAILABLE[@]} - 1)); gpu=${AVAILABLE[$slot]}; unset 'AVAILABLE[$slot]'
        IFS='|' read -r group seed <<<"${JOBS[$next]}"
        WIND_PROGRESS_PREFIX="[gpu=$gpu $group seed=$seed]" \
            WIND_PROGRESS_POSITION="${POSITION[$gpu]}" WIND_PROGRESS_LEAVE=0 \
            bash "$PROJECT_ROOT/shell/run_cross_resolution_robustness_job.sh" "$gpu" "$group" "$seed" &
        pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]="${JOBS[$next]}"; next=$((next + 1))
    done
    status=0; wait -n -p finished "${ACTIVE[@]}" || status=$?
    AVAILABLE+=("${PID_GPU[$finished]}"); remaining=(); for pid in "${ACTIVE[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done; ACTIVE=("${remaining[@]}")
    if (( status != 0 )); then echo "Failed: ${PID_JOB[$finished]}" >&2; abort_launcher; fi
    paper_tqdm_update "${PID_JOB[$finished]} gpu=${PID_GPU[$finished]}"
done
paper_tqdm_finish
touch "$ROOT/EVALUATION_COMPLETE"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_cross_resolution_robustness.py" --project-root "$PROJECT_ROOT"
