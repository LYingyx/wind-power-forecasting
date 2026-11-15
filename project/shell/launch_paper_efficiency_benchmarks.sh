#!/usr/bin/env bash
set -euo pipefail
GPU_CSV="${1:-0}"; PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"; source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"; OUTPUT_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-${PROJECT_ROOT}/artifacts/paper_evidence_test_v1}"; SEED="${WIND_BENCHMARK_SEED:-42}"
IFS=',' read -r -a GPUS <<<"$GPU_CSV"; SPECS=(
 "Direct|${PROJECT_ROOT}/artifacts/incremental_mechanism_v2/groups/direct_d64"
 "Normal|${PROJECT_ROOT}/artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h"
 "KD|${PROJECT_ROOT}/artifacts/ensemble_stabilization_v1/groups/kd_050"
)
if [[ ${#GPUS[@]} -lt 1 ]]; then echo "At least one physical GPU is required" >&2; exit 2; fi
declare -A GPU_POSITION; for index in "${!GPUS[@]}"; do gpu=${GPUS[$index]}; if [[ -n "${GPU_POSITION[$gpu]+x}" ]]; then echo "Duplicate physical GPU: $gpu" >&2; exit 2; fi; GPU_POSITION[$gpu]=$((index + 1)); done
mkdir -p "$OUTPUT_ROOT/logs"; printf 'index\tphysical_gpu\tpid\tlabel\tseed\n' >"$OUTPUT_ROOT/logs/efficiency_jobs.tsv"
paper_tqdm_start "${#SPECS[@]}" "Efficiency benchmark" "model"
ACTIVE=(); AVAILABLE=("${GPUS[@]}"); next=0; declare -A PID_GPU PID_JOB
abort_launcher() { for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit 130; }; trap abort_launcher INT TERM
while (( next < ${#SPECS[@]} || ${#ACTIVE[@]} > 0 )); do
 while (( next < ${#SPECS[@]} && ${#AVAILABLE[@]} > 0 )); do
  slot=$((${#AVAILABLE[@]} - 1)); gpu=${AVAILABLE[$slot]}; unset 'AVAILABLE[$slot]'; IFS='|' read -r label root <<<"${SPECS[$next]}"
  WIND_PAPER_EVIDENCE_ROOT="$OUTPUT_ROOT" WIND_PROGRESS_PREFIX="[gpu=$gpu $label]" WIND_PROGRESS_POSITION="${GPU_POSITION[$gpu]}" WIND_PROGRESS_LEAVE=0 bash "$PROJECT_ROOT/shell/run_paper_efficiency_job.sh" "$gpu" "$label" "$root" "$SEED" &
  pid=$!; ACTIVE+=("$pid"); PID_GPU[$pid]=$gpu; PID_JOB[$pid]=$label; printf '%s\t%s\t%s\t%s\t%s\n' "$next" "$gpu" "$pid" "$label" "$SEED" >>"$OUTPUT_ROOT/logs/efficiency_jobs.tsv"; next=$((next + 1))
 done
 status=0; wait -n -p finished "${ACTIVE[@]}" || status=$?; AVAILABLE+=("${PID_GPU[$finished]}"); remaining=(); for pid in "${ACTIVE[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done; ACTIVE=("${remaining[@]}"); if (( status != 0 )); then echo "Failed: ${PID_JOB[$finished]}" >&2; for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit "$status"; fi; paper_tqdm_update "${PID_JOB[$finished]} gpu=${PID_GPU[$finished]}"
done
paper_tqdm_finish
"${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}" - "$OUTPUT_ROOT/efficiency" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[json.loads(p.read_text()) for p in sorted(root.glob('*_seed*.json'))]
if rows:
 fields=list(dict.fromkeys(k for r in rows for k in r));
 with (root/'benchmarks.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 by={r['label']:r for r in rows}
 if 'Normal' in by and 'KD' in by:
  (root/'deployment_comparison.json').write_text(json.dumps({'kd_single_ms_per_origin':by['KD']['milliseconds_per_origin'],'estimated_normal_10member_ms_per_origin':10*by['Normal']['milliseconds_per_origin'],'estimated_speedup_vs_10member':10*by['Normal']['milliseconds_per_origin']/by['KD']['milliseconds_per_origin'],'note':'10-member latency is a sequential estimate; report as estimate unless measured under the deployment serving topology.'},indent=2))
PY
echo "[efficiency] complete: $OUTPUT_ROOT/efficiency"
