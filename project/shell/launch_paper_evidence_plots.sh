#!/usr/bin/env bash
set -euo pipefail

MODE_CSV="${1:-cluster}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
source "$PROJECT_ROOT/shell/tqdm_parallel_lib.sh"
EVIDENCE_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-$PROJECT_ROOT/artifacts/paper_evidence_test_v1}"
mkdir -p "$EVIDENCE_ROOT"
IFS=',' read -r -a MODES <<<"$MODE_CSV"
PIDS=(); LABELS=()
paper_tqdm_start "${#MODES[@]}" "Evidence plotting" "set"
abort_launcher() { for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done; paper_tqdm_abort; wait 2>/dev/null || true; exit 130; }
trap abort_launcher INT TERM
for index in "${!MODES[@]}"; do
    mode=${MODES[$index]}
    case "$mode" in
        cluster) command=(bash "$PROJECT_ROOT/shell/generate_paper_cluster_evidence.sh") ;;
        station) command=(bash "$PROJECT_ROOT/shell/generate_paper_station_evidence.sh") ;;
        weather) command=(bash "$PROJECT_ROOT/shell/analyze_paper_weather_robustness.sh") ;;
        *) echo "Unknown plot mode: $mode (use cluster,station,weather)" >&2; exit 2 ;;
    esac
    WIND_PROGRESS_PREFIX="[$mode]" WIND_PROGRESS_POSITION=$((index + 1)) WIND_PROGRESS_LEAVE=0 "${command[@]}" >"$EVIDENCE_ROOT/${mode}_plot.log" 2> >(tee -a "$EVIDENCE_ROOT/${mode}_plot.log" >&2) &
    PIDS+=("$!"); LABELS+=("$mode")
done
failed=0
for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then echo "Plot generation failed: ${LABELS[$index]}" >&2; failed=1; else paper_tqdm_update "${LABELS[$index]}"; fi
done
paper_tqdm_finish
exit "$failed"
