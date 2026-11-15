#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then echo "usage: $0 PHYSICAL_GPU LABEL GROUP_ROOT SEED" >&2; exit 2; fi
GPU=$1; LABEL=$2; GROUP_ROOT=$3; SEED=$4
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
OUTPUT_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-${PROJECT_ROOT}/artifacts/paper_evidence_test_v1}"; OUTPUT="$OUTPUT_ROOT/efficiency/${LABEL}_seed${SEED}.json"; SPLIT="${WIND_PAPER_EVIDENCE_SPLIT:-test}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"; export CUDA_VISIBLE_DEVICES="$GPU"; mkdir -p "$(dirname "$OUTPUT")"
LOG="${OUTPUT%.json}.log"
if [[ "${WIND_TQDM_VISIBLE:-1}" == "1" ]]; then
 "$PYTHON_BIN" "$PROJECT_ROOT/scripts/benchmark_inference.py" --run-dir "$GROUP_ROOT/seed$SEED" --label "$LABEL" --output "$OUTPUT" --split "$SPLIT" --device cuda --warmup "${WIND_BENCHMARK_WARMUP:-20}" --iterations "${WIND_BENCHMARK_ITERATIONS:-100}" >"$LOG" 2> >(tee -a "$LOG" >&2)
else
 "$PYTHON_BIN" "$PROJECT_ROOT/scripts/benchmark_inference.py" --run-dir "$GROUP_ROOT/seed$SEED" --label "$LABEL" --output "$OUTPUT" --split "$SPLIT" --device cuda --warmup "${WIND_BENCHMARK_WARMUP:-20}" --iterations "${WIND_BENCHMARK_ITERATIONS:-100}" >"$LOG" 2>&1
fi
