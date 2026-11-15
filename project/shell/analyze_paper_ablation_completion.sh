#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/analyze_paper_ablation_completion.py \
    --root "${WIND_PAPER_ABLATION_ROOT:-artifacts/paper_ablation_completion_v1}" \
    "$@"
