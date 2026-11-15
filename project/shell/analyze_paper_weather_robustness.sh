#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"; SPLIT="${WIND_PAPER_EVIDENCE_SPLIT:-test}"; EVIDENCE_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-artifacts/paper_evidence_test_v1}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"; cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/analyze_weather_robustness.py --split "$SPLIT" --root "$EVIDENCE_ROOT/weather_robustness" --original-root "Normal=$EVIDENCE_ROOT/cluster_exports/Normal" --original-root "KD=$EVIDENCE_ROOT/cluster_exports/KD" --output-dir "$EVIDENCE_ROOT/weather_robustness_analysis" "$@"
