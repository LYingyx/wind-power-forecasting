#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"; SPLIT="${WIND_PAPER_EVIDENCE_SPLIT:-test}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"; cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/generate_station_evidence.py --split "$SPLIT" --exports-root "${WIND_PAPER_EVIDENCE_ROOT:-artifacts/paper_evidence_test_v1}/station_exports" --output-dir "${WIND_PAPER_EVIDENCE_ROOT:-artifacts/paper_evidence_test_v1}/station" "$@"
