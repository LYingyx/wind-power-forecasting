#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"; SPLIT="${WIND_PAPER_EVIDENCE_SPLIT:-test}"; EVIDENCE_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-artifacts/paper_evidence_test_v1}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"; cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/generate_forecast_evidence.py --split "$SPLIT" --group "Direct=$EVIDENCE_ROOT/cluster_exports/Direct" --group "Normal=$EVIDENCE_ROOT/cluster_exports/Normal" --group "KD=$EVIDENCE_ROOT/cluster_exports/KD" --output-dir "$EVIDENCE_ROOT/cluster" "$@"
