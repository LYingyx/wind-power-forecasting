#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DECISION="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/formal/analysis_val/decision.json"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" - "$DECISION" <<'PY'
import json, pathlib, sys
decision = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not decision.get("stage_b_authorized", False):
    raise SystemExit("Stage B gate is closed by formal/analysis_val/decision.json")
PY
export CHRONOS_PYTHON="$PYTHON_BIN"
export WIND_CR_ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/mechanism/screening"
export WIND_CR_EXPERIMENT_NAMESPACE="cross_resolution_temporal_v2/mechanism/screening"
export WIND_CR_GROUPS="t2_phase_off,t2_linear,t2_no_diff"
export WIND_CR_SEEDS="42,46,50"
bash "$PROJECT_ROOT/shell/launch_cross_resolution_temporal.sh" "${1:-0}"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_cross_resolution_followups.py" \
    --project-root "$PROJECT_ROOT" --stage mechanism
