#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DECISION="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/mechanism/analysis_screening/decision.json"
GROUP_CSV="$($PYTHON_BIN - "$DECISION" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(",".join(data["formal_expansion_groups"]))
PY
)"
if [[ -z "$GROUP_CSV" ]]; then
    echo "[mechanism-formal] no mechanism passed the preregistered expansion gate"
else
    export CHRONOS_PYTHON="$PYTHON_BIN"
    export WIND_CR_ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/mechanism/formal"
    export WIND_CR_EXPERIMENT_NAMESPACE="cross_resolution_temporal_v2/mechanism/formal"
    export WIND_CR_GROUPS="$GROUP_CSV"
    export WIND_CR_SEEDS="43,44,45,47,48,49,51"
    bash "$PROJECT_ROOT/shell/launch_cross_resolution_temporal.sh" "${1:-0}"
fi
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_cross_resolution_mechanism_formal.py" \
    --project-root "$PROJECT_ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/resolve_cross_resolution_selection.py" \
    --project-root "$PROJECT_ROOT" \
    --output "$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/temporal_selection.json"
