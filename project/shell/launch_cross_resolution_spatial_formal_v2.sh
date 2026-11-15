#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
SCREEN_DECISION="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/spatial/analysis_screening/decision.json"
SELECTION="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/temporal_selection.json"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
GROUP_CSV="$("$PYTHON_BIN" - "$SCREEN_DECISION" <<'PY'
import json, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(",".join(data["formal_expansion_groups"]))
PY
)"
if [[ -n "$GROUP_CSV" ]]; then
    export CHRONOS_PYTHON="$PYTHON_BIN"
    export WIND_CR_ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/spatial/formal"
    export WIND_CR_EXPERIMENT_NAMESPACE="cross_resolution_temporal_v2/spatial/formal"
    export WIND_CR_GROUPS="$GROUP_CSV"
    export WIND_CR_SEEDS="43,44,45,47,48,49,51"
    export WIND_CR_TIME_QUERY_MODE="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resolve_cross_resolution_selection.py" --project-root "$PROJECT_ROOT" --field selected_time_query_mode)"
    export WIND_CR_BRACKET_TIME_FEATURE_MODE="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resolve_cross_resolution_selection.py" --project-root "$PROJECT_ROOT" --field selected_bracket_time_feature_mode)"
    export WIND_CR_BRACKET_MAPPER_ACTIVATION="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resolve_cross_resolution_selection.py" --project-root "$PROJECT_ROOT" --field selected_bracket_mapper_activation)"
    export WIND_CR_FORCING_DIFFERENCE_INPUT="$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/resolve_cross_resolution_selection.py" --project-root "$PROJECT_ROOT" --field selected_forcing_difference_input)"
    bash "$PROJECT_ROOT/shell/launch_cross_resolution_temporal.sh" "${1:-0}"
else
    echo "[spatial-formal] no learned projection passed the preregistered expansion gate"
fi
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_cross_resolution_spatial_formal.py" \
    --project-root "$PROJECT_ROOT"
touch "$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/spatial/SPATIAL_STRUCTURE_FROZEN"
