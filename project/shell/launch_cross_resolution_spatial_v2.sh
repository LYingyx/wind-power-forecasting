#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELECTION="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/temporal_selection.json"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/resolve_cross_resolution_selection.py" \
    --project-root "$PROJECT_ROOT" --output "$SELECTION" >/dev/null
"$PYTHON_BIN" - "$SELECTION" <<'PY'
import json, pathlib, sys
decision = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not decision.get("temporal_structure_frozen", False):
    raise SystemExit("Spatial gate is closed: the temporal structure is not frozen")
PY
export CHRONOS_PYTHON="$PYTHON_BIN"
export WIND_CR_ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/spatial/screening"
export WIND_CR_EXPERIMENT_NAMESPACE="cross_resolution_temporal_v2/spatial/screening"
export WIND_CR_GROUPS="s0_raw_bilinear,s1_latent_bilinear,s2_learned_2x2,s3_fixed_gaussian_4x4,s4_learned_4x4_nogeom"
export WIND_CR_SEEDS="42,46,50"
export WIND_CR_TIME_QUERY_MODE="$("$PYTHON_BIN" - "$SELECTION" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_time_query_mode"])
PY
)"
export WIND_CR_BRACKET_TIME_FEATURE_MODE="$("$PYTHON_BIN" - "$SELECTION" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_bracket_time_feature_mode"])
PY
)"
export WIND_CR_BRACKET_MAPPER_ACTIVATION="$("$PYTHON_BIN" - "$SELECTION" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_bracket_mapper_activation"])
PY
)"
export WIND_CR_FORCING_DIFFERENCE_INPUT="$("$PYTHON_BIN" - "$SELECTION" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_forcing_difference_input"])
PY
)"
bash "$PROJECT_ROOT/shell/launch_cross_resolution_temporal.sh" "${1:-0}"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_cross_resolution_followups.py" \
    --project-root "$PROJECT_ROOT" --stage spatial
