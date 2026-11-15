#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_CSV="${1:-0}"
export CHRONOS_PYTHON="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WIND_CR_ROOT="${PROJECT_ROOT}/artifacts/cross_resolution_temporal_v2/formal"
export WIND_CR_EXPERIMENT_NAMESPACE="cross_resolution_temporal_v2/formal"
export WIND_CR_GROUPS="t0_raw_linear,t1_latent_linear,t2_bracket_nonlinear"
export WIND_CR_SEEDS="43,44,45,47,48,49,51"

"$CHRONOS_PYTHON" "$PROJECT_ROOT/scripts/build_cross_resolution_manifest.py" \
    --project-root "$PROJECT_ROOT"
bash "$PROJECT_ROOT/shell/launch_cross_resolution_temporal.sh" "$GPU_CSV"
"$CHRONOS_PYTHON" "$PROJECT_ROOT/scripts/build_cross_resolution_manifest.py" \
    --project-root "$PROJECT_ROOT" --require-complete
"$CHRONOS_PYTHON" "$PROJECT_ROOT/scripts/analyze_cross_resolution_formal.py" \
    --project-root "$PROJECT_ROOT"
