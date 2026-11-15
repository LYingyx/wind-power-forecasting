#!/usr/bin/env bash
set -euo pipefail

GPU_CSV=${1:-0}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
bash "$PROJECT_ROOT/shell/launch_cross_resolution_final_group_v2.sh" normal "$GPU_CSV"
FIRST_GPU=${GPU_CSV%%,*}
bash "$PROJECT_ROOT/shell/prepare_cross_resolution_final_teacher_v2.sh" "$FIRST_GPU"
bash "$PROJECT_ROOT/shell/launch_cross_resolution_final_group_v2.sh" kd "$GPU_CSV"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_cross_resolution_final_validation.py" --project-root "$PROJECT_ROOT"
