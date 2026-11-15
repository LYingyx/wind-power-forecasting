#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/final"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$ROOT/NORMAL_COMPLETE" ]] || { echo "Final Normal checkpoints are incomplete" >&2; exit 3; }
for seed in {42..51}; do
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_cross_resolution_final_run.py" \
        --project-root "$PROJECT_ROOT" --run-dir "$ROOT/groups/normal/seed$seed" --group normal --seed "$seed" >/dev/null
done
TEACHER="$ROOT/teacher/train_teacher_normal.npz"
mkdir -p "$ROOT/teacher"
if [[ ! -s "$TEACHER" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" \
        "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_ensemble_teacher.py" \
        --group-root "$ROOT/groups/normal" --output "$TEACHER" --device cuda
fi
"$PYTHON_BIN" - "$TEACHER" <<'PY'
import numpy as np, pathlib, sys
p=pathlib.Path(sys.argv[1])
with np.load(p) as a:
    assert str(a['teacher_kind'].item()) == 'ensemble_mean'
    np.testing.assert_array_equal(a['teacher_seeds'], np.arange(42,52))
    assert np.isfinite(a['teacher_station_factor_mean']).all()
    assert np.isfinite(a['teacher_cluster_factor_mean']).all()
print(f'[teacher-audit] valid {p}')
PY
touch "$ROOT/teacher/COMPLETE"
