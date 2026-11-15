#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then echo "usage: $0 PHYSICAL_GPU normal|kd SEED" >&2; exit 2; fi
GPU=$1 GROUP=$2 SEED=$3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
ROOT="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/final"
DECISION="$ROOT/analysis_val/decision.json"
RUN_DIR="$ROOT/groups/$GROUP/seed$SEED"
"$PYTHON_BIN" - "$DECISION" <<'PY'
import json, pathlib, sys
d=json.loads(pathlib.Path(sys.argv[1]).read_text())
if not d.get('architecture_and_representative_frozen_for_2024'):
    raise SystemExit('2024 gate is closed')
PY
[[ -f "$RUN_DIR/COMPLETE" ]] || { echo "Final validation run is incomplete: $RUN_DIR" >&2; exit 3; }
exec {LOCK_FD}>"$RUN_DIR/.test.lock"; flock -x "$LOCK_FD"
if [[ -f "$RUN_DIR/TEST_COMPLETE" && -s "$RUN_DIR/metrics_test.json" && -s "$RUN_DIR/predictions_test.npz" ]]; then
    echo "[cross-resolution-test] reuse group=$GROUP seed=$SEED"; exit 0
fi
rm -f "$RUN_DIR/TEST_COMPLETE"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$RUN_DIR/config.yaml" --checkpoint "$RUN_DIR/best_model.pt" --split test \
    --output "$RUN_DIR/metrics_test.json" --predictions-output "$RUN_DIR/predictions_test.npz" \
    --device cuda >"$RUN_DIR/test.log" 2> >(tee -a "$RUN_DIR/test.log" >&2)
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json, numpy as np, pathlib, sys
d=pathlib.Path(sys.argv[1])
m=json.loads((d/'metrics_test.json').read_text())
assert np.isfinite(float(m['cluster_mae_mw']))
with np.load(d/'predictions_test.npz') as a:
    p,t=a['cluster_prediction_mw'],a['cluster_target_mw']
    assert p.shape == t.shape and p.ndim == 2 and p.shape[1] == 24
    assert np.isfinite(p).all() and np.isfinite(t).all()
PY
touch "$RUN_DIR/TEST_COMPLETE"
echo "[cross-resolution-test] complete gpu=$GPU group=$GROUP seed=$SEED"
