#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PHYSICAL_GPU GROUP SEED" >&2
    exit 2
fi
GPU=$1 GROUP=$2 SEED=$3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
MANIFEST="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/formal/run_manifest.json"
SELECTION="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/temporal_selection.json"
RUN_DIR="$($PYTHON_BIN - "$MANIFEST" "$SELECTION" "$GROUP" "$SEED" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
selection = json.loads(pathlib.Path(sys.argv[2]).read_text())
group, seed = sys.argv[3], int(sys.argv[4])
if group == "T2_final":
    matches = [r for r in selection["runs"] if int(r["seed"]) == seed]
else:
    matches = [r for r in manifest["runs"] if r["group"] == group and int(r["seed"]) == seed]
if len(matches) != 1 or not pathlib.Path(matches[0]["directory"], "COMPLETE").is_file():
    raise SystemExit("selected temporal run is missing or invalid")
print(matches[0]["directory"])
PY
)"
OUTPUT_DIR="$PROJECT_ROOT/artifacts/cross_resolution_temporal_v2/robustness/groups/$GROUP/seed$SEED"
mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.run.lock"
flock -x "$LOCK_FD"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    valid=true
    for perturbation in r1_2hour drop1 drop2 drop3 drop4 drop5 drop6; do
        [[ -f "$OUTPUT_DIR/$perturbation/COMPLETE" \
            && -s "$OUTPUT_DIR/$perturbation/metrics_val.json" \
            && -s "$OUTPUT_DIR/$perturbation/predictions_val.npz" ]] || valid=false
    done
    if [[ "$valid" == true ]]; then
        echo "[robustness] audited-reuse group=$GROUP seed=$SEED"
        exit 0
    fi
fi
rm -f "$OUTPUT_DIR/COMPLETE"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_cross_resolution_robustness.py" \
    --config "$RUN_DIR/config.yaml" --checkpoint "$RUN_DIR/best_model.pt" \
    --output-root "$OUTPUT_DIR" --device cuda >"$OUTPUT_DIR/run.log" \
    2> >(tee -a "$OUTPUT_DIR/run.log" >&2)
for perturbation in r1_2hour drop1 drop2 drop3 drop4 drop5 drop6; do
    if [[ ! -f "$OUTPUT_DIR/$perturbation/COMPLETE" \
        || ! -s "$OUTPUT_DIR/$perturbation/metrics_val.json" \
        || ! -s "$OUTPUT_DIR/$perturbation/predictions_val.npz" ]]; then
        echo "[robustness] artifact audit failed group=$GROUP seed=$SEED scenario=$perturbation" >&2
        exit 4
    fi
done
touch "$OUTPUT_DIR/COMPLETE"
echo "[robustness] complete gpu=$GPU group=$GROUP seed=$SEED"
