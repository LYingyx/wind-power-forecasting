#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 PHYSICAL_GPU LABEL GROUP_ROOT SEED SPLIT" >&2
    exit 2
fi
GPU=$1; LABEL=$2; GROUP_ROOT=$3; SEED=$4; SPLIT=$5
if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then echo "split must be val or test" >&2; exit 2; fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
OUTPUT_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-${PROJECT_ROOT}/artifacts/paper_evidence_test_v1}"
RUN_DIR="${GROUP_ROOT}/seed${SEED}"
OUTPUT_DIR="${OUTPUT_ROOT}/cluster_exports/${LABEL}/seed${SEED}"
MARKER="${OUTPUT_DIR}/${SPLIT^^}_COMPLETE"
for file in config.yaml best_model.pt; do if [[ ! -f "$RUN_DIR/$file" ]]; then echo "Missing $RUN_DIR/$file" >&2; exit 3; fi; done
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
mkdir -p "$OUTPUT_DIR"
exec {LOCK_FD}>"$OUTPUT_DIR/.${SPLIT}_evaluation.lock"; flock -x "$LOCK_FD"
if [[ -f "$MARKER" ]]; then echo "[test-export] reuse label=$LABEL seed=$SEED split=$SPLIT"; exit 0; fi
LOG="$OUTPUT_DIR/run_${SPLIT}.log"
COMMAND=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" --config "$RUN_DIR/config.yaml" --checkpoint "$RUN_DIR/best_model.pt" --split "$SPLIT" --output "$OUTPUT_DIR/metrics_${SPLIT}.json" --predictions-output "$OUTPUT_DIR/predictions_${SPLIT}.npz" --device cuda)
if [[ "${WIND_TQDM_VISIBLE:-1}" == "1" ]]; then
    "${COMMAND[@]}" >"$LOG" 2> >(tee -a "$LOG" >&2)
else
    "${COMMAND[@]}" >"$LOG" 2>&1
fi
touch "$MARKER"
echo "[test-export] complete physical_gpu=$GPU label=$LABEL seed=$SEED split=$SPLIT"
