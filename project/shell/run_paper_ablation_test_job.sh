#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then echo "usage: $0 PHYSICAL_GPU GROUP SEED SPLIT" >&2; exit 2; fi
GPU=$1; GROUP=$2; SEED=$3; SPLIT=$4
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
ROOT="${WIND_PAPER_ABLATION_ROOT:-${PROJECT_ROOT}/artifacts/paper_ablation_completion_v1}"; RUN_DIR="$ROOT/groups/$GROUP/seed$SEED"; MARKER="$RUN_DIR/${SPLIT^^}_COMPLETE"
for file in COMPLETE config.yaml best_model.pt; do if [[ ! -f "$RUN_DIR/$file" ]]; then echo "Missing trained artifact: $RUN_DIR/$file" >&2; exit 3; fi; done
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" HDF5_USE_FILE_LOCKING=FALSE CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$RUN_DIR"; exec {LOCK_FD}>"$RUN_DIR/.${SPLIT}_evaluation.lock"; flock -x "$LOCK_FD"
if [[ -f "$MARKER" ]]; then echo "[ablation-$SPLIT] reuse group=$GROUP seed=$SEED"; exit 0; fi
LOG="$RUN_DIR/evaluate_${SPLIT}.log"; COMMAND=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" --config "$RUN_DIR/config.yaml" --checkpoint "$RUN_DIR/best_model.pt" --split "$SPLIT" --output "$RUN_DIR/metrics_${SPLIT}.json" --predictions-output "$RUN_DIR/predictions_${SPLIT}.npz" --device cuda)
if [[ "${WIND_TQDM_VISIBLE:-1}" == "1" ]]; then "${COMMAND[@]}" >"$LOG" 2> >(tee -a "$LOG" >&2); else "${COMMAND[@]}" >"$LOG" 2>&1; fi
touch "$MARKER"; echo "[ablation-$SPLIT] complete gpu=$GPU group=$GROUP seed=$SEED"
