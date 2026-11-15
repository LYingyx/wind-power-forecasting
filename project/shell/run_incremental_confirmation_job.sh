#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU GROUP SOURCE_CANDIDATE SEED" >&2
  exit 2
fi
GPU=$1
GROUP=$2
SOURCE_CANDIDATE=$3
SEED=$4
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
SCREENING_ROOT="${WIND_SCREENING_ROOT:-${PROJECT_ROOT}/artifacts/incremental_predictability_v1/screening}"
EXPERIMENT_ROOT="${WIND_EXPERIMENT_ROOT:-${PROJECT_ROOT}/artifacts/incremental_predictability_v1/confirmation}"
EPOCHS="${WIND_CONFIRMATION_EPOCHS:-80}"
SOURCE_CONFIG="${SCREENING_ROOT}/${SOURCE_CANDIDATE}/seed42/config.yaml"
HISTORY_CHECKPOINT="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed${SEED}/best_model.pt"
OUTPUT_DIR="${EXPERIMENT_ROOT}/${GROUP}/seed${SEED}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/COMPLETE" ]]; then
  exit 0
fi
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_experiment_config.py" \
  --base "${SOURCE_CONFIG}" \
  --output "${OUTPUT_DIR}/config.yaml" \
  --experiment-id "${GROUP}/seed${SEED}" \
  seed="${SEED}" \
  >"${OUTPUT_DIR}/config.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/train_forced_evolution.py" \
  --config "${OUTPUT_DIR}/config.yaml" \
  --history-checkpoint "${HISTORY_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --epochs "${EPOCHS}" \
  >"${OUTPUT_DIR}/run.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/evaluate_forced_evolution.py" \
  --config "${OUTPUT_DIR}/config.yaml" \
  --checkpoint "${OUTPUT_DIR}/best_model.pt" \
  --split val \
  --output "${OUTPUT_DIR}/metrics_val.json" \
  --predictions-output "${OUTPUT_DIR}/predictions_val.npz" \
  --device cuda \
  >>"${OUTPUT_DIR}/run.log" 2>&1
touch "${OUTPUT_DIR}/COMPLETE"
