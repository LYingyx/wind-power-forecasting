#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU STAGE GROUP SEED SPLIT" >&2
  exit 2
fi
GPU=$1
STAGE=$2
GROUP=$3
SEED=$4
SPLIT=$5
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
ROOT="${WIND_SPATIAL_INTERACTION_ROOT:-${PROJECT_ROOT}/artifacts/local_state_spatial_interaction_v1}"
BASE_CONFIG="${PROJECT_ROOT}/configs/local_state_spatial_interaction_v1.yaml"
OUTPUT_DIR="${ROOT}/${STAGE}/${GROUP}/seed${SEED}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

case "${GROUP}" in
  bilinear_none|bilinear_add|bilinear_int)
    SPATIAL_MODE=raw_bilinear
    ;;
  fixed4_none|fixed4_add|fixed4_int)
    SPATIAL_MODE=fixed_gaussian_4x4
    ;;
  learned4_none|learned4_add|learned4_int)
    SPATIAL_MODE=learned_4x4_wind
    ;;
  *)
    echo "unknown group: ${GROUP}" >&2
    exit 2
    ;;
esac

STATE_SOURCE=none
INTERACTION_MODE=none
case "${GROUP}" in
  *_none)
    ;;
  *_add)
    STATE_SOURCE=local_power
    INTERACTION_MODE=output_additive
    ;;
  *_int)
    STATE_SOURCE=local_power
    INTERACTION_MODE=faithful_concat
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
if [[ "${SPLIT}" == "train" && -f "${OUTPUT_DIR}/COMPLETE" && \
      -f "${OUTPUT_DIR}/best_model.pt" && -f "${OUTPUT_DIR}/metrics_val.json" && \
      -f "${OUTPUT_DIR}/predictions_val.npz" ]]; then
  exit 0
fi
if [[ "${SPLIT}" == "test" && -f "${OUTPUT_DIR}/TEST_COMPLETE" && \
      -f "${OUTPUT_DIR}/metrics_test.json" && -f "${OUTPUT_DIR}/predictions_test.npz" ]]; then
  exit 0
fi

if [[ ! -f "${OUTPUT_DIR}/config.yaml" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_experiment_config.py" \
    --base "${BASE_CONFIG}" \
    --output "${OUTPUT_DIR}/config.yaml" \
    --experiment-id "local_state_spatial_interaction_v1/${STAGE}/${GROUP}/seed${SEED}" \
    seed="${SEED}" \
    model.spatial_weather_mode="${SPATIAL_MODE}" \
    model.response_state_source="${STATE_SOURCE}" \
    model.response_interaction_mode="${INTERACTION_MODE}" \
    training.cycles_per_batch="${WIND_SPATIAL_INTERACTION_CYCLES_PER_BATCH:-3}" \
    >"${OUTPUT_DIR}/config.log" 2>&1
else
  echo "[config] Reusing immutable ${OUTPUT_DIR}/config.yaml" >>"${OUTPUT_DIR}/config.log"
fi

if [[ "${SPLIT}" == "train" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/train_forced_evolution.py" \
    --config "${OUTPUT_DIR}/config.yaml" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --epochs 40 \
    >"${OUTPUT_DIR}/run.log" 2>&1
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/evaluate_forced_evolution.py" \
    --config "${OUTPUT_DIR}/config.yaml" \
    --checkpoint "${OUTPUT_DIR}/best_model.pt" \
    --split val \
    --output "${OUTPUT_DIR}/metrics_val.json" \
    --predictions-output "${OUTPUT_DIR}/predictions_val.npz" \
    --export-station-predictions \
    --device cuda \
    >>"${OUTPUT_DIR}/run.log" 2>&1
  touch "${OUTPUT_DIR}/COMPLETE"
else
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/evaluate_forced_evolution.py" \
    --config "${OUTPUT_DIR}/config.yaml" \
    --checkpoint "${OUTPUT_DIR}/best_model.pt" \
    --split test \
    --output "${OUTPUT_DIR}/metrics_test.json" \
    --predictions-output "${OUTPUT_DIR}/predictions_test.npz" \
    --export-station-predictions \
    --device cuda \
    >"${OUTPUT_DIR}/test.log" 2>&1
  touch "${OUTPUT_DIR}/TEST_COMPLETE"
fi
