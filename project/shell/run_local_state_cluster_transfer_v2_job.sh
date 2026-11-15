#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 GPU STAGE GROUP SEED INITIAL_CHECKPOINT EPOCHS SPLIT" >&2
  exit 2
fi
GPU=$1
STAGE=$2
GROUP=$3
SEED=$4
INITIAL_CHECKPOINT=$5
EPOCHS=$6
SPLIT=$7
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}"
ROOT="${WIND_CLUSTER_TRANSFER_V2_ROOT:-${PROJECT_ROOT}/artifacts/local_state_cluster_transfer_v2}"
BASE_CONFIG="${PROJECT_ROOT}/configs/local_state_cluster_transfer_v2.yaml"
OUTPUT_DIR="${ROOT}/${STAGE}/${GROUP}/seed${SEED}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

case "${GROUP}" in
  b_*) SPATIAL_MODE=raw_bilinear ;;
  f_*) SPATIAL_MODE=fixed_gaussian_4x4 ;;
  *) echo "unknown spatial group: ${GROUP}" >&2; exit 2 ;;
esac

case "${GROUP}" in
  *_shared_add) INTERACTION_MODE=output_additive ;;
  *_independent_add) INTERACTION_MODE=independent_output_additive ;;
  *_int) INTERACTION_MODE=faithful_concat ;;
  *) echo "unknown response group: ${GROUP}" >&2; exit 2 ;;
esac

STATION_WEIGHT=1.0
if [[ "${STAGE}" == "station" ]]; then
  CLUSTER_WEIGHT=0.0
  CHECKPOINT_METRIC=station_mae_cf
  LEARNING_RATE=0.0002
elif [[ "${STAGE}" == "transfer" ]]; then
  CLUSTER_WEIGHT=2.0
  CHECKPOINT_METRIC=cluster_mae_mw
  LEARNING_RATE=0.00005
else
  echo "unknown stage: ${STAGE}" >&2
  exit 2
fi

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
    --experiment-id "local_state_cluster_transfer_v2/${STAGE}/${GROUP}/seed${SEED}" \
    seed="${SEED}" \
    model.spatial_weather_mode="${SPATIAL_MODE}" \
    model.response_interaction_mode="${INTERACTION_MODE}" \
    training.cycles_per_batch="${WIND_CLUSTER_TRANSFER_V2_CYCLES_PER_BATCH:-3}" \
    training.deterministic.learning_rate="${LEARNING_RATE}" \
    training.deterministic.station_loss_weight="${STATION_WEIGHT}" \
    training.deterministic.cluster_loss_weight="${CLUSTER_WEIGHT}" \
    training.deterministic.checkpoint_metric="${CHECKPOINT_METRIC}" \
    >"${OUTPUT_DIR}/config.log" 2>&1
else
  echo "[config] Reusing immutable ${OUTPUT_DIR}/config.yaml" >>"${OUTPUT_DIR}/config.log"
fi

if [[ "${SPLIT}" == "train" ]]; then
  TRAIN_ARGS=()
  if [[ -n "${INITIAL_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--initial-checkpoint "${INITIAL_CHECKPOINT}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/train_forced_evolution.py" \
    --config "${OUTPUT_DIR}/config.yaml" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --epochs "${EPOCHS}" \
    "${TRAIN_ARGS[@]}" \
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
