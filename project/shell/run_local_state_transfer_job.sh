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
ROOT="${WIND_STATE_TRANSFER_ROOT:-${PROJECT_ROOT}/artifacts/local_state_transfer_multistage_v1}"
BASE_CONFIG="${PROJECT_ROOT}/configs/local_state_transfer_multistage_v1.yaml"
OUTPUT_DIR="${ROOT}/${STAGE}/${GROUP}/seed${SEED}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

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

STATE_SOURCE=none
INTERACTION_MODE=none
FINE_TUNE_POLICY=all
STATION_WEIGHT=1.0
CLUSTER_WEIGHT=2.0
CHECKPOINT_METRIC=cluster_mae_mw
LEARNING_RATE=0.0002
BACKBONE_MULTIPLIER=1.0

case "${GROUP}" in
  s0|e0|p0_free)
    ;;
  s_add|e_add|p_add_free)
    STATE_SOURCE=local_power
    INTERACTION_MODE=additive_main_effect
    ;;
  s_int|e_int|p_int_free|p_int_preserve)
    STATE_SOURCE=local_power
    INTERACTION_MODE=faithful_concat
    ;;
  *)
    echo "unknown group: ${GROUP}" >&2
    exit 2
    ;;
esac

if [[ "${GROUP}" == s* ]]; then
  CLUSTER_WEIGHT=0.0
  CHECKPOINT_METRIC=station_mae_cf
fi
if [[ "${GROUP}" == p* ]]; then
  LEARNING_RATE=0.00005
fi
if [[ "${GROUP}" == "p_int_preserve" ]]; then
  FINE_TUNE_POLICY=freeze_state_response
fi

if [[ ! -f "${OUTPUT_DIR}/config.yaml" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_experiment_config.py" \
    --base "${BASE_CONFIG}" \
    --output "${OUTPUT_DIR}/config.yaml" \
    --experiment-id "local_state_transfer_multistage_v1/${STAGE}/${GROUP}/seed${SEED}" \
    seed="${SEED}" \
    model.response_state_source="${STATE_SOURCE}" \
    model.response_interaction_mode="${INTERACTION_MODE}" \
    model.fine_tune_policy="${FINE_TUNE_POLICY}" \
    model.backbone_lr_multiplier="${BACKBONE_MULTIPLIER}" \
    training.cycles_per_batch="${WIND_STATE_TRANSFER_CYCLES_PER_BATCH:-6}" \
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
