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
ROOT="${WIND_LOCAL_GLOBAL_ROOT:-${PROJECT_ROOT}/artifacts/local_to_global_raw_bilinear_v1}"
EXPERIMENT_ID="${WIND_LOCAL_GLOBAL_EXPERIMENT_ID:-local_to_global_raw_bilinear_v1}"
BASE_CONFIG="${PROJECT_ROOT}/configs/local_to_global_raw_bilinear.yaml"
OUTPUT_DIR="${ROOT}/${STAGE}/${GROUP}/seed${SEED}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "${OUTPUT_DIR}"
if [[ "${SPLIT}" == "train" && -f "${OUTPUT_DIR}/COMPLETE" ]]; then
  if [[ -f "${OUTPUT_DIR}/config.yaml" && -f "${OUTPUT_DIR}/best_model.pt" && \
        -f "${OUTPUT_DIR}/metrics_val.json" && -f "${OUTPUT_DIR}/predictions_val.npz" ]]; then
    exit 0
  fi
  echo "[resume] Incomplete train marker detected; rebuilding ${OUTPUT_DIR}" >&2
  rm -f "${OUTPUT_DIR}/COMPLETE"
fi
if [[ "${SPLIT}" == "test" && -f "${OUTPUT_DIR}/TEST_COMPLETE" ]]; then
  if [[ -f "${OUTPUT_DIR}/best_model.pt" && -f "${OUTPUT_DIR}/metrics_test.json" && \
        -f "${OUTPUT_DIR}/predictions_test.npz" ]]; then
    exit 0
  fi
  echo "[resume] Incomplete test marker detected; rebuilding ${OUTPUT_DIR}" >&2
  rm -f "${OUTPUT_DIR}/TEST_COMPLETE"
fi

STATE_MODE=none
STATE_OPERATOR=none
STATION_MODE=none
STATION_WEIGHT=1.0
LEARNING_RATE=0.0002
BACKBONE_MULTIPLIER=1.0
case "${GROUP}" in
  l1_c)
    STATION_WEIGHT=0.0
    ;;
  l1_m)
    ;;
  l1_mft)
    LEARNING_RATE=0.00002
    ;;
  l2_agg)
    STATE_MODE=aggregate_recent
    STATE_OPERATOR=residual_multiply
    BACKBONE_MULTIPLIER=0.1
    ;;
  l3_local)
    STATE_MODE=local_recent
    STATE_OPERATOR=residual_multiply
    BACKBONE_MULTIPLIER=0.1
    ;;
  g_int)
    STATE_MODE=aggregate_recent
    STATE_OPERATOR=zero_invariant_mlp
    BACKBONE_MULTIPLIER=0.1
    ;;
  l_int)
    STATE_MODE=local_recent
    STATE_OPERATOR=zero_invariant_mlp
    BACKBONE_MULTIPLIER=0.1
    ;;
  g_cat)
    STATE_MODE=aggregate_recent
    STATE_OPERATOR=concat_mlp
    BACKBONE_MULTIPLIER=0.1
    ;;
  l_cat)
    STATE_MODE=local_recent
    STATE_OPERATOR=concat_mlp
    BACKBONE_MULTIPLIER=0.1
    ;;
  l4_id16)
    STATE_MODE=local_recent
    STATE_OPERATOR=residual_multiply
    STATION_MODE=id_embedding
    BACKBONE_MULTIPLIER=0.1
    ;;
  l5_affine)
    STATE_MODE=local_recent
    STATE_OPERATOR=residual_multiply
    STATION_MODE=affine
    BACKBONE_MULTIPLIER=0.1
    ;;
  *)
    echo "unknown group: ${GROUP}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${OUTPUT_DIR}/config.yaml" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_experiment_config.py" \
    --base "${BASE_CONFIG}" \
    --output "${OUTPUT_DIR}/config.yaml" \
    --experiment-id "${EXPERIMENT_ID}/${STAGE}/${GROUP}/seed${SEED}" \
    seed="${SEED}" \
    model.state_mode="${STATE_MODE}" \
    model.state_interaction_operator="${STATE_OPERATOR}" \
    model.station_specific_mode="${STATION_MODE}" \
    model.backbone_lr_multiplier="${BACKBONE_MULTIPLIER}" \
    training.cycles_per_batch="${WIND_LOCAL_CYCLES_PER_BATCH:-12}" \
    training.deterministic.learning_rate="${LEARNING_RATE}" \
    training.deterministic.station_loss_weight="${STATION_WEIGHT}" \
    >"${OUTPUT_DIR}/config.log" 2>&1
else
  echo "[config] Reusing immutable ${OUTPUT_DIR}/config.yaml" >>"${OUTPUT_DIR}/config.log"
fi

if [[ "${SPLIT}" == "train" ]]; then
  TRAIN_ARGS=()
  if [[ -n "${INITIAL_CHECKPOINT}" ]]; then
    TRAIN_ARGS+=(--initial-checkpoint "${INITIAL_CHECKPOINT}")
  fi
  if [[ "${STATE_OPERATOR}" == "zero_invariant_mlp" || "${STATE_OPERATOR}" == "concat_mlp" ]]; then
    TRAIN_ARGS+=(--initial-checkpoint-allow-missing-prefix "state_interaction.")
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
