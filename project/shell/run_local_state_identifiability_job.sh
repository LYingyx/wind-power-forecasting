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
ROOT="${WIND_STATE_ID_ROOT:-${PROJECT_ROOT}/artifacts/local_state_identifiability_v2}"
BASE_CONFIG="${PROJECT_ROOT}/configs/local_state_identifiability_v2.yaml"
OUTPUT_DIR="${ROOT}/${STAGE}/${GROUP}/seed${SEED}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

mkdir -p "${OUTPUT_DIR}"
if [[ "${SPLIT}" == train && -f "${OUTPUT_DIR}/COMPLETE" ]]; then
  [[ -f "${OUTPUT_DIR}/best_model.pt" && -f "${OUTPUT_DIR}/metrics_val.json" && -f "${OUTPUT_DIR}/predictions_val.npz" ]] && exit 0
  rm -f "${OUTPUT_DIR}/COMPLETE"
fi
if [[ "${SPLIT}" == test && -f "${OUTPUT_DIR}/TEST_COMPLETE" ]]; then
  [[ -f "${OUTPUT_DIR}/metrics_test.json" && -f "${OUTPUT_DIR}/predictions_test.npz" ]] && exit 0
  rm -f "${OUTPUT_DIR}/TEST_COMPLETE"
fi

STATE=none
CONTEXT=none
INTERACTION=none
VISIBLE=4
DONOR=none
ID_WEIGHT=0.0
case "${GROUP}" in
  b0) ;;
  b_static) CONTEXT=static; INTERACTION=strict ;;
  p_noid) STATE=local_power; INTERACTION=strict ;;
  p_static|p4_static) STATE=local_power; CONTEXT=static; INTERACTION=strict ;;
  p_direct) STATE=local_power; CONTEXT=static; INTERACTION=direct ;;
  g_static) STATE=aggregate_power; CONTEXT=static; INTERACTION=strict ;;
  gl_static|hierarchical_static) STATE=hierarchical; CONTEXT=static; INTERACTION=strict ;;
  p12_static) STATE=local_power; CONTEXT=static; INTERACTION=strict; VISIBLE=12 ;;
  physical_static) STATE=physical; CONTEXT=static; INTERACTION=strict; VISIBLE=12 ;;
  io_pair_static) STATE=io_pair; CONTEXT=static; INTERACTION=strict; VISIBLE=12 ;;
  io_pair_id_static) STATE=io_pair_id; CONTEXT=static; INTERACTION=strict; VISIBLE=12; ID_WEIGHT=0.2 ;;
  id_only) CONTEXT=learned_id; INTERACTION=strict ;;
  affine_only) CONTEXT=static_affine; INTERACTION=strict ;;
  p_id) STATE=local_power; CONTEXT=learned_id; INTERACTION=strict ;;
  p_affine) STATE=local_power; CONTEXT=static_affine; INTERACTION=strict ;;
  *_shuf)
    DONOR=strong_sample
    BASE_GROUP=${GROUP%_shuf}
    case "${BASE_GROUP}" in
      p_noid) STATE=local_power; INTERACTION=strict ;;
      p_static|p4_static) STATE=local_power; CONTEXT=static; INTERACTION=strict ;;
      p_direct) STATE=local_power; CONTEXT=static; INTERACTION=direct ;;
      g_static) STATE=aggregate_power; CONTEXT=static; INTERACTION=strict ;;
      gl_static|hierarchical_static) STATE=hierarchical; CONTEXT=static; INTERACTION=strict ;;
      p12_static) STATE=local_power; CONTEXT=static; INTERACTION=strict; VISIBLE=12 ;;
      physical_static) STATE=physical; CONTEXT=static; INTERACTION=strict; VISIBLE=12 ;;
      io_pair_static) STATE=io_pair; CONTEXT=static; INTERACTION=strict; VISIBLE=12 ;;
      io_pair_id_static) STATE=io_pair_id; CONTEXT=static; INTERACTION=strict; VISIBLE=12; ID_WEIGHT=0.2 ;;
      p_id) STATE=local_power; CONTEXT=learned_id; INTERACTION=strict ;;
      p_affine) STATE=local_power; CONTEXT=static_affine; INTERACTION=strict ;;
      *) echo "unknown shuffled group: ${GROUP}" >&2; exit 2 ;;
    esac
    ;;
  *) echo "unknown group: ${GROUP}" >&2; exit 2 ;;
esac

if [[ ! -f "${OUTPUT_DIR}/config.yaml" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_experiment_config.py" \
    --base "${BASE_CONFIG}" --output "${OUTPUT_DIR}/config.yaml" \
    --experiment-id "local_state_identifiability_v2/${STAGE}/${GROUP}/seed${SEED}" \
    seed="${SEED}" \
    data.response_state_donor_mode="${DONOR}" \
    model.response_state_source="${STATE}" \
    model.station_context_mode="${CONTEXT}" \
    model.response_interaction_mode="${INTERACTION}" \
    model.state_visible_steps="${VISIBLE}" \
    training.cycles_per_batch="${WIND_STATE_CYCLES_PER_BATCH:-8}" \
    training.deterministic.response_identification_loss_weight="${ID_WEIGHT}" \
    >"${OUTPUT_DIR}/config.log" 2>&1
fi

if [[ "${SPLIT}" == train ]]; then
  INIT_ARGS=()
  if [[ -n "${INITIAL_CHECKPOINT}" ]]; then
    INIT_ARGS+=(--initial-checkpoint "${INITIAL_CHECKPOINT}")
    for prefix in global_state_encoder. hierarchical_state_projection. physical_state_encoder. io_token_projection. io_state_trajectory. static_context_encoder. response_station_id. response_station_id_projection. response_interaction. direct_state_head. identification_head. station_static_features far_station_index; do
      INIT_ARGS+=(--initial-checkpoint-allow-missing-prefix "${prefix}")
    done
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/train_forced_evolution.py" \
    --config "${OUTPUT_DIR}/config.yaml" --output-dir "${OUTPUT_DIR}" --device cuda \
    --epochs "${EPOCHS}" "${INIT_ARGS[@]}" >"${OUTPUT_DIR}/run.log" 2>&1
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_forced_evolution.py" \
    --config "${OUTPUT_DIR}/config.yaml" --checkpoint "${OUTPUT_DIR}/best_model.pt" \
    --split val --output "${OUTPUT_DIR}/metrics_val.json" \
    --predictions-output "${OUTPUT_DIR}/predictions_val.npz" --export-station-predictions \
    --device cuda >>"${OUTPUT_DIR}/run.log" 2>&1
  touch "${OUTPUT_DIR}/COMPLETE"
else
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluate_forced_evolution.py" \
    --config "${OUTPUT_DIR}/config.yaml" --checkpoint "${OUTPUT_DIR}/best_model.pt" \
    --split test --output "${OUTPUT_DIR}/metrics_test.json" \
    --predictions-output "${OUTPUT_DIR}/predictions_test.npz" --export-station-predictions \
    --device cuda >"${OUTPUT_DIR}/test.log" 2>&1
  touch "${OUTPUT_DIR}/TEST_COMPLETE"
fi
