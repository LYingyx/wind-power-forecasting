#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
SCREENING_ROOT="${WIND_SCREENING_ROOT:-${PROJECT_ROOT}/artifacts/incremental_predictability_v1/screening}"
EXPERIMENT_ROOT="${WIND_EXPERIMENT_ROOT:-${PROJECT_ROOT}/artifacts/incremental_predictability_v1/confirmation}"
SELECTED_CANDIDATE="${WIND_SELECTED_CANDIDATE:?Set WIND_SELECTED_CANDIDATE to the selected screening directory name}"
CONFIRMATION_EPOCHS="${WIND_CONFIRMATION_EPOCHS:-80}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

run_confirmation() {
  local gpu=$1
  local group=$2
  local source_candidate=$3
  local seed=$4
  local source_config="${SCREENING_ROOT}/${source_candidate}/seed42/config.yaml"
  local history_checkpoint="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed${seed}/best_model.pt"
  local output_dir="${EXPERIMENT_ROOT}/${group}/seed${seed}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/COMPLETE" ]]; then
    return
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_experiment_config.py" \
    --base "${source_config}" \
    --output "${output_dir}/config.yaml" \
    --experiment-id "${group}/seed${seed}" \
    seed="${seed}" \
    >"${output_dir}/config.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/train_forced_evolution.py" \
    --config "${output_dir}/config.yaml" \
    --history-checkpoint "${history_checkpoint}" \
    --output-dir "${output_dir}" \
    --device cuda \
    --epochs "${CONFIRMATION_EPOCHS}" \
    >"${output_dir}/run.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/evaluate_forced_evolution.py" \
    --config "${output_dir}/config.yaml" \
    --checkpoint "${output_dir}/best_model.pt" \
    --split val \
    --output "${output_dir}/metrics_val.json" \
    --predictions-output "${output_dir}/predictions_val.npz" \
    --device cuda \
    >>"${output_dir}/run.log" 2>&1
  touch "${output_dir}/COMPLETE"
}

mkdir -p "${EXPERIMENT_ROOT}"
run_confirmation 0 full c0_full 42 & pid0=$!
run_confirmation 1 full c0_full 43 & pid1=$!
run_confirmation 3 full c0_full 44 & pid2=$!
run_confirmation 5 selected "${SELECTED_CANDIDATE}" 42 & pid3=$!
run_confirmation 6 selected "${SELECTED_CANDIDATE}" 43 & pid4=$!
run_confirmation 7 selected "${SELECTED_CANDIDATE}" 44 & pid5=$!
failed=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}" "${pid4}" "${pid5}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
