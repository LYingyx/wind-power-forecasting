#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
BASE_CONFIG="${PROJECT_ROOT}/configs/incremental_correction_pad5.yaml"
EXPERIMENT_ROOT="${WIND_EXPERIMENT_ROOT:-${PROJECT_ROOT}/artifacts/incremental_predictability_v1/screening}"
EPOCHS="${WIND_SCREENING_EPOCHS:-50}"
HISTORY_CHECKPOINT="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed42/best_model.pt"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

run_one() {
  local gpu=$1
  local name=$2
  local projection=$3
  local temporal=$4
  local output_dir="${EXPERIMENT_ROOT}/${name}/seed42"
  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_experiment_config.py" \
    --base "${BASE_CONFIG}" \
    --output "${output_dir}/config.yaml" \
    --experiment-id "${name}/seed42" \
    seed=42 \
    model.freeze_history_baseline=false \
    model.projection_mode="${projection}" \
    model.temporal_components="${temporal}" \
    training.deterministic.baseline_auxiliary_loss_weight=0.5 \
    >"${output_dir}/config.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/train_forced_evolution.py" \
    --config "${output_dir}/config.yaml" \
    --history-checkpoint "${HISTORY_CHECKPOINT}" \
    --output-dir "${output_dir}" \
    --device cuda \
    --epochs "${EPOCHS}" \
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

run_one 3 c8_joint_refine_full full 24 & pid0=$!
run_one 4 c9_joint_refine_temporal4 temporal 4 & pid1=$!
failed=0
for pid in "${pid0}" "${pid1}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
