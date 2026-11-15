#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
BASE_CONFIG="${WIND_BASE_CONFIG:-${PROJECT_ROOT}/configs/incremental_correction_pad5.yaml}"
EXPERIMENT_ROOT="${WIND_EXPERIMENT_ROOT:-${PROJECT_ROOT}/artifacts/incremental_predictability_v1/screening}"
SCREENING_EPOCHS="${WIND_SCREENING_EPOCHS:-50}"
SMOKE_MODE="${WIND_SMOKE:-0}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

PCA_BASIS="${PROJECT_ROOT}/artifacts/incremental_predictability_v1/bases/residual_pca_mw.npy"
RANDOM_BASIS="${PROJECT_ROOT}/artifacts/incremental_predictability_v1/bases/random_orthogonal.npy"
HISTORY_CHECKPOINT="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed42/best_model.pt"

run_candidate() {
  local gpu=$1
  local name=$2
  local projection=$3
  local temporal=$4
  local spatial=$5
  local basis=$6
  local conditioning=$7
  local output_dir="${EXPERIMENT_ROOT}/${name}/seed42"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/COMPLETE" ]]; then
    return
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/materialize_experiment_config.py" \
    --base "${BASE_CONFIG}" \
    --output "${output_dir}/config.yaml" \
    --experiment-id "${name}/seed42" \
    seed=42 \
    model.projection_mode="${projection}" \
    model.temporal_components="${temporal}" \
    model.spatial_components="${spatial}" \
    model.spatial_basis_path="${basis}" \
    model.correction_conditioning="${conditioning}" \
    >"${output_dir}/config.log" 2>&1
  local train_limits=(--epochs "${SCREENING_EPOCHS}")
  local eval_limits=()
  if [[ "${SMOKE_MODE}" == "1" ]]; then
    train_limits=(--epochs 1 --max-train-batches 1 --max-val-batches 1)
    eval_limits=(--max-batches 1)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/train_forced_evolution.py" \
    --config "${output_dir}/config.yaml" \
    --history-checkpoint "${HISTORY_CHECKPOINT}" \
    --output-dir "${output_dir}" \
    --device cuda \
    "${train_limits[@]}" \
    >"${output_dir}/run.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/evaluate_forced_evolution.py" \
    --config "${output_dir}/config.yaml" \
    --checkpoint "${output_dir}/best_model.pt" \
    --split val \
    --output "${output_dir}/metrics_val.json" \
    --predictions-output "${output_dir}/predictions_val.npz" \
    --device cuda \
    "${eval_limits[@]}" \
    >>"${output_dir}/run.log" 2>&1
  touch "${output_dir}/COMPLETE"
}

run_lane() {
  local lane=$1
  case "${lane}" in
    0)
      run_candidate 0 c0_full full 24 8 "${PCA_BASIS}" history_weather
      run_candidate 0 c6_random8 spatial 24 8 "${RANDOM_BASIS}" history_weather
      ;;
    1)
      run_candidate 1 c1_temporal4 temporal 4 8 "${PCA_BASIS}" history_weather
      run_candidate 1 c7_weather_only_full full 24 8 "${PCA_BASIS}" weather_only
      ;;
    2)
      run_candidate 5 c2_spatial2 spatial 24 2 "${PCA_BASIS}" history_weather
      ;;
    3)
      run_candidate 6 c3_spatial8 spatial 24 8 "${PCA_BASIS}" history_weather
      ;;
    4)
      run_candidate 7 c4_joint4_spatial2 joint 4 2 "${PCA_BASIS}" history_weather
      run_candidate 7 c5_joint4_spatial8 joint 4 8 "${PCA_BASIS}" history_weather
      ;;
    *)
      echo "Unknown lane ${lane}" >&2
      return 2
      ;;
  esac
}

mkdir -p "${EXPERIMENT_ROOT}"
pids=()
for lane in 0 1 2 3 4; do
  run_lane "${lane}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
