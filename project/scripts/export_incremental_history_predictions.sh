#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/liying/anaconda3/envs/torch241/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/incremental_predictability_v1}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

declare -a JOBS=(
  "42 val 0"
  "43 val 1"
  "44 val 5"
  "42 train 3"
  "43 train 6"
  "44 train 7"
)

mkdir -p "${ARTIFACT_ROOT}/exports"
pids=()
for job in "${JOBS[@]}"; do
  read -r seed split gpu <<<"${job}"
  source_dir="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed${seed}"
  output_dir="${ARTIFACT_ROOT}/exports/history_seed${seed}"
  mkdir -p "${output_dir}"
  if [[ -s "${output_dir}/predictions_${split}_station.npz" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/evaluate_forced_evolution.py" \
    --config "${source_dir}/config.yaml" \
    --checkpoint "${source_dir}/best_model.pt" \
    --split "${split}" \
    --device cuda \
    --output "${output_dir}/metrics_${split}.json" \
    --predictions-output "${output_dir}/predictions_${split}_station.npz" \
    --export-station-predictions \
    >"${output_dir}/export_${split}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
exit "${failed}"
