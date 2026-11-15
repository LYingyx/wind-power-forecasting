#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 9 && $# -ne 11 ]]; then
  echo "usage: $0 GPU STAGE GROUP SEED HISTORY_MODE FORCING_MODE CONTEXT_MODE INTERACTION_MODE SPLIT [READOUT OUTPUT_MODE]" >&2
  exit 2
fi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}" \
  "${PROJECT_ROOT}/scripts/run_farm_interaction_enhancement_v1_job.py" "$@"
