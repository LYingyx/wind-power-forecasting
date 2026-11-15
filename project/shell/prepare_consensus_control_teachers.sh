#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_CONSENSUS_ROOT:-${PROJECT_ROOT}/artifacts/consensus_controls_v1}"
RECENT_ROOT="${PROJECT_ROOT}/artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h"
FULL_ROOT="${PROJECT_ROOT}/artifacts/conditional_forcing_validation_v1/groups/anchor_020"
TEACHER_ROOT="${EXPERIMENT_ROOT}/teacher"
SEEDS=(42 43 44 45 46 47 48 49 50 51)

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
mkdir -p "$TEACHER_ROOT/single_recent"

FULL_TEACHER="$TEACHER_ROOT/train_teacher_full_history_ensemble.npz"
if [[ ! -f "$FULL_TEACHER" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
        "$PROJECT_ROOT/scripts/build_ensemble_teacher.py" \
        --group-root "$FULL_ROOT" \
        --output "$FULL_TEACHER" \
        --device cuda
fi

for teacher_seed in "${SEEDS[@]}"; do
    output="$TEACHER_ROOT/single_recent/train_teacher_seed${teacher_seed}.npz"
    if [[ -f "$output" ]]; then
        continue
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
        "$PROJECT_ROOT/scripts/build_ensemble_teacher.py" \
        --group-root "$RECENT_ROOT" \
        --output "$output" \
        --seeds "$teacher_seed" \
        --allow-single-member \
        --device cuda
done

touch "$TEACHER_ROOT/COMPLETE"
echo "[consensus-control] teacher preparation complete: $TEACHER_ROOT"
