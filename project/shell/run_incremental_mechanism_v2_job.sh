#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 GPU KIND GROUP SEED [DOTTED.OVERRIDE=VALUE ...]" >&2
    exit 2
fi

GPU=$1
KIND=$2
GROUP=$3
SEED=$4
shift 4

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_INCREMENTAL_V2_ROOT:-${PROJECT_ROOT}/artifacts/incremental_mechanism_v2}"
EPOCHS="${WIND_INCREMENTAL_V2_EPOCHS:-80}"
HISTORY_SOURCE="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only/seed42/config.yaml"
INCREMENTAL_SOURCE="${PROJECT_ROOT}/artifacts/incremental_predictability_v1/confirmation/joint_full/seed42/config.yaml"
LEGACY_HISTORY_ROOT="${PROJECT_ROOT}/artifacts/paper_experiments_pad5_v1/fusion/t1_history_only"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"

case "$KIND" in
    history)
        BASE_CONFIG="$HISTORY_SOURCE"
        OUTPUT_DIR="${EXPERIMENT_ROOT}/history_prior/seed${SEED}"
        ;;
    direct|incremental)
        BASE_CONFIG="$INCREMENTAL_SOURCE"
        OUTPUT_DIR="${EXPERIMENT_ROOT}/groups/${GROUP}/seed${SEED}"
        ;;
    *)
        echo "Unsupported job kind: $KIND" >&2
        exit 2
        ;;
esac

mkdir -p "$OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/COMPLETE" ]]; then
    echo "[incremental-v2] reuse kind=$KIND group=$GROUP seed=$SEED"
    exit 0
fi

COMMON_OVERRIDES=("seed=${SEED}")
if [[ "$KIND" == "direct" ]]; then
    COMMON_OVERRIDES+=(
        "model.architecture=forced_evolution"
        "training.deterministic.baseline_auxiliary_loss_weight=0.0"
    )
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/materialize_experiment_config.py" \
    --base "$BASE_CONFIG" \
    --output "$OUTPUT_DIR/config.yaml" \
    --experiment-id "${GROUP}/seed${SEED}" \
    "${COMMON_OVERRIDES[@]}" "$@" \
    >"$OUTPUT_DIR/config.log" 2>&1

TRAIN_ARGS=(
    "$PROJECT_ROOT/scripts/train_forced_evolution.py"
    --config "$OUTPUT_DIR/config.yaml"
    --output-dir "$OUTPUT_DIR"
    --device cuda
    --epochs "$EPOCHS"
)
if [[ "$KIND" == "incremental" ]]; then
    if [[ -f "${LEGACY_HISTORY_ROOT}/seed${SEED}/best_model.pt" ]]; then
        HISTORY_CHECKPOINT="${LEGACY_HISTORY_ROOT}/seed${SEED}/best_model.pt"
    else
        HISTORY_CHECKPOINT="${EXPERIMENT_ROOT}/history_prior/seed${SEED}/best_model.pt"
    fi
    if [[ ! -f "$HISTORY_CHECKPOINT" ]]; then
        echo "Missing history checkpoint for seed ${SEED}: ${HISTORY_CHECKPOINT}" >&2
        exit 3
    fi
    TRAIN_ARGS+=(--history-checkpoint "$HISTORY_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "${TRAIN_ARGS[@]}" \
    >"$OUTPUT_DIR/run.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/evaluate_forced_evolution.py" \
    --config "$OUTPUT_DIR/config.yaml" \
    --checkpoint "$OUTPUT_DIR/best_model.pt" \
    --split val \
    --output "$OUTPUT_DIR/metrics_val.json" \
    --predictions-output "$OUTPUT_DIR/predictions_val.npz" \
    --device cuda \
    >>"$OUTPUT_DIR/run.log" 2>&1

touch "$OUTPUT_DIR/COMPLETE"
echo "[incremental-v2] complete kind=$KIND group=$GROUP seed=$SEED"
