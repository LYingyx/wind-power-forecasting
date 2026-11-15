#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
DRY_RUN="${2:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_HS_ROOT:-${PROJECT_ROOT}/artifacts/horizon_specialist_v1}"
SCREEN_DECISION="$EXPERIMENT_ROOT/analysis_screening/decision.json"
FORMAL_ANALYSIS="$EXPERIMENT_ROOT/analysis_val/analysis.json"
[[ -f "$SCREEN_DECISION" ]] || { echo "Missing screening decision" >&2; exit 3; }
[[ -f "$FORMAL_ANALYSIS" ]] || { echo "Missing formal validation analysis" >&2; exit 3; }
LIGHT=$(jq -r '.selected_lightweight // empty' "$SCREEN_DECISION")
[[ -n "$LIGHT" ]] || { echo "No lightweight candidate was selected" >&2; exit 3; }
SUCCESS=$(jq -r --arg group "$LIGHT" \
    '[.comparisons[] | select(.candidate==$group and .reference=="Normal") | (.proceed_to_kd // .formal_success)] | first // false' \
    "$FORMAL_ANALYSIS")
if [[ "$SUCCESS" != "true" ]]; then
    echo "[horizon-specialist] KD stopped: $LIGHT passed neither superiority nor the pre-test single-model non-inferiority amendment"
    touch "$EXPERIMENT_ROOT/KD_STOPPED"
    exit 0
fi
TEACHER="$EXPERIMENT_ROOT/teacher/train_teacher_${LIGHT}.npz"
KD_GROUP="kd_${LIGHT}"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[horizon-specialist] would build teacher=$TEACHER and train group=$KD_GROUP"
    exit 0
fi
IFS=',' read -r -a GPUS <<<"$GPU_CSV"
mkdir -p "$EXPERIMENT_ROOT/teacher"
if [[ ! -s "$TEACHER" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" \
        "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_ensemble_teacher.py" \
        --group-root "$EXPERIMENT_ROOT/groups/$LIGHT" \
        --output "$TEACHER" --device cuda
fi
WIND_HS_SCREEN_GROUPS="$KD_GROUP" \
WIND_HS_SCREEN_SEEDS="42,43,44,45,46,47,48,49,50,51" \
WIND_HS_REQUIRED_SEED_COUNT=10 WIND_HS_STAGE_LABEL=kd WIND_HS_APPEND_HIGH_COST=0 \
WIND_HS_JOB_TEACHER_ARCHIVE="$TEACHER" \
    bash "$PROJECT_ROOT/shell/launch_horizon_specialist_screening.sh" "$GPU_CSV"
