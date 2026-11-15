#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
DRY_RUN="${2:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${WIND_HS_ROOT:-${PROJECT_ROOT}/artifacts/horizon_specialist_v1}"
DECISION="$EXPERIMENT_ROOT/analysis_screening/decision.json"
[[ -f "$DECISION" ]] || { echo "Missing final screening decision: $DECISION" >&2; exit 3; }
LIGHT=$(jq -r '.selected_lightweight // empty' "$DECISION")
HIGH=$(jq -r '.selected_high_cost // empty' "$DECISION")
EXP_GROUPS=()
[[ -z "$LIGHT" ]] || EXP_GROUPS+=("$LIGHT")
if [[ -n "$HIGH" ]]; then EXP_GROUPS+=(h0_dual_full_taper "$HIGH"); fi
if [[ ${#EXP_GROUPS[@]} -eq 0 ]]; then
    echo "[horizon-specialist] no candidate passed screening; formal stage stopped"
    touch "$EXPERIMENT_ROOT/FORMAL_STOPPED"
    exit 0
fi
GROUP_CSV=$(IFS=,; echo "${EXP_GROUPS[*]}")
ARGS=("$GPU_CSV")
[[ "$DRY_RUN" == "--dry-run" ]] && ARGS+=(--dry-run)
WIND_HS_SCREEN_GROUPS="$GROUP_CSV" \
WIND_HS_SCREEN_SEEDS="42,43,44,45,46,47,48,49,50,51" \
WIND_HS_REQUIRED_SEED_COUNT=10 \
WIND_HS_STAGE_LABEL=formal \
WIND_HS_APPEND_HIGH_COST=0 \
    bash "$PROJECT_ROOT/shell/launch_horizon_specialist_screening.sh" "${ARGS[@]}"
