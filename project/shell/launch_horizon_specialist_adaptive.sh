#!/usr/bin/env bash
set -euo pipefail

GPU_CSV="${1:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
EXPERIMENT_ROOT="${WIND_HS_ROOT:-${PROJECT_ROOT}/artifacts/horizon_specialist_v1}"
DECISION="$EXPERIMENT_ROOT/analysis_screening/decision.json"
[[ -f "$DECISION" ]] || { echo "Missing screening decision: $DECISION" >&2; exit 3; }
EXP_GROUPS=()
[[ "$(jq -r '.run_l4' "$DECISION")" == "true" ]] && EXP_GROUPS+=(l4_learned_view_taper)
[[ "$(jq -r '.run_h2' "$DECISION")" == "true" ]] && EXP_GROUPS+=(h2_dual_learned_taper)
if [[ ${#EXP_GROUPS[@]} -eq 0 ]]; then
    echo "[horizon-specialist] adaptive groups stopped by screening rules"
    touch "$EXPERIMENT_ROOT/ADAPTIVE_NOT_REQUIRED"
    exit 0
fi
GROUP_CSV=$(IFS=,; echo "${EXP_GROUPS[*]}")
WIND_HS_SCREEN_GROUPS="$GROUP_CSV" \
    bash "$PROJECT_ROOT/shell/launch_horizon_specialist_screening.sh" "$GPU_CSV"
touch "$EXPERIMENT_ROOT/ADAPTIVE_COMPLETE"
