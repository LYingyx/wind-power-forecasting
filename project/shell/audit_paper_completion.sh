#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION_ROOT="${WIND_PAPER_ABLATION_ROOT:-$PROJECT_ROOT/artifacts/paper_ablation_completion_v1}"
EVIDENCE_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-$PROJECT_ROOT/artifacts/paper_evidence_test_v1}"
ARCH=(nearest_hour independent_weather_path no_wind_geometry fixed_bilinear no_aux_weather no_anchor frozen_baseline)
KD=(kd_station_only kd_cluster_only kd_no_aux_weather kd_no_anchor)
count_marker() { find "$1" -name "$2" 2>/dev/null | wc -l; }
printf '=== Ablation training (validation-selected checkpoints) ===\n'
for group in "${ARCH[@]}" "${KD[@]}"; do printf '%-28s %2s/10 complete, %2s/10 test-evaluated\n' "$group" "$(count_marker "$ABLATION_ROOT/groups/$group" COMPLETE)" "$(count_marker "$ABLATION_ROOT/groups/$group" TEST_COMPLETE)"; done
printf '\n=== Held-out test evidence ===\n'
for label in Direct FullHistory Normal KD; do printf '%-28s %2s/10 cluster exports\n' "$label" "$(count_marker "$EVIDENCE_ROOT/cluster_exports/$label" TEST_COMPLETE)"; done
printf '%-28s %s station exports\n' 'Direct/Normal/KD' "$(count_marker "$EVIDENCE_ROOT/station_exports" TEST_COMPLETE)"
printf '%-28s %s/100 permutation evals\n' 'Normal/KD weather' "$(count_marker "$EVIDENCE_ROOT/weather_robustness" TEST_COMPLETE)"
printf '%-28s %s PNG, %s PDF\n' 'test figures' "$(find "$EVIDENCE_ROOT" -name '*.png' 2>/dev/null | wc -l)" "$(find "$EVIDENCE_ROOT" -name '*.pdf' 2>/dev/null | wc -l)"
printf '\nLegacy validation figures remain in: %s\n' "$PROJECT_ROOT/artifacts/paper_evidence_v1"
printf 'Final test evidence belongs in:      %s\n' "$EVIDENCE_ROOT"
