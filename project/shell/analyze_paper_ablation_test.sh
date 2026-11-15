#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON_BIN="${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/torch241/bin/python}"
ABLATION_ROOT="${WIND_PAPER_ABLATION_ROOT:-$PROJECT_ROOT/artifacts/paper_ablation_completion_v1}"; EVIDENCE_ROOT="${WIND_PAPER_EVIDENCE_ROOT:-$PROJECT_ROOT/artifacts/paper_evidence_test_v1}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"; cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/analyze_paper_ablation_completion.py --split test --root "$ABLATION_ROOT" --direct-reference "$EVIDENCE_ROOT/cluster_exports/Direct" --full-history-reference "$EVIDENCE_ROOT/cluster_exports/FullHistory" --normal-reference "$EVIDENCE_ROOT/cluster_exports/Normal" --kd-reference "$EVIDENCE_ROOT/cluster_exports/KD" --output-dir "$ABLATION_ROOT/analysis_test" "$@"
