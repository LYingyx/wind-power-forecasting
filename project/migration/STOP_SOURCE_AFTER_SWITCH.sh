#!/usr/bin/env bash
set -euo pipefail

# Run only after the destination has passed its smoke checks and is running.
for session in wcf_cross_resume_20260921 wcf_fewshot_20260921; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux kill-session -t "${session}"
    echo "stopped ${session}"
  fi
done
