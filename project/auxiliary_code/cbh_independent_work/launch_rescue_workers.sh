#!/usr/bin/env bash
set -euo pipefail

ROOT=/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/cbh_per_station_independent_20260918_v1
PYTHON=/data/liying/anaconda3/envs/chronos/bin/python
SESSION=cbh_independent_rescue

mkdir -p "$ROOT/logs" "$ROOT/status/rescue_claims"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi

first=1
for gpu in 0 1 3 4 5 7; do
  command="cd '$ROOT/code';"
  for slot in 0 1; do
    rescue="gpu${gpu}_${slot}"
    log="$ROOT/logs/rescue_${rescue}.log"
    command+=" env CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '$PYTHON' rescue_failures.py --rescue-id '$rescue' >'$log' 2>&1 &"
  done
  command+=" wait"
  if [[ "$first" -eq 1 ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$command"
    first=0
  else
    tmux new-window -t "$SESSION" -n "gpu${gpu}" "$command"
  fi
done

tmux list-windows -t "$SESSION"
