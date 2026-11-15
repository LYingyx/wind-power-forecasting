#!/usr/bin/env bash
set -euo pipefail

ROOT=/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/cbh_per_station_independent_20260918_v1
PYTHON=/data/liying/anaconda3/envs/chronos/bin/python
SESSION=cbh_independent_all56_10seed

mkdir -p "$ROOT/logs" "$ROOT/status/workers" "$ROOT/status/failures"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi

for gpu in $(seq 0 7); do
  command="cd '$ROOT/code';"
  for local_worker in $(seq 0 7); do
    worker=$((gpu * 8 + local_worker))
    log="$ROOT/logs/worker_${worker}_gpu${gpu}.log"
    command+=" env CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '$PYTHON' train_independent_seed.py --worker-id $worker --worker-count 64 >'$log' 2>&1 &"
  done
  command+=" wait"
  if [[ "$gpu" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$command"
  else
    tmux new-window -t "$SESSION" -n "gpu${gpu}" "$command"
  fi
done

tmux list-windows -t "$SESSION"
