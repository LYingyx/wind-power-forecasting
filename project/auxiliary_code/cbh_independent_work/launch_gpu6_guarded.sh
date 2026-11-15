#!/usr/bin/env bash
set -euo pipefail

ROOT=/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/cbh_per_station_independent_20260918_v1
PYTHON=/data/liying/anaconda3/envs/chronos/bin/python
SESSION=cbh_independent_gpu6_guarded

command="cd '$ROOT/code';"
for slot in 0 1 2 3; do
  rescue="gpu6_guarded_${slot}"
  log="$ROOT/logs/rescue_${rescue}.log"
  command+=" env CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '$PYTHON' rescue_failures.py --rescue-id '$rescue' --physical-gpu 6 --min-free-mib 1200 >'$log' 2>&1 &"
done
command+=" wait"

tmux new-session -d -s "$SESSION" -n gpu6 "$command"
