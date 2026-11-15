#!/usr/bin/env bash
set -u
ROOT=/storage/liying/new_try/wind_cluster_forced_evolution
OUT=$ROOT/artifacts/trajectory_conditioned_correction_20260913
LOG=$OUT/hourly_monitor.log
while true; do
  ts=$(date '+%F %T %Z')
  procs=$(pgrep -fc 'train_forced_evolution.py|launch_trajectory_conditioned' || true)
  best=$(find "$OUT" -type f -name best_model.pt 2>/dev/null | wc -l)
  complete=$(find "$OUT" -type f -name COMPLETE 2>/dev/null | wc -l)
  errors=$(grep -RIlE 'Traceback|CUDA out of memory|FileNotFoundError|RuntimeError' "$OUT" --include='launcher.log' 2>/dev/null | wc -l)
  printf '%s processes=%s best_model=%s complete=%s error_logs=%s\n' "$ts" "$procs" "$best" "$complete" "$errors" >> "$LOG"
  sleep 3600
done
