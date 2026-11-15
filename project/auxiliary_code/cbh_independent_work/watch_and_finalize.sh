#!/usr/bin/env bash
set -euo pipefail

ROOT=/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/cbh_per_station_independent_20260918_v1
PYTHON=/data/liying/anaconda3/envs/chronos/bin/python

while true; do
  successes=$(find "$ROOT/independent_seed" -name SUCCESS.json 2>/dev/null | wc -l)
  failures=$(find "$ROOT/status/failures" -name '*.json' 2>/dev/null | wc -l)
  workers=$(find "$ROOT/status/workers" -name 'worker_*.json' 2>/dev/null | wc -l)
  printf '{"timestamp":"%s","successes":%s,"failures_waiting":%s,"main_workers_finished":%s}\n' "$(date --iso-8601=seconds)" "$successes" "$failures" "$workers" > "$ROOT/status/PROGRESS.json.tmp"
  mv "$ROOT/status/PROGRESS.json.tmp" "$ROOT/status/PROGRESS.json"
  if [[ "$successes" -eq 560 ]]; then
    break
  fi
  sleep 60
done

cd "$ROOT/code"
"$PYTHON" aggregate_independent.py > "$ROOT/logs/finalize.log" 2>&1
