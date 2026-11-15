#!/usr/bin/env bash
set -euo pipefail

# Install a relocatable snapshot and resume both registered queues.
# Usage on the destination host:
#   bash migration/install_and_resume.sh /storage/liying/new_try/wind_cluster_forced_evolution

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${1:?destination project root is required}"
ENV_ROOT="${DEST_ROOT}/.migration_env/chronos"
OLD_ROOT="/storage/liying/new_try/wind_cluster_forced_evolution"
OLD_PYTHON="/data/liying/anaconda3/envs/chronos/bin/python"

if [[ ! -f "${BUNDLE_ROOT}/BUNDLE_COMPLETE" ]]; then
  echo "BUNDLE_COMPLETE is missing; run this script from the extracted bundle." >&2
  exit 2
fi

mkdir -p "${DEST_ROOT}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${BUNDLE_ROOT}/project/" "${DEST_ROOT}/"
else
  cp -a "${BUNDLE_ROOT}/project/." "${DEST_ROOT}/"
fi

mkdir -p "${DEST_ROOT}/.migration_env"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  mkdir -p "${ENV_ROOT}"
  tar --zstd -xf "${BUNDLE_ROOT}/env/chronos.tar.zst" -C "${ENV_ROOT}"
  if [[ -x "${ENV_ROOT}/bin/conda-unpack" ]]; then
    "${ENV_ROOT}/bin/conda-unpack" || true
  fi
fi

# Rewrite only text source files; binary checkpoints and arrays are untouched.
DEST_ROOT="${DEST_ROOT}" ENV_ROOT="${ENV_ROOT}" OLD_ROOT="${OLD_ROOT}" OLD_PYTHON="${OLD_PYTHON}" \
"${ENV_ROOT}/bin/python" "${BUNDLE_ROOT}/migration/relocate_paths.py"

# A live source snapshot can contain running rows.  Requeue only non-completed
# rows so the destination safely resumes without duplicating completed outputs.
"${ENV_ROOT}/bin/python" - "${DEST_ROOT}" <<'PY'
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1])
databases = [
    root / "artifacts/cross_farm_shared_specific_20260920_v1/status/transfer_hparam.sqlite",
    root / "artifacts/cross_farm_shared_specific_20260920_v1/status/transfer_dev.sqlite",
    root / "artifacts/cross_farm_shared_specific_20260920_v1/status/transfer_final.sqlite",
    root / "artifacts/selected30_fewshot_baselines_20260920_v1/status/jobs.sqlite",
]
for path in databases:
    if not path.exists():
        continue
    db = sqlite3.connect(path, timeout=120)
    db.execute("""update jobs set status='pending', attempts=0, worker=null, pid=null,
                 started=null, finished=null, returncode=null, error_tail=null
                 where status in ('running','failed')""")
    db.commit()
    db.close()
PY

CROSS_STATUS="${DEST_ROOT}/artifacts/cross_farm_shared_specific_20260920_v1/status"
FEW_STATUS="${DEST_ROOT}/artifacts/selected30_fewshot_baselines_20260920_v1/status"
rm -f "${CROSS_STATUS}/"*.pid "${FEW_STATUS}/"*.pid
rm -f "${CROSS_STATUS}/TRANSFER_CONTROLLER_STOPPED.json" "${FEW_STATUS}/CONTROLLER_STOPPED.json"
rm -rf "${CROSS_STATUS}/gpu_locks"
mkdir -p "${CROSS_STATUS}/gpu_locks"

"${ENV_ROOT}/bin/python" - "${DEST_ROOT}" <<'PY'
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1])
marker = root / "MIGRATION_INSTALLED.json"
marker.write_text(json.dumps({
    "installed_at": time.time(),
    "project_root": str(root),
    "gpus": [0, 3, 4, 5, 6, 7],
    "workers_per_gpu": 8,
    "resume_mode": "shared per-GPU lock with 4096 MiB free-memory gate",
}, indent=2) + "\n")
PY

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for persistent background execution on the destination." >&2
  exit 3
fi

cd "${DEST_ROOT}"
if ! tmux has-session -t wcf_cross_migrated 2>/dev/null; then
  tmux new-session -d -s wcf_cross_migrated \
    "cd '${DEST_ROOT}' && exec '${ENV_ROOT}/bin/python' artifacts/cross_farm_shared_specific_20260920_v1/code/experiment_watchdog.py >> artifacts/cross_farm_shared_specific_20260920_v1/logs/watchdog_migrated.log 2>&1"
fi
if ! tmux has-session -t wcf_fewshot_migrated 2>/dev/null; then
  tmux new-session -d -s wcf_fewshot_migrated \
    "cd '${DEST_ROOT}' && exec '${ENV_ROOT}/bin/python' artifacts/selected30_fewshot_baselines_20260920_v1/code/watchdog.py >> artifacts/selected30_fewshot_baselines_20260920_v1/logs/watchdog_migrated.log 2>&1"
fi

echo "Installed at: ${DEST_ROOT}"
echo "Environment:  ${ENV_ROOT}"
echo "Sessions:     wcf_cross_migrated, wcf_fewshot_migrated"
echo "GPU policy:   0,3,4,5,6,7; 8 workers/GPU; shared lock; 4096 MiB gate"
