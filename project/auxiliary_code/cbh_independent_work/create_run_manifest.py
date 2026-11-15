from __future__ import annotations

import hashlib
import json
from pathlib import Path

from independent_models import OUT, CACHE, SEEDS, TRAINING_PROTOCOL, atomic_json


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    files = [
        OUT / "code/independent_models.py",
        OUT / "code/prepare_local_weather.py",
        OUT / "code/train_independent_seed.py",
        OUT / "code/launch_64_workers.sh",
        OUT / "code/rescue_failures.py",
        OUT / "code/launch_rescue_workers.sh",
        OUT / "code/launch_gpu6_guarded.sh",
        OUT / "code/audit_preflight.py",
        OUT / "code/aggregate_independent.py",
        OUT / "code/detailed_analysis.py",
        OUT / "code/watch_and_finalize.sh",
        OUT / "code/aggregate_shared.py",
        OUT / "status/SHARED_EXTRACTION_AUDIT.json",
        OUT / "status/LOCAL_WEATHER_AUDIT.json",
        OUT / "status/PREFLIGHT_AUDIT.json",
        OUT / "status/INDEPENDENT_FINAL_AUDIT.json",
        OUT / "status/DETAILED_ANALYSIS_AUDIT.json",
    ]
    manifest = {
        "experiment": "cbh_per_station_independent_20260918_v1",
        "objective": "56-site x 10-seed shared CBH ablation expansion and fully independent site comparison",
        "sites": list(range(56)),
        "seeds": list(SEEDS),
        "training_protocol": TRAINING_PROTOCOL,
        "main_workers": 64,
        "main_gpu_layout": {str(gpu): 8 for gpu in range(8)},
        "rescue_workers": 12,
        "rescue_gpu_layout": {str(gpu): 2 for gpu in (0, 1, 3, 4, 5, 7)},
        "gpu6_guarded_workers": 4,
        "gpu6_guard_minimum_free_mib": 1200,
        "data_manifest": str(CACHE / "manifest.json"),
        "data_manifest_sha256": sha256(CACHE / "manifest.json"),
        "files": {str(path.relative_to(OUT)): sha256(path) for path in files},
        "tmux_sessions": [
            "cbh_independent_all56_10seed",
            "cbh_independent_rescue",
            "cbh_independent_gpu6_guarded",
            "cbh_independent_finalize",
        ],
    }
    atomic_json(OUT / "RUN_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
