from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parent.parent
RAW_CACHE = PROJECT / "data" / "model_cache_pad5_f8"
LOCAL_CACHE = PROJECT / "data" / "local_bilinear_56farm_v1"
SOURCE = PROJECT / "data" / "local_state_identifiability_v2"
OUTPUT = PROJECT / "data" / "local_state_spatial_interaction_v1"
SPLITS = ("train", "val", "test")
REQUIRED = (
    "past_weather.npy",
    "past_weather_cycle_index.npy",
    "past_weather_time_offset_hours.npy",
    "past_station_power.npy",
    "strong_sample_donor_index.npy",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    raw_manifest = load_json(RAW_CACHE / "manifest.json")
    local_manifest = load_json(LOCAL_CACHE / "manifest.json")
    source_manifest = load_json(SOURCE / "manifest.json")
    if source_manifest["base_cache_fingerprint"] != local_manifest["preprocess_fingerprint"]:
        raise ValueError("Existing response-state sidecar does not match the local cache")
    if local_manifest["source_preprocess_fingerprint"] != raw_manifest["preprocess_fingerprint"]:
        raise ValueError("Raw-grid and local-bilinear caches do not share provenance")

    split_record = {}
    for split in SPLITS:
        raw_index = np.load(RAW_CACHE / split / "sample_index.npy", mmap_mode="r")
        local_index = np.load(LOCAL_CACHE / split / "sample_index.npy", mmap_mode="r")
        np.testing.assert_array_equal(raw_index, local_index)
        donor = np.load(SOURCE / split / "strong_sample_donor_index.npy", mmap_mode="r")
        if donor.shape != raw_index.shape or np.any(donor == np.arange(len(donor))):
            raise ValueError(f"Invalid donor derangement for {split}")
        target_dir = OUTPUT / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED:
            source = (SOURCE / split / name).resolve()
            target = target_dir / name
            if target.is_symlink():
                if target.resolve() != source:
                    raise FileExistsError(f"Sidecar link points elsewhere: {target}")
            elif target.exists():
                raise FileExistsError(f"Refusing to replace existing sidecar file: {target}")
            else:
                target.symlink_to(source)
        split_record[split] = {
            "samples": int(len(raw_index)),
            "strong_donor_fixed_points": 0,
        }

    identity_payload = {
        "base_cache_fingerprint": raw_manifest["preprocess_fingerprint"],
        "source_sidecar_fingerprint": source_manifest["preprocess_fingerprint"],
        "protocol": "sample-aligned reuse of fixed donor derangement for raw-grid experiment",
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "task_level": "local_state_spatial_interaction",
        **identity_payload,
        "preprocess_fingerprint": identity,
        "past_weather_role": (
            "loader compatibility only; all registered models use local_power state, "
            "not the bilinear historical-weather array"
        ),
        "splits": split_record,
        "source_sidecar": str(SOURCE),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT / f".manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, OUTPUT / "manifest.json")
    (OUTPUT / "COMPLETE").touch()
    print(f"[sidecar] prepared {OUTPUT}")


if __name__ == "__main__":
    main()
