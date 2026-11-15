from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import h5py
import numpy as np
from numpy.lib.format import open_memmap

from wind_forcing.models.components import build_station_patch_metadata


SPLITS = ("train", "val", "test")
PAST_NODES = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, values: np.ndarray) -> dict[str, Any]:
    np.save(path, values, allow_pickle=False)
    return {"shape": list(values.shape), "dtype": str(values.dtype), "sha256": sha256(path)}


def deranged_donors(
    origins: np.ndarray, phases: np.ndarray, ages: np.ndarray, seed: int
) -> np.ndarray:
    months = origins.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64) % 12
    strata = np.stack((months, phases.astype(np.int64), ages.astype(np.int64)), axis=1)
    _, inverse = np.unique(strata, axis=0, return_inverse=True)
    donor = np.empty(len(origins), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for value in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == value)
        if len(members) < 2:
            raise ValueError(f"Strong-donor stratum {value} has only {len(members)} sample")
        shuffled = members.copy()
        rng.shuffle(shuffled)
        donor[shuffled] = np.roll(shuffled, int(rng.integers(1, len(shuffled))))
    if np.any(donor == np.arange(len(donor))):
        raise RuntimeError("Strong donor map is not a derangement")
    return donor


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal local Pangu-power state sidecar")
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-cycles", type=int, default=64)
    args = parser.parse_args()
    source = Path(args.source_h5).resolve()
    base = Path(args.base_cache).resolve()
    output = Path(args.output).resolve()
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    identity_payload = {
        "source_h5": str(source),
        "source_size": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "base_cache_fingerprint": base_manifest["preprocess_fingerprint"],
        "past_nodes": PAST_NODES,
        "protocol": "issue<=valid<=origin; bilinear local forcing; fixed stratified derangement",
    }
    fingerprint = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    if (output / "manifest.json").is_file() and (output / "COMPLETE").is_file():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("preprocess_fingerprint") == fingerprint:
            print(f"[local-state-cache] reusing {output}", flush=True)
            return
        raise FileExistsError(f"Stale sidecar exists: {output}")
    if output.exists():
        raise FileExistsError(f"Incomplete sidecar exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    arrays: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    try:
        with h5py.File(source, "r") as handle:
            station_lat = np.asarray(handle["stations/latitude"], dtype=np.float64)
            station_lon = np.asarray(handle["stations/longitude"], dtype=np.float64)
            capacity = np.asarray(handle["stations/capacity_mw"], dtype=np.float32)
            weather_lat = np.asarray(handle["crop/latitude"], dtype=np.float64)
            weather_lon = np.asarray(handle["crop/longitude"], dtype=np.float64)
            patch, _, weights = build_station_patch_metadata(
                weather_lat, weather_lon, station_lat, station_lon, "bilinear", 1
            )
            flat_patch = patch.reshape(-1)
            power_time = np.asarray(handle["power/timestamp_unix"], dtype=np.int64)
            station_mw = np.asarray(handle["power/station_mw"], dtype=np.float32)
            nwp = handle["nwp/values"]
            for split_index, split in enumerate(SPLITS):
                target = temporary / split
                target.mkdir()
                samples = handle[f"splits/{split}"]
                cycles = handle[f"stage1_splits/{split}"]
                base_split = base / split
                base_samples = np.load(base_split / "sample_index.npy", allow_pickle=False)
                source_samples = np.asarray(samples["sample_index"], dtype=np.int64)
                np.testing.assert_array_equal(base_samples, source_samples)
                base_cycles = np.load(base_split / "cycle_sample_index.npy", allow_pickle=False)
                source_cycles = np.asarray(cycles["sample_index"], dtype=np.int64)
                np.testing.assert_array_equal(base_cycles, source_cycles)
                sample_cycle_ids = np.asarray(samples["stage1_cycle_index"], dtype=np.int64)
                cycle_index = np.searchsorted(source_cycles, sample_cycle_ids)
                np.testing.assert_array_equal(source_cycles[cycle_index], sample_cycle_ids)
                frame = np.asarray(cycles["nwp_frame_index"][:, :PAST_NODES], dtype=np.int64)
                valid = np.asarray(cycles["nwp_valid_unix"][:, :PAST_NODES], dtype=np.int64)
                issue = np.asarray(cycles["nwp_issue_unix"][:, :PAST_NODES], dtype=np.int64)
                origin = np.asarray(samples["origin_unix"], dtype=np.int64)
                sample_valid = valid[cycle_index]
                sample_issue = issue[cycle_index]
                if not (
                    np.all(sample_issue <= sample_valid)
                    and np.all(sample_valid <= origin[:, None])
                    and np.all(np.diff(sample_valid, axis=1) == 3600)
                ):
                    raise ValueError(f"Causal historical Pangu audit failed for {split}")

                weather_shape = (len(source_cycles), PAST_NODES, len(capacity), int(nwp.shape[1]))
                weather_path = target / "past_weather.npy"
                weather = open_memmap(weather_path, mode="w+", dtype=np.float32, shape=weather_shape)
                for start in range(0, len(source_cycles), args.chunk_cycles):
                    stop = min(start + args.chunk_cycles, len(source_cycles))
                    indices = frame[start:stop]
                    unique, inverse = np.unique(indices, return_inverse=True)
                    block = np.asarray(nwp[unique], dtype=np.float32)
                    values = block.reshape(len(unique), block.shape[1], -1)[:, :, flat_patch]
                    values = values.reshape(
                        len(unique), block.shape[1], len(capacity), 4
                    ).transpose(0, 2, 3, 1)
                    local = (values * weights[None, :, :, None]).sum(2)
                    weather[start:stop] = local[inverse.reshape(indices.shape)]
                weather.flush()
                del weather

                power_index = np.searchsorted(power_time, sample_valid)
                if np.any(power_index >= len(power_time)) or not np.array_equal(
                    power_time[power_index], sample_valid
                ):
                    raise ValueError(f"Historical power alignment failed for {split}")
                flat_index = power_index.reshape(-1)
                past_station = np.asarray(station_mw[flat_index], dtype=np.float32)
                past_station = past_station.reshape(len(origin), PAST_NODES, len(capacity))
                past_station = (past_station / capacity[None, None]).transpose(0, 2, 1)
                if not np.isfinite(past_station).all():
                    raise ValueError(f"Non-finite historical station power for {split}")
                donor = deranged_donors(
                    origin,
                    np.asarray(samples["origin_phase"], dtype=np.int64),
                    np.asarray(samples["nwp_age_minutes"], dtype=np.int64),
                    seed=1701 + split_index,
                )
                current = np.load(base_split / "station_history.npy", mmap_mode="r")[:, :, -1]
                local_corr = float(np.corrcoef(current.reshape(-1), current[donor].reshape(-1))[0, 1])
                cluster = np.load(base_split / "cluster_history.npy", mmap_mode="r")[:, -1]
                global_corr = float(np.corrcoef(cluster, cluster[donor])[0, 1])
                values_to_save = {
                    "past_weather_cycle_index": cycle_index.astype(np.int64),
                    "past_weather_time_offset_hours": (
                        (sample_valid - origin[:, None]).astype(np.float32) / 3600.0
                    ),
                    "past_station_power": past_station.astype(np.float32),
                    "strong_sample_donor_index": donor,
                    "past_weather_valid_unix": valid,
                    "past_weather_issue_unix": issue,
                }
                arrays.append(
                    {
                        "relative_path": f"{split}/past_weather.npy",
                        "shape": list(weather_shape),
                        "dtype": "float32",
                        "sha256": sha256(weather_path),
                    }
                )
                for name, value in values_to_save.items():
                    path = target / f"{name}.npy"
                    arrays.append({"relative_path": f"{split}/{path.name}", **save(path, value)})
                summaries[split] = {
                    "samples": len(origin),
                    "cycles": len(source_cycles),
                    "maximum_issue_minus_origin_seconds": int((sample_issue - origin[:, None]).max()),
                    "maximum_valid_minus_origin_seconds": int((sample_valid - origin[:, None]).max()),
                    "strong_donor_fixed_points": int(np.sum(donor == np.arange(len(donor)))),
                    "strong_donor_local_current_correlation": local_corr,
                    "strong_donor_cluster_current_correlation": global_corr,
                }
        manifest = {
            "schema_version": 1,
            "task_level": "local_state_identifiability",
            **identity_payload,
            "preprocess_fingerprint": fingerprint,
            "time_protocol": "all historical issue <= valid <= forecast origin",
            "donor_protocol": "UTC month x origin phase x NWP age fixed derangement",
            "splits": summaries,
            "arrays": arrays,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (temporary / "COMPLETE").touch()
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"[local-state-cache] ready {output}", flush=True)


if __name__ == "__main__":
    main()
