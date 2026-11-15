from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from numpy.lib.format import open_memmap
import torch
from torch.utils.data import DataLoader, Dataset

from wind_forcing.data.dataset import CycleGroupedBatchSampler


SPLITS = ("train", "val", "test")
HISTORY_NWP_NODES = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, *, mmap: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def _save(path: Path, values: np.ndarray) -> dict[str, Any]:
    np.save(path, values, allow_pickle=False)
    return {
        "relative_path": "",
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": _sha256(path),
    }


class AggregateResponseCacheBuilder:
    """Build causal historical Pangu--aggregate-power pairs from the source HDF5."""

    def __init__(
        self,
        source_h5: str | Path,
        aggregate_cache: str | Path,
        destination: str | Path,
        *,
        chunk_cycles: int = 32,
    ) -> None:
        self.source_h5 = Path(source_h5).resolve()
        self.aggregate_cache = Path(aggregate_cache).resolve()
        self.destination = Path(destination).resolve()
        self.chunk_cycles = int(chunk_cycles)
        if self.chunk_cycles < 1:
            raise ValueError("chunk_cycles must be positive")

    def _aggregate_manifest(self) -> dict[str, Any]:
        with (self.aggregate_cache / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != 2 or manifest.get("task_level") != "aggregate_cluster":
            raise ValueError("Expected the audited aggregate-cluster schema-v2 cache")
        return manifest

    def _identity(self, aggregate: Mapping[str, Any]) -> str:
        stat = self.source_h5.stat()
        payload = {
            "source_h5": str(self.source_h5),
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "aggregate_fingerprint": aggregate["preprocess_fingerprint"],
            "history_nwp_nodes": HISTORY_NWP_NODES,
            "crop": aggregate["crop"],
            "protocol": "historical_issue_and_realized_cluster_power_pairs",
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def is_current(self) -> bool:
        manifest_path = self.destination / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            aggregate = self._aggregate_manifest()
            if manifest.get("schema_version") != 1:
                return False
            if manifest.get("preprocess_fingerprint") != self._identity(aggregate):
                return False
            for entry in manifest["arrays"]:
                array = _load(self.destination / entry["relative_path"], mmap=True)
                if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                    return False
            return True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def build(self) -> Path:
        aggregate = self._aggregate_manifest()
        if self.is_current():
            print(f"[response-cache] reusing {self.destination}", flush=True)
            return self.destination
        if self.destination.exists():
            raise FileExistsError(
                f"Stale response cache exists: {self.destination}. Move it aside before rebuilding."
            )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.destination.name}.tmp.", dir=self.destination.parent)
        )
        try:
            manifest = self._build_into(temporary, aggregate)
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
            )
            os.replace(temporary, self.destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(f"[response-cache] ready {self.destination}", flush=True)
        return self.destination

    def _build_into(self, root: Path, aggregate: Mapping[str, Any]) -> dict[str, Any]:
        arrays: list[dict[str, Any]] = []
        split_summary: dict[str, Any] = {}
        crop = aggregate["crop"]
        row_slice = slice(int(crop["row_start"]), int(crop["row_stop"]))
        column_slice = slice(int(crop["column_start"]), int(crop["column_stop"]))
        total_capacity = float(aggregate["total_capacity_mw"])
        with h5py.File(self.source_h5, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != 3:
                raise ValueError("Response cache requires the source schema-v3 HDF5")
            nwp_values = handle["nwp/values"]
            for split in SPLITS:
                output = root / split
                output.mkdir()
                sample_group = handle[f"splits/{split}"]
                cycle_group = handle[f"stage1_splits/{split}"]
                aggregate_dir = self.aggregate_cache / split
                aggregate_sample = _load(aggregate_dir / "sample_index.npy")
                source_sample = np.asarray(sample_group["sample_index"], dtype=np.int64)
                if not np.array_equal(aggregate_sample, source_sample):
                    raise ValueError(f"Aggregate/source sample order mismatch for {split}")
                aggregate_cycle_ids = _load(aggregate_dir / "cycle_sample_index.npy")
                source_cycle_ids = np.asarray(cycle_group["sample_index"], dtype=np.int64)
                if not np.array_equal(aggregate_cycle_ids, source_cycle_ids):
                    raise ValueError(f"Aggregate/source cycle order mismatch for {split}")
                sample_cycle_ids = np.asarray(sample_group["stage1_cycle_index"], dtype=np.int64)
                cycle_index = np.searchsorted(source_cycle_ids, sample_cycle_ids)
                if (
                    np.any(cycle_index >= len(source_cycle_ids))
                    or not np.array_equal(source_cycle_ids[cycle_index], sample_cycle_ids)
                ):
                    raise ValueError(f"Response cycle mapping is invalid for {split}")

                frame_index = np.asarray(
                    cycle_group["nwp_frame_index"][:, :HISTORY_NWP_NODES], dtype=np.int64
                )
                valid_unix = np.asarray(
                    cycle_group["nwp_valid_unix"][:, :HISTORY_NWP_NODES], dtype=np.int64
                )
                issue_unix = np.asarray(
                    cycle_group["nwp_issue_unix"][:, :HISTORY_NWP_NODES], dtype=np.int64
                )
                lead_index = np.asarray(
                    cycle_group["nwp_lead_index"][:, :HISTORY_NWP_NODES], dtype=np.int64
                )
                origins = _load(aggregate_dir / "origin_unix.npy")
                sample_valid = valid_unix[cycle_index]
                sample_issue = issue_unix[cycle_index]
                if not np.all(sample_issue <= sample_valid):
                    raise ValueError(f"Historical Pangu issue is later than validity for {split}")
                if not np.all(sample_valid <= origins[:, None]):
                    raise ValueError(f"Historical Pangu validity exceeds origin for {split}")
                if not np.all(sample_issue <= origins[:, None]):
                    raise ValueError(f"Historical Pangu issue exceeds origin for {split}")
                if not np.all(np.diff(sample_valid, axis=1) == 3600):
                    raise ValueError(f"Historical Pangu nodes are not hourly for {split}")

                power_factor = np.asarray(sample_group["history_node_power"], dtype=np.float32)
                if power_factor.shape != (len(origins), HISTORY_NWP_NODES):
                    raise ValueError(f"Historical node-power shape is invalid for {split}")
                power_mw = power_factor * total_capacity
                if not np.isfinite(power_mw).all():
                    raise ValueError(f"Historical node power is non-finite for {split}")

                weather_shape = (
                    len(source_cycle_ids), HISTORY_NWP_NODES, int(nwp_values.shape[1]),
                    int(crop["shape"][0]), int(crop["shape"][1]),
                )
                weather_path = output / "past_weather.npy"
                weather = open_memmap(weather_path, mode="w+", dtype=np.float32, shape=weather_shape)
                for start in range(0, len(source_cycle_ids), self.chunk_cycles):
                    stop = min(start + self.chunk_cycles, len(source_cycle_ids))
                    indices = frame_index[start:stop]
                    unique, inverse = np.unique(indices, return_inverse=True)
                    frames = np.asarray(nwp_values[unique], dtype=np.float32)
                    frames = frames[:, :, row_slice, column_slice]
                    packed = frames[inverse.reshape(indices.shape)]
                    if not np.isfinite(packed).all():
                        raise ValueError(f"Historical weather contains non-finite values for {split}")
                    weather[start:stop] = packed
                weather.flush()
                del weather

                values = {
                    "past_weather_valid_unix": valid_unix,
                    "past_weather_issue_unix": issue_unix,
                    "past_weather_lead_index": lead_index,
                    "past_weather_cycle_index": cycle_index.astype(np.int64),
                    "past_weather_time_offset_hours": (
                        (sample_valid - origins[:, None]).astype(np.float32) / 3600.0
                    ),
                    "past_cluster_power_mw": power_mw.astype(np.float32),
                    "sample_index": source_sample,
                }
                weather_entry = {
                    "relative_path": f"{split}/past_weather.npy",
                    "shape": list(weather_shape),
                    "dtype": "float32",
                    "sha256": _sha256(weather_path),
                }
                arrays.append(weather_entry)
                for name, value in values.items():
                    path = output / f"{name}.npy"
                    entry = _save(path, value)
                    entry["relative_path"] = f"{split}/{name}.npy"
                    arrays.append(entry)
                split_summary[split] = {
                    "sample_count": int(len(origins)),
                    "cycle_count": int(len(source_cycle_ids)),
                    "past_weather_shape": list(weather_shape[1:]),
                    "past_power_shape": list(power_mw.shape[1:]),
                    "maximum_issue_minus_valid_seconds": int((sample_issue - sample_valid).max()),
                    "maximum_valid_minus_origin_seconds": int((sample_valid - origins[:, None]).max()),
                }
        return {
            "schema_version": 1,
            "task_level": "aggregate_response_state",
            "source_h5": str(self.source_h5),
            "aggregate_cache": str(self.aggregate_cache),
            "aggregate_fingerprint": aggregate["preprocess_fingerprint"],
            "preprocess_fingerprint": self._identity(aggregate),
            "station_sample_tensors_present": False,
            "history_nwp_nodes": HISTORY_NWP_NODES,
            "time_protocol": "all historical issue <= valid <= forecast origin",
            "power_storage_unit": "MW",
            "splits": split_summary,
            "arrays": arrays,
        }

    def audit(self, *, verify_checksums: bool = False) -> dict[str, Any]:
        with (self.destination / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("task_level") != "aggregate_response_state":
            raise ValueError("Wrong response-cache task level")
        if manifest.get("station_sample_tensors_present") is not False:
            raise ValueError("Response cache must not contain station sample tensors")
        checked = 0
        for entry in manifest["arrays"]:
            path = self.destination / entry["relative_path"]
            array = _load(path, mmap=True)
            if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                raise ValueError(f"Response-cache metadata mismatch: {entry['relative_path']}")
            if verify_checksums and _sha256(path) != entry["sha256"]:
                raise ValueError(f"Response-cache checksum mismatch: {entry['relative_path']}")
            checked += 1
        for split in SPLITS:
            directory = self.destination / split
            if any("station" in path.name for path in directory.glob("*.npy")):
                raise ValueError(f"Station tensor found in response cache: {split}")
            origin = _load(self.aggregate_cache / split / "origin_unix.npy")
            cycle = _load(directory / "past_weather_cycle_index.npy")
            valid = _load(directory / "past_weather_valid_unix.npy")[cycle]
            issue = _load(directory / "past_weather_issue_unix.npy")[cycle]
            if not (np.all(issue <= valid) and np.all(valid <= origin[:, None])):
                raise ValueError(f"Response-cache causal-time audit failed: {split}")
        result = {
            "cache": str(self.destination),
            "preprocess_fingerprint": manifest["preprocess_fingerprint"],
            "arrays_checked": checked,
            "checksums_verified": bool(verify_checksums),
            "station_sample_tensors_present": False,
            "causal_time_protocol": True,
        }
        print(f"[response-cache-audit] {result}", flush=True)
        return result


class AggregateResponseForecastDataset(Dataset[int]):
    """Aggregate samples backed by seed-specific frozen R2 hourly latent caches."""

    def __init__(
        self,
        aggregate_cache: str | Path,
        response_cache: str | Path,
        latent_cache: str | Path,
        split: str,
        *,
        verbose: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.aggregate_cache = Path(aggregate_cache).resolve()
        self.response_cache = Path(response_cache).resolve()
        self.latent_cache = Path(latent_cache).resolve()
        with (self.aggregate_cache / "manifest.json").open(encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        with (self.response_cache / "manifest.json").open(encoding="utf-8") as stream:
            self.response_manifest = json.load(stream)
        with (self.latent_cache / split / "manifest.json").open(encoding="utf-8") as stream:
            self.latent_manifest = json.load(stream)
        if self.response_manifest.get("aggregate_fingerprint") != self.manifest.get("preprocess_fingerprint"):
            raise ValueError("Aggregate and response caches have different provenance")
        if self.latent_manifest.get("response_fingerprint") != self.response_manifest.get("preprocess_fingerprint"):
            raise ValueError("Latent and response caches have different provenance")
        if self.latent_manifest.get("split") != split:
            raise ValueError("Latent-cache split mismatch")
        self.total_capacity_mw = float(self.manifest["total_capacity_mw"])
        static = self.aggregate_cache / "static"
        self.static = {
            name: _load(static / f"{name}.npy")
            for name in (
                "weather_latitude", "weather_longitude", "nwp_mean", "nwp_std",
                "nwp_channel_name", "capacity_density",
            )
        }
        aggregate_dir = self.aggregate_cache / split
        response_dir = self.response_cache / split
        self.arrays = {
            name: _load(aggregate_dir / f"{name}.npy")
            for name in (
                "origin_unix", "sample_index", "origin_phase", "nwp_age_minutes",
                "weather_cycle_index", "weather_time_offset_hours", "cluster_history_mw",
                "cluster_future_mw", "cluster_future_capacity_warning",
            )
        }
        self.arrays.update(
            {
                name: _load(response_dir / f"{name}.npy")
                for name in (
                    "past_weather_cycle_index", "past_weather_time_offset_hours",
                    "past_cluster_power_mw",
                )
            }
        )
        if not np.array_equal(
            self.arrays["weather_cycle_index"], self.arrays["past_weather_cycle_index"]
        ):
            raise ValueError("Past/future cycle indices are not aligned")
        self.future_hourly_latent = _load(
            self.latent_cache / split / "future_hourly_latent.npy", mmap=True
        )
        self.past_hourly_latent = _load(
            self.latent_cache / split / "past_hourly_latent.npy", mmap=True
        )
        self.length = len(self.arrays["origin_unix"])
        if any(len(values) != self.length for values in self.arrays.values()):
            raise ValueError("Response-state sample arrays have inconsistent lengths")
        if verbose:
            print(
                f"[aggregate-response-dataset:{split}] samples={self.length} "
                f"future_latent={self.future_hourly_latent.shape} "
                f"past_latent={self.past_hourly_latent.shape}",
                flush=True,
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index

    def close(self) -> None:
        for values in (self.future_hourly_latent, self.past_hourly_latent):
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()


class AggregateResponseCollator:
    def __init__(self, dataset: AggregateResponseForecastDataset) -> None:
        self.dataset = dataset

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        index = np.asarray(indices, dtype=np.int64)
        cycles = np.asarray(self.dataset.arrays["weather_cycle_index"][index], dtype=np.int64)
        unique, inverse = np.unique(cycles, return_inverse=True)
        batch: dict[str, torch.Tensor] = {
            "future_hourly_weather_latent": torch.from_numpy(
                np.ascontiguousarray(np.asarray(self.dataset.future_hourly_latent[unique], dtype=np.float32))
            ),
            "weather_batch_index": torch.from_numpy(inverse.astype(np.int64)),
            "past_hourly_weather_latent": torch.from_numpy(
                np.ascontiguousarray(np.asarray(self.dataset.past_hourly_latent[cycles], dtype=np.float32))
            ),
        }
        for name, values in self.dataset.arrays.items():
            selected = np.ascontiguousarray(values[index])
            if name in {"cluster_history_mw", "cluster_future_mw", "past_cluster_power_mw"}:
                key = {
                    "cluster_history_mw": "cluster_history",
                    "cluster_future_mw": "cluster_future",
                    "past_cluster_power_mw": "past_cluster_power",
                }[name]
                batch[key] = torch.from_numpy(
                    selected.astype(np.float32) / self.dataset.total_capacity_mw
                )
            else:
                if selected.dtype == np.bool_:
                    tensor = torch.from_numpy(selected.astype(np.bool_))
                elif np.issubdtype(selected.dtype, np.floating):
                    tensor = torch.from_numpy(selected.astype(np.float32))
                else:
                    tensor = torch.from_numpy(selected.astype(np.int64))
                batch[name] = tensor
        return batch


def create_aggregate_response_loader(
    dataset: AggregateResponseForecastDataset,
    *,
    cycles_per_batch: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    sampler = CycleGroupedBatchSampler(
        dataset.arrays["weather_cycle_index"], cycles_per_batch, shuffle, seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=AggregateResponseCollator(dataset),
    )


def response_cache_manifest_identity(cache: str | Path) -> str:
    with (Path(cache) / "manifest.json").open(encoding="utf-8") as stream:
        return str(json.load(stream)["preprocess_fingerprint"])

