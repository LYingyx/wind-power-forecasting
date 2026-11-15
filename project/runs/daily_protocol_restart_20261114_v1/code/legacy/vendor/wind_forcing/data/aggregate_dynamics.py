from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wind_forcing.data.dataset import CycleGroupedBatchSampler
from wind_forcing.data.aggregate_response import AggregateResponseForecastDataset


SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_identity(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class AggregateDynamicsTimeCacheBuilder:
    """Cache causal metadata for estimating Pangu weather at forecast origin.

    The deployed future path starts at the first valid time after the origin for
    three of four 15-minute phases.  We therefore use the previous available
    Pangu issue and its two hourly nodes bracketing the origin.  The source HDF5
    stores the left node in the stage-I history window; the immediately next
    lead is resolved through ``run_lead_to_frame``.  At a split boundary where
    that frame is absent, the first node of the newer issue is an explicitly
    audited fallback with the same valid time.
    """

    def __init__(self, source_h5: str | Path, aggregate_cache: str | Path, destination: str | Path) -> None:
        self.source_h5 = Path(source_h5).resolve()
        self.aggregate_cache = Path(aggregate_cache).resolve()
        self.destination = Path(destination).resolve()

    def build(self) -> Path:
        with (self.aggregate_cache / "manifest.json").open(encoding="utf-8") as stream:
            aggregate = json.load(stream)
        source_stat = self.source_h5.stat()
        identity_payload = {
            "source_h5": str(self.source_h5),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "aggregate_fingerprint": aggregate["preprocess_fingerprint"],
            "protocol": "previous_available_issue_brackets_origin",
        }
        fingerprint = _json_identity(identity_payload)
        manifest_path = self.destination / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as stream:
                current = json.load(stream)
            if current.get("preprocess_fingerprint") != fingerprint:
                raise RuntimeError(f"Stale dynamics time cache: {self.destination}")
            self.audit()
            return self.destination

        self.destination.mkdir(parents=True, exist_ok=False)
        arrays: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}
        with h5py.File(self.source_h5, "r") as handle:
            run_to_frame = handle["nwp/run_lead_to_frame"]
            run_valid = handle["nwp/valid"]
            frame_run = np.asarray(handle["nwp/frame_run_index"], dtype=np.int64)
            issue = np.asarray(handle["nwp/issue_unix"], dtype=np.int64)
            for split in SPLITS:
                output = self.destination / split
                output.mkdir()
                cycles = handle[f"stage1_splits/{split}"]
                samples = handle[f"splits/{split}"]
                aggregate_dir = self.aggregate_cache / split
                aggregate_cycle_ids = np.load(aggregate_dir / "cycle_sample_index.npy", allow_pickle=False)
                if not np.array_equal(aggregate_cycle_ids, np.asarray(cycles["sample_index"], dtype=np.int64)):
                    raise ValueError(f"Cycle order mismatch for {split}")
                aggregate_samples = np.load(aggregate_dir / "sample_index.npy", allow_pickle=False)
                if not np.array_equal(aggregate_samples, np.asarray(samples["sample_index"], dtype=np.int64)):
                    raise ValueError(f"Sample order mismatch for {split}")

                old_runs = np.asarray(cycles["nwp_run_index"][:, 5], dtype=np.int64)
                left_frames = np.asarray(cycles["nwp_frame_index"][:, 5], dtype=np.int64)
                left_leads = np.asarray(cycles["nwp_lead_index"][:, 5], dtype=np.int64)
                right_leads = left_leads + 1
                right_frames = np.full(len(old_runs), -1, dtype=np.int64)
                fallback = np.zeros(len(old_runs), dtype=np.bool_)
                for index, (run, lead) in enumerate(zip(old_runs, right_leads)):
                    if lead < run_to_frame.shape[1] and bool(run_valid[run, lead]):
                        right_frames[index] = int(run_to_frame[run, lead])
                    else:
                        right_frames[index] = int(cycles["nwp_frame_index"][index, 6])
                        fallback[index] = True
                if np.any(right_frames < 0):
                    raise ValueError(f"Unresolved current-weather right frame for {split}")
                if not np.array_equal(frame_run[left_frames], old_runs):
                    raise ValueError(f"Left frame issue mismatch for {split}")

                old_issue = np.asarray(issue[old_runs], dtype=np.int64)
                left_valid = old_issue + (left_leads + 1) * 3600
                right_valid = left_valid + 3600
                cycle_ids = np.asarray(samples["stage1_cycle_index"], dtype=np.int64)
                local = np.searchsorted(aggregate_cycle_ids, cycle_ids)
                origins = np.asarray(samples["origin_unix"], dtype=np.int64)
                offsets = np.stack(
                    (left_valid[local] - origins, right_valid[local] - origins), axis=-1
                ).astype(np.float32) / 3600.0
                if not (np.all(offsets[:, 0] <= 0.0) and np.all(offsets[:, 1] >= 0.0)):
                    raise ValueError(f"Current weather does not bracket origin for {split}")
                issue_pair = np.stack((old_issue, issue[frame_run[right_frames]]), axis=-1)
                if not np.all(issue_pair[local] <= origins[:, None]):
                    raise ValueError(f"Current-weather frame was not available at origin for {split}")

                values = {
                    "left_frame_index": left_frames,
                    "right_frame_index": right_frames,
                    "right_cycle_fallback": fallback,
                    "current_weather_time_offset_hours": offsets,
                    "current_weather_issue_unix": issue_pair[local].astype(np.int64),
                    "sample_index": aggregate_samples.astype(np.int64),
                }
                for name, value in values.items():
                    path = output / f"{name}.npy"
                    np.save(path, value, allow_pickle=False)
                    arrays.append({
                        "relative_path": f"{split}/{name}.npy",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": _sha256(path),
                    })
                summaries[split] = {
                    "sample_count": int(len(origins)),
                    "cycle_count": int(len(old_runs)),
                    "fallback_cycle_count": int(fallback.sum()),
                    "left_offset_min_hours": float(offsets[:, 0].min()),
                    "right_offset_max_hours": float(offsets[:, 1].max()),
                }
        payload = {
            "schema_version": 1,
            "task_level": "aggregate_weather_dynamics_time",
            **identity_payload,
            "preprocess_fingerprint": fingerprint,
            "station_sample_tensors_present": False,
            "time_protocol": "both interpolation nodes issued no later than origin",
            "splits": summaries,
            "arrays": arrays,
        }
        _atomic_json(manifest_path, payload)
        self.audit()
        return self.destination

    def audit(self, verify_checksums: bool = False) -> dict[str, Any]:
        with (self.destination / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("task_level") != "aggregate_weather_dynamics_time":
            raise ValueError("Wrong dynamics-time cache")
        checked = 0
        for entry in manifest["arrays"]:
            path = self.destination / entry["relative_path"]
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(values.shape) != entry["shape"] or str(values.dtype) != entry["dtype"]:
                raise ValueError(f"Dynamics-time metadata mismatch: {entry['relative_path']}")
            if verify_checksums and _sha256(path) != entry["sha256"]:
                raise ValueError(f"Dynamics-time checksum mismatch: {entry['relative_path']}")
            checked += 1
        return {"cache": str(self.destination), "arrays_checked": checked, "causal": True}


class AggregateWeatherDynamicsDataset(Dataset[int]):
    """Response-state dataset plus a seed-specific latent at forecast origin."""

    def __init__(
        self,
        aggregate_cache: str | Path,
        response_cache: str | Path,
        response_latent_cache: str | Path,
        dynamics_time_cache: str | Path,
        dynamics_latent_cache: str | Path,
        split: str,
        *,
        realization_cache: str | Path | None = None,
        exclude_origin_start_unix: int | None = None,
        exclude_origin_stop_unix: int | None = None,
        exclusion_embargo_seconds: int = 0,
        verbose: bool = True,
    ) -> None:
        self.base = AggregateResponseForecastDataset(
            aggregate_cache, response_cache, response_latent_cache, split, verbose=False
        )
        self.manifest = self.base.manifest
        self.response_manifest = self.base.response_manifest
        self.latent_manifest = self.base.latent_manifest
        self.total_capacity_mw = self.base.total_capacity_mw
        self.static = self.base.static
        self.arrays = self.base.arrays
        time_root = Path(dynamics_time_cache).resolve()
        with (time_root / "manifest.json").open(encoding="utf-8") as stream:
            self.dynamics_time_manifest = json.load(stream)
        if self.dynamics_time_manifest.get("aggregate_fingerprint") != self.manifest.get("preprocess_fingerprint"):
            raise ValueError("Dynamics-time and aggregate cache provenance differ")
        current_root = Path(dynamics_latent_cache).resolve() / split
        with (current_root / "manifest.json").open(encoding="utf-8") as stream:
            self.dynamics_latent_manifest = json.load(stream)
        if self.dynamics_latent_manifest.get("base_checkpoint_sha256") != self.latent_manifest.get("base_checkpoint_sha256"):
            raise ValueError("Current and future R2 latent caches use different checkpoints")
        self.current_weather_latent = np.load(
            current_root / "current_weather_latent.npy", mmap_mode="r", allow_pickle=False
        )
        offsets = np.load(
            time_root / split / "current_weather_time_offset_hours.npy", mmap_mode="r", allow_pickle=False
        )
        self.arrays = dict(self.arrays)
        self.arrays["current_weather_time_offset_hours"] = offsets
        self.realization_factor = None
        self.realization_manifest = None
        if realization_cache is not None:
            realization_root = Path(realization_cache).resolve() / split
            with (realization_root / "manifest.json").open(encoding="utf-8") as stream:
                self.realization_manifest = json.load(stream)
            self.realization_factor = np.load(
                realization_root / "past_realization_factor.npy", mmap_mode="r", allow_pickle=False
            )
        if len(self.current_weather_latent) != len(self.base):
            raise ValueError("Current-weather latent sample count mismatch")
        if self.realization_factor is not None and len(self.realization_factor) != len(self.base):
            raise ValueError("Realization-factor sample count mismatch")
        self.sample_indices = np.arange(len(self.base), dtype=np.int64)
        if exclude_origin_start_unix is not None or exclude_origin_stop_unix is not None:
            if exclude_origin_start_unix is None or exclude_origin_stop_unix is None:
                raise ValueError("Both exclusion range endpoints are required")
            start = int(exclude_origin_start_unix) - int(exclusion_embargo_seconds)
            stop = int(exclude_origin_stop_unix) + int(exclusion_embargo_seconds)
            origins = np.asarray(self.arrays["origin_unix"], dtype=np.int64)
            self.sample_indices = np.flatnonzero((origins < start) | (origins > stop)).astype(np.int64)
            if not len(self.sample_indices):
                raise ValueError("Dynamics exclusion removed every sample")
        if verbose:
            print(
                f"[aggregate-dynamics:{split}] samples={len(self.sample_indices)}/{len(self.base)} "
                f"current={self.current_weather_latent.shape} realization={self.realization_factor is not None}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> int:
        return index

    def close(self) -> None:
        self.base.close()
        for values in (self.current_weather_latent, self.realization_factor):
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()


class AggregateWeatherDynamicsCollator:
    def __init__(self, dataset: AggregateWeatherDynamicsDataset) -> None:
        self.dataset = dataset

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        positions = np.asarray(indices, dtype=np.int64)
        index = self.dataset.sample_indices[positions]
        cycles = np.asarray(self.dataset.arrays["weather_cycle_index"][index], dtype=np.int64)
        unique, inverse = np.unique(cycles, return_inverse=True)
        batch: dict[str, torch.Tensor] = {
            "future_hourly_weather_latent": torch.from_numpy(np.ascontiguousarray(
                np.asarray(self.dataset.base.future_hourly_latent[unique], dtype=np.float32)
            )),
            "past_hourly_weather_latent": torch.from_numpy(np.ascontiguousarray(
                np.asarray(self.dataset.base.past_hourly_latent[cycles], dtype=np.float32)
            )),
            "current_weather_latent": torch.from_numpy(np.ascontiguousarray(
                np.asarray(self.dataset.current_weather_latent[index], dtype=np.float32)
            )),
            "weather_batch_index": torch.from_numpy(inverse.astype(np.int64)),
        }
        if self.dataset.realization_factor is not None:
            batch["past_realization_factor"] = torch.from_numpy(np.ascontiguousarray(
                np.asarray(self.dataset.realization_factor[index], dtype=np.float32)
            ))
        for name, values in self.dataset.arrays.items():
            selected = np.ascontiguousarray(values[index])
            if name in {"cluster_history_mw", "cluster_future_mw", "past_cluster_power_mw"}:
                key = {
                    "cluster_history_mw": "cluster_history",
                    "cluster_future_mw": "cluster_future",
                    "past_cluster_power_mw": "past_cluster_power",
                }[name]
                batch[key] = torch.from_numpy(selected.astype(np.float32) / self.dataset.total_capacity_mw)
            elif selected.dtype == np.bool_:
                batch[name] = torch.from_numpy(selected.astype(np.bool_))
            elif np.issubdtype(selected.dtype, np.floating):
                batch[name] = torch.from_numpy(selected.astype(np.float32))
            else:
                batch[name] = torch.from_numpy(selected.astype(np.int64))
        return batch


def create_aggregate_dynamics_loader(
    dataset: AggregateWeatherDynamicsDataset,
    *,
    cycles_per_batch: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    sampler = CycleGroupedBatchSampler(
        np.asarray(dataset.arrays["weather_cycle_index"])[dataset.sample_indices],
        cycles_per_batch, shuffle, seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=AggregateWeatherDynamicsCollator(dataset),
    )
