from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wind_forcing.data.dataset import CycleGroupedBatchSampler


SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, mmap: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def _save(path: Path, values: np.ndarray) -> dict[str, Any]:
    np.save(path, values, allow_pickle=False)
    return {
        "relative_path": "",
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": _sha256(path),
        "storage": "owned",
    }


def capacity_density_bilinear(
    weather_latitude: np.ndarray,
    weather_longitude: np.ndarray,
    station_latitude: np.ndarray,
    station_longitude: np.ndarray,
    station_capacity_mw: np.ndarray,
) -> np.ndarray:
    """Conservatively scatter station capacities onto four enclosing grid nodes."""
    latitude = np.asarray(weather_latitude, dtype=np.float64)
    longitude = np.asarray(weather_longitude, dtype=np.float64)
    station_lat = np.asarray(station_latitude, dtype=np.float64)
    station_lon = np.asarray(station_longitude, dtype=np.float64)
    capacity = np.asarray(station_capacity_mw, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("Weather coordinates must be one-dimensional")
    if not (np.all(np.diff(latitude) < 0) and np.all(np.diff(longitude) > 0)):
        raise ValueError("Expected descending latitude and ascending longitude")
    if not (station_lat.shape == station_lon.shape == capacity.shape):
        raise ValueError("Station coordinates and capacities have inconsistent shapes")
    if np.any(capacity <= 0) or not np.isfinite(capacity).all():
        raise ValueError("Station capacities must be finite and positive")

    density = np.zeros((len(latitude), len(longitude)), dtype=np.float64)
    for farm_lat, farm_lon, farm_capacity in zip(station_lat, station_lon, capacity):
        north = int(np.searchsorted(-latitude, -farm_lat, side="right") - 1)
        west = int(np.searchsorted(longitude, farm_lon, side="right") - 1)
        if not (0 <= north < len(latitude) - 1 and 0 <= west < len(longitude) - 1):
            raise ValueError("A station lies outside the weather grid")
        west_fraction = float(
            (farm_lon - longitude[west]) / (longitude[west + 1] - longitude[west])
        )
        north_fraction = float(
            (farm_lat - latitude[north + 1]) / (latitude[north] - latitude[north + 1])
        )
        weights = (
            (1.0 - west_fraction) * north_fraction,
            west_fraction * north_fraction,
            (1.0 - west_fraction) * (1.0 - north_fraction),
            west_fraction * (1.0 - north_fraction),
        )
        nodes = ((north, west), (north, west + 1), (north + 1, west), (north + 1, west + 1))
        for (row, column), weight in zip(nodes, weights):
            density[row, column] += farm_capacity * weight
    if not np.isclose(density.sum(), capacity.sum(), atol=1e-5, rtol=1e-10):
        raise ValueError("Capacity scatter failed conservation audit")
    density /= density.sum()
    return density.astype(np.float32)


class AggregateCacheBuilder:
    """Derive a station-free aggregate cache from an audited schema-v1 cache."""

    sample_names = (
        "origin_unix",
        "sample_index",
        "origin_phase",
        "nwp_age_minutes",
        "weather_cycle_index",
        "weather_time_offset_hours",
    )
    cycle_names = ("weather_valid_unix", "cycle_sample_index")

    def __init__(self, source: str | Path, destination: str | Path) -> None:
        self.source = Path(source).resolve()
        self.destination = Path(destination).resolve()

    def _source_manifest(self) -> dict[str, Any]:
        path = self.source / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if int(manifest.get("schema_version", -1)) != 1:
            raise ValueError("Aggregate derivation requires a schema-v1 model cache")
        return manifest

    def is_current(self) -> bool:
        path = self.destination / "manifest.json"
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            source = self._source_manifest()
            if manifest.get("schema_version") != 2:
                return False
            if manifest.get("parent_preprocess_fingerprint") != source.get("preprocess_fingerprint"):
                return False
            for entry in manifest["arrays"]:
                array = _load(self.destination / entry["relative_path"], mmap=True)
                if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                    return False
            return True
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def audit(self, *, verify_checksums: bool = False) -> dict[str, Any]:
        """Validate the station-free schema and optionally every stored digest."""
        path = self.destination / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != 2 or manifest.get("task_level") != "aggregate_cluster":
            raise ValueError("Aggregate cache manifest has the wrong schema")
        if manifest.get("station_sample_tensors_present") is not False:
            raise ValueError("Aggregate manifest does not prohibit station sample tensors")
        checked = 0
        for entry in manifest["arrays"]:
            array_path = self.destination / entry["relative_path"]
            array = _load(array_path, mmap=True)
            if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                raise ValueError(f"Array metadata mismatch: {entry['relative_path']}")
            if "station" in Path(entry["relative_path"]).name and not entry["relative_path"].startswith("static/"):
                raise ValueError(f"Station sample tensor leaked into aggregate cache: {entry['relative_path']}")
            if verify_checksums and _sha256(array_path) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch: {entry['relative_path']}")
            checked += 1
        for split in SPLITS:
            directory_names = {item.name for item in (self.destination / split).glob("*.npy")}
            if any("station" in name for name in directory_names):
                raise ValueError(f"Station array found in aggregate {split} directory")
        density = _load(self.destination / "static" / "capacity_density.npy")
        if density.shape != (24, 25) or np.any(density < 0) or not np.isclose(density.sum(), 1.0, atol=1e-6):
            raise ValueError("Capacity-density map failed normalization audit")
        result = {
            "cache": str(self.destination),
            "preprocess_fingerprint": manifest["preprocess_fingerprint"],
            "arrays_checked": checked,
            "checksums_verified": bool(verify_checksums),
            "station_sample_tensors_present": False,
            "capacity_density_sum": float(density.sum()),
        }
        print(f"[aggregate-audit-cache] {result}", flush=True)
        return result

    def build(self) -> Path:
        source_manifest = self._source_manifest()
        if self.is_current():
            print(f"[aggregate-prepare] reusing {self.destination}", flush=True)
            return self.destination
        if self.destination.exists():
            raise FileExistsError(
                f"Stale aggregate cache exists: {self.destination}. Move it aside before rebuilding."
            )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.destination.name}.tmp.", dir=self.destination.parent)
        )
        try:
            manifest = self._build_into(temporary, source_manifest)
            with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=True, indent=2)
            os.replace(temporary, self.destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(f"[aggregate-prepare] ready {self.destination}", flush=True)
        return self.destination

    def _build_into(self, root: Path, source_manifest: Mapping[str, Any]) -> dict[str, Any]:
        arrays: list[dict[str, Any]] = []
        static_source = self.source / "static"
        static_output = root / "static"
        static_output.mkdir()
        static_names = (
            "weather_latitude",
            "weather_longitude",
            "nwp_mean",
            "nwp_std",
            "nwp_channel_name",
        )
        for name in static_names:
            values = np.asarray(_load(static_source / f"{name}.npy"))
            path = static_output / f"{name}.npy"
            entry = _save(path, values)
            entry["relative_path"] = f"static/{name}.npy"
            arrays.append(entry)
        capacity_density = capacity_density_bilinear(
            _load(static_source / "weather_latitude.npy"),
            _load(static_source / "weather_longitude.npy"),
            _load(static_source / "station_latitude.npy"),
            _load(static_source / "station_longitude.npy"),
            _load(static_source / "station_capacity_mw.npy"),
        )
        density_path = static_output / "capacity_density.npy"
        entry = _save(density_path, capacity_density)
        entry["relative_path"] = "static/capacity_density.npy"
        arrays.append(entry)

        total_capacity = float(source_manifest["total_capacity_mw"])
        split_manifest: dict[str, Any] = {}
        for split in SPLITS:
            source_dir = self.source / split
            output_dir = root / split
            output_dir.mkdir()
            factor_history = np.asarray(_load(source_dir / "cluster_history.npy"), dtype=np.float32)
            factor_future = np.asarray(_load(source_dir / "cluster_future.npy"), dtype=np.float32)
            values: dict[str, np.ndarray] = {
                name: np.asarray(_load(source_dir / f"{name}.npy"))
                for name in (*self.sample_names, *self.cycle_names)
            }
            values["cluster_history_mw"] = factor_history * total_capacity
            values["cluster_future_mw"] = factor_future * total_capacity
            warning = np.asarray(_load(source_dir / "future_capacity_warning.npy"), dtype=bool)
            values["cluster_future_capacity_warning"] = warning.any(axis=1)
            for name, value in values.items():
                path = output_dir / f"{name}.npy"
                item = _save(path, value)
                item["relative_path"] = f"{split}/{name}.npy"
                arrays.append(item)

            source_weather = source_dir / "weather.npy"
            output_weather = output_dir / "weather.npy"
            storage = "hardlink"
            try:
                os.link(source_weather, output_weather)
            except OSError:
                shutil.copy2(source_weather, output_weather)
                storage = "copy"
            weather = _load(output_weather, mmap=True)
            arrays.append(
                {
                    "relative_path": f"{split}/weather.npy",
                    "shape": list(weather.shape),
                    "dtype": str(weather.dtype),
                    "sha256": _sha256(output_weather),
                    "storage": storage,
                }
            )
            origins = values["origin_unix"].astype("datetime64[s]")
            split_manifest[split] = {
                "sample_count": int(len(factor_history)),
                "cycle_count": int(len(weather)),
                "origin_start": str(origins.min()),
                "origin_stop": str(origins.max()),
                "weather_shape": list(weather.shape[1:]),
            }
        identity_payload = {
            "parent": source_manifest["preprocess_fingerprint"],
            "task": "aggregate_cluster_v1",
            "total_capacity_mw": total_capacity,
            "history_steps": int(source_manifest["history_power_steps"]),
            "forecast_steps": int(source_manifest["forecast_power_steps"]),
        }
        identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": 2,
            "task_level": "aggregate_cluster",
            "source_cache": str(self.source),
            "parent_preprocess_fingerprint": source_manifest["preprocess_fingerprint"],
            "preprocess_fingerprint": identity,
            "station_sample_tensors_present": False,
            "weather_values_are_prestandardized": True,
            "power_storage_unit": "MW",
            "model_power_unit": "capacity_factor",
            "capacity_density_method": "four_node_bilinear_scatter",
            "total_capacity_mw": total_capacity,
            "history_power_steps": int(source_manifest["history_power_steps"]),
            "forecast_power_steps": int(source_manifest["forecast_power_steps"]),
            "future_nwp_nodes": int(source_manifest["future_nwp_nodes"]),
            "target_lead_hours": list(source_manifest["target_lead_hours"]),
            "crop": source_manifest["crop"],
            "splits": split_manifest,
            "arrays": arrays,
        }


class AggregateForecastDataset(Dataset[int]):
    sample_names = (
        "origin_unix",
        "sample_index",
        "origin_phase",
        "nwp_age_minutes",
        "weather_cycle_index",
        "weather_time_offset_hours",
        "cluster_history_mw",
        "cluster_future_mw",
        "cluster_future_capacity_warning",
    )

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        *,
        include_weather: bool = True,
        preload_weather: bool | str = "auto",
        preload_weather_max_gib: float = 16.0,
        verbose: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.cache_dir = Path(cache_dir).resolve()
        with (self.cache_dir / "manifest.json").open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest.get("schema_version") != 2 or self.manifest.get("task_level") != "aggregate_cluster":
            raise ValueError("Expected aggregate-cluster schema-v2 cache")
        self.total_capacity_mw = float(self.manifest["total_capacity_mw"])
        static_dir = self.cache_dir / "static"
        self.static = {
            name: _load(static_dir / f"{name}.npy")
            for name in (
                "weather_latitude",
                "weather_longitude",
                "nwp_mean",
                "nwp_std",
                "nwp_channel_name",
                "capacity_density",
            )
        }
        split_dir = self.cache_dir / split
        self.arrays = {name: _load(split_dir / f"{name}.npy") for name in self.sample_names}
        self.include_weather = bool(include_weather)
        self.weather: np.ndarray | None = None
        if self.include_weather:
            mapped = _load(split_dir / "weather.npy", mmap=True)
            setting = str(preload_weather).lower() if isinstance(preload_weather, str) else preload_weather
            preload = setting is True or setting in {"1", "true", "yes"}
            if setting == "auto":
                preload = mapped.nbytes <= int(preload_weather_max_gib * 1024**3)
            self.weather = np.array(mapped, copy=True) if preload else mapped
        self.length = len(self.arrays["origin_unix"])
        if any(len(value) != self.length for value in self.arrays.values()):
            raise ValueError("Aggregate sample arrays have inconsistent lengths")
        if verbose:
            print(
                f"[aggregate-dataset:{split}] samples={self.length} "
                f"weather={self.include_weather} history=[24] target=[24]",
                flush=True,
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index

    def close(self) -> None:
        mmap = getattr(self.weather, "_mmap", None)
        if mmap is not None:
            mmap.close()


class AggregateForecastCollator:
    def __init__(self, dataset: AggregateForecastDataset) -> None:
        self.dataset = dataset

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        index = np.asarray(indices, dtype=np.int64)
        batch: dict[str, torch.Tensor] = {}
        if self.dataset.include_weather:
            if self.dataset.weather is None:
                raise RuntimeError("Weather store is unavailable")
            cycles = np.asarray(self.dataset.arrays["weather_cycle_index"][index], dtype=np.int64)
            unique, inverse = np.unique(cycles, return_inverse=True)
            batch["weather"] = torch.from_numpy(
                np.ascontiguousarray(np.asarray(self.dataset.weather[unique], dtype=np.float32))
            )
            batch["weather_batch_index"] = torch.from_numpy(inverse.astype(np.int64))
        for name, values in self.dataset.arrays.items():
            selected = np.ascontiguousarray(values[index])
            if name == "cluster_history_mw":
                batch["cluster_history"] = torch.from_numpy(
                    selected.astype(np.float32) / self.dataset.total_capacity_mw
                )
            elif name == "cluster_future_mw":
                batch["cluster_future"] = torch.from_numpy(
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


def create_aggregate_loader(
    dataset: AggregateForecastDataset,
    *,
    cycles_per_batch: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader[dict[str, torch.Tensor]]:
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": AggregateForecastCollator(dataset),
    }
    if dataset.include_weather:
        sampler = CycleGroupedBatchSampler(
            dataset.arrays["weather_cycle_index"], cycles_per_batch, shuffle, seed
        )
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=4 * cycles_per_batch,
        shuffle=shuffle,
        generator=generator,
        **kwargs,
    )
