from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import h5py
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CropSpec:
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int
    north: float
    south: float
    west: float
    east: float
    resolution: float
    requested_padding_grid_points: int

    @property
    def height(self) -> int:
        return self.row_stop - self.row_start

    @property
    def width(self) -> int:
        return self.column_stop - self.column_start


def _decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def _exact_axis_index(axis: np.ndarray, value: float) -> int:
    matches = np.flatnonzero(np.isclose(axis, value, atol=1e-8, rtol=0.0))
    if len(matches) != 1:
        raise ValueError(f"Requested crop boundary {value:.8f} is absent from source axis")
    return int(matches[0])


def compute_station_envelope_crop(
    latitude: Sequence[float],
    longitude: Sequence[float],
    station_latitude: Sequence[float],
    station_longitude: Sequence[float],
    padding_grid_points: int,
) -> CropSpec:
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    station_lat = np.asarray(station_latitude, dtype=np.float64)
    station_lon = np.asarray(station_longitude, dtype=np.float64)
    if padding_grid_points < 0:
        raise ValueError("padding_grid_points must be non-negative")
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2:
        raise ValueError("Weather coordinates must be non-trivial one-dimensional axes")
    lat_step = np.diff(lat)
    lon_step = np.diff(lon)
    resolution = float(abs(lat_step[0]))
    if not (
        np.allclose(lat_step, -resolution)
        and np.allclose(lon_step, resolution)
        and np.isclose(abs(lon_step[0]), resolution)
    ):
        raise ValueError("Expected descending latitude and ascending longitude grids")
    if not (
        np.isfinite(station_lat).all()
        and np.isfinite(station_lon).all()
        and station_lat.size
        and station_lat.shape == station_lon.shape
    ):
        raise ValueError("Station coordinates are invalid")

    north = (
        math.ceil(float(station_lat.max()) / resolution) * resolution
        + padding_grid_points * resolution
    )
    south = (
        math.floor(float(station_lat.min()) / resolution) * resolution
        - padding_grid_points * resolution
    )
    west = (
        math.floor(float(station_lon.min()) / resolution) * resolution
        - padding_grid_points * resolution
    )
    east = (
        math.ceil(float(station_lon.max()) / resolution) * resolution
        + padding_grid_points * resolution
    )
    row_start = _exact_axis_index(lat, north)
    row_stop = _exact_axis_index(lat, south) + 1
    column_start = _exact_axis_index(lon, west)
    column_stop = _exact_axis_index(lon, east) + 1
    return CropSpec(
        row_start=row_start,
        row_stop=row_stop,
        column_start=column_start,
        column_stop=column_stop,
        north=north,
        south=south,
        west=west,
        east=east,
        resolution=resolution,
        requested_padding_grid_points=padding_grid_points,
    )


def _save_array(directory: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    path = directory / f"{name}.npy"
    np.save(path, values, allow_pickle=False)
    return {"file": path.name, "shape": list(values.shape), "dtype": str(values.dtype)}


class ModelDataPreparer:
    """Build model-ready arrays without spatial or temporal interpolation."""

    def __init__(self, data_config: dict[str, Any]) -> None:
        self.config = dict(data_config)
        self.source = Path(str(self.config["source_h5"])).resolve()
        self.destination = Path(str(self.config["cache_dir"])).resolve()
        self.padding = int(self.config.get("region_padding_grid_points", 5))
        self.history_power_steps = int(self.config.get("history_power_steps", 24))
        self.forecast_power_steps = int(self.config.get("forecast_power_steps", 24))
        self.history_nwp_nodes = int(self.config.get("history_nwp_nodes", 6))
        self.future_nwp_nodes = int(self.config.get("future_nwp_nodes", 8))
        self.chunk_cycles = int(self.config.get("cache_chunk_cycles", 32))
        self.weather_dtype = np.dtype(str(self.config.get("weather_storage_dtype", "float32")))
        if self.weather_dtype not in (np.dtype("float32"), np.dtype("float16")):
            raise ValueError("weather_storage_dtype must be float32 or float16")
        if min(
            self.history_power_steps,
            self.forecast_power_steps,
            self.future_nwp_nodes,
            self.chunk_cycles,
        ) < 1:
            raise ValueError("Window lengths and cache_chunk_cycles must be positive")

    def _identity(self) -> dict[str, Any]:
        stat = self.source.stat()
        relevant = {
            "region_padding_grid_points": self.padding,
            "history_power_steps": self.history_power_steps,
            "forecast_power_steps": self.forecast_power_steps,
            "history_nwp_nodes": self.history_nwp_nodes,
            "future_nwp_nodes": self.future_nwp_nodes,
            "weather_storage_dtype": str(self.weather_dtype),
        }
        fingerprint = hashlib.sha256(
            json.dumps(relevant, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "source_h5": str(self.source),
            "source_size_bytes": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "preprocess_fingerprint": fingerprint,
        }

    def is_current(self) -> bool:
        manifest_path = self.destination / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            if any(manifest.get(key) != value for key, value in self._identity().items()):
                return False
            for entry in manifest["arrays"]:
                path = self.destination / entry["relative_path"]
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                    return False
            return True
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def build(self) -> Path:
        if not self.source.is_file():
            raise FileNotFoundError(f"Source HDF5 does not exist: {self.source}")
        if self.is_current():
            print(f"[prepare] Reusing current model cache: {self.destination}")
            return self.destination
        if self.destination.exists():
            raise FileExistsError(
                f"A stale or incomplete cache already exists: {self.destination}. "
                "Move it aside before rebuilding."
            )

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{self.destination.name}.tmp.", dir=self.destination.parent
            )
        )
        try:
            manifest = self._build_into(temporary)
            with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=True, indent=2)
            os.replace(temporary, self.destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(f"[prepare] Model cache ready: {self.destination}")
        return self.destination

    def _build_into(self, output_root: Path) -> dict[str, Any]:
        arrays: list[dict[str, Any]] = []
        with h5py.File(self.source, "r") as handle:
            if not bool(handle["nwp"].attrs.get("values_are_prestandardized", False)):
                raise ValueError("Expected pre-standardized NWP values in source HDF5")
            station_lat = np.asarray(handle["stations/latitude"], dtype=np.float64)
            station_lon = np.asarray(handle["stations/longitude"], dtype=np.float64)
            capacity = np.asarray(handle["stations/capacity_mw"], dtype=np.float32)
            total_capacity = float(handle["stations"].attrs["total_capacity_mw"])
            source_lat = np.asarray(handle["crop/latitude"], dtype=np.float64)
            source_lon = np.asarray(handle["crop/longitude"], dtype=np.float64)
            crop = compute_station_envelope_crop(
                source_lat, source_lon, station_lat, station_lon, self.padding
            )
            if len(capacity) != 56 or not np.isclose(capacity.sum(), total_capacity, atol=1e-3):
                raise ValueError("Station capacity metadata is inconsistent")

            static_dir = output_root / "static"
            static_dir.mkdir()
            static_values = {
                "station_id": _decode_strings(np.asarray(handle["stations/id"])),
                "station_name": _decode_strings(np.asarray(handle["stations/name"])),
                "station_latitude": station_lat.astype(np.float32),
                "station_longitude": station_lon.astype(np.float32),
                "station_capacity_mw": capacity,
                "weather_latitude": source_lat[crop.row_start : crop.row_stop],
                "weather_longitude": source_lon[crop.column_start : crop.column_stop],
                "nwp_mean": np.asarray(handle["normalization/mean"], dtype=np.float32),
                "nwp_std": np.asarray(handle["normalization/std"], dtype=np.float32),
                "nwp_channel_name": _decode_strings(
                    np.asarray(handle["normalization/channel_short_name"])
                ),
            }
            for name, values in static_values.items():
                entry = _save_array(static_dir, name, values)
                arrays.append({"relative_path": f"static/{entry['file']}", **entry})

            power_time = np.asarray(handle["power/timestamp_unix"], dtype=np.int64)
            station_power = np.asarray(handle["power/station_mw"], dtype=np.float32)
            capacity_warning = np.asarray(handle["power/capacity_warning"], dtype=bool)
            split_summaries: dict[str, Any] = {}
            for split in SPLITS:
                summary, split_entries = self._build_split(
                    handle,
                    split,
                    output_root / split,
                    crop,
                    power_time,
                    station_power,
                    capacity_warning,
                    capacity,
                    total_capacity,
                )
                split_summaries[split] = summary
                arrays.extend(split_entries)

        return {
            "schema_version": 1,
            **self._identity(),
            "no_spatial_interpolation": True,
            "no_temporal_interpolation": True,
            "weather_values_are_prestandardized": True,
            "crop": {**asdict(crop), "shape": [crop.height, crop.width]},
            "station_count": int(len(capacity)),
            "total_capacity_mw": total_capacity,
            "history_power_steps": self.history_power_steps,
            "forecast_power_steps": self.forecast_power_steps,
            "future_nwp_nodes": self.future_nwp_nodes,
            "target_lead_hours": (
                np.arange(1, self.forecast_power_steps + 1, dtype=np.float64) / 4.0
            ).tolist(),
            "splits": split_summaries,
            "arrays": arrays,
        }

    def _build_split(
        self,
        handle: h5py.File,
        split: str,
        output_dir: Path,
        crop: CropSpec,
        power_time: np.ndarray,
        station_power: np.ndarray,
        capacity_warning: np.ndarray,
        capacity: np.ndarray,
        total_capacity: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        output_dir.mkdir()
        entries: list[dict[str, Any]] = []
        cycle_group = handle[f"stage1_splits/{split}"]
        sample_group = handle[f"splits/{split}"]
        cycle_ids = np.asarray(cycle_group["sample_index"], dtype=np.int64)
        sample_cycle_ids = np.asarray(sample_group["stage1_cycle_index"], dtype=np.int64)
        local_cycle_index = np.searchsorted(cycle_ids, sample_cycle_ids)
        if (
            np.any(local_cycle_index >= len(cycle_ids))
            or not np.array_equal(cycle_ids[local_cycle_index], sample_cycle_ids)
        ):
            raise ValueError(f"Sample-to-cycle mapping is inconsistent for {split}")

        node_start = self.history_nwp_nodes
        node_stop = node_start + self.future_nwp_nodes
        if node_stop > cycle_group["nwp_frame_index"].shape[1]:
            raise ValueError("Requested future NWP nodes exceed the source cycle window")
        frame_index = cycle_group["nwp_frame_index"][:, node_start:node_stop]
        valid_unix = np.asarray(
            cycle_group["nwp_valid_unix"][:, node_start:node_stop], dtype=np.int64
        )
        nwp_store = handle["nwp/values"]
        weather_shape = (
            len(cycle_ids),
            self.future_nwp_nodes,
            int(nwp_store.shape[1]),
            crop.height,
            crop.width,
        )
        weather_path = output_dir / "weather.npy"
        weather = open_memmap(
            weather_path, mode="w+", dtype=self.weather_dtype, shape=weather_shape
        )
        with tqdm(total=len(cycle_ids), desc=f"Preparing weather ({split})", unit="cycle") as bar:
            for start in range(0, len(cycle_ids), self.chunk_cycles):
                stop = min(start + self.chunk_cycles, len(cycle_ids))
                indices = np.asarray(frame_index[start:stop], dtype=np.int64)
                unique, inverse = np.unique(indices, return_inverse=True)
                values = np.asarray(nwp_store[unique], dtype=np.float32)
                values = values[
                    :,
                    :,
                    crop.row_start : crop.row_stop,
                    crop.column_start : crop.column_stop,
                ]
                packed = values[inverse.reshape(indices.shape)]
                if not np.isfinite(packed).all():
                    raise ValueError(f"Non-finite weather values encountered in {split}")
                weather[start:stop] = packed.astype(self.weather_dtype, copy=False)
                bar.update(stop - start)
        weather.flush()
        del weather
        entries.append(
            {
                "relative_path": f"{split}/weather.npy",
                "file": "weather.npy",
                "shape": list(weather_shape),
                "dtype": str(self.weather_dtype),
            }
        )

        valid_positions = np.searchsorted(power_time, valid_unix)
        if (
            np.any(valid_positions >= len(power_time))
            or not np.array_equal(power_time[valid_positions], valid_unix)
        ):
            raise ValueError(f"NWP valid times do not align with station power for {split}")
        weather_target = (
            station_power[valid_positions] / capacity.reshape(1, 1, -1)
        ).transpose(0, 2, 1).astype(np.float32)
        weather_warning = capacity_warning[valid_positions].transpose(0, 2, 1)
        if not np.isfinite(weather_target).all():
            raise ValueError(f"Stage-I station targets contain missing values for {split}")

        origins = np.asarray(sample_group["origin_unix"], dtype=np.int64)
        power_start = np.asarray(sample_group["power_start_index"], dtype=np.int64)
        history_offsets = np.arange(self.history_power_steps, dtype=np.int64)
        future_offsets = self.history_power_steps + np.arange(
            self.forecast_power_steps, dtype=np.int64
        )
        history_positions = power_start[:, None] + history_offsets
        future_positions = power_start[:, None] + future_offsets
        station_history = (
            station_power[history_positions] / capacity.reshape(1, 1, -1)
        ).transpose(0, 2, 1).astype(np.float32)
        station_future = (
            station_power[future_positions] / capacity.reshape(1, 1, -1)
        ).transpose(0, 2, 1).astype(np.float32)
        future_warning = capacity_warning[future_positions].transpose(0, 2, 1)
        if not np.isfinite(station_history).all() or not np.isfinite(station_future).all():
            raise ValueError(f"Accepted sample power contains missing values for {split}")
        cluster_history = np.asarray(sample_group["history_power"], dtype=np.float32)
        cluster_future = np.asarray(sample_group["future_power"], dtype=np.float32)
        reconstructed = (
            station_future * capacity.reshape(1, -1, 1)
        ).sum(axis=1) / total_capacity
        if not np.allclose(reconstructed, cluster_future, atol=2e-6, rtol=1e-6):
            raise ValueError(f"Station and cluster future targets disagree for {split}")

        sample_values = {
            "origin_unix": origins,
            "sample_index": np.asarray(sample_group["sample_index"], dtype=np.int64),
            "origin_phase": np.asarray(sample_group["origin_phase"], dtype=np.int64),
            "nwp_age_minutes": np.asarray(sample_group["nwp_age_minutes"], dtype=np.int64),
            "weather_cycle_index": local_cycle_index.astype(np.int64),
            "weather_time_offset_hours": (
                valid_unix[local_cycle_index] - origins[:, None]
            ).astype(np.float32)
            / 3600.0,
            "station_history": station_history,
            "station_future": station_future,
            "cluster_history": cluster_history,
            "cluster_future": cluster_future,
            "future_capacity_warning": future_warning,
        }
        cycle_values = {
            "weather_valid_unix": valid_unix,
            "weather_station_target": weather_target,
            "weather_station_capacity_warning": weather_warning,
            "cycle_sample_index": cycle_ids,
        }
        for name, values in {**sample_values, **cycle_values}.items():
            entry = _save_array(output_dir, name, values)
            entries.append(
                {"relative_path": f"{split}/{entry['file']}", **entry}
            )
        return (
            {
                "sample_count": int(len(origins)),
                "cycle_count": int(len(cycle_ids)),
                "weather_shape": list(weather_shape[1:]),
                "station_history_shape": list(station_history.shape[1:]),
                "station_future_shape": list(station_future.shape[1:]),
            },
            entries,
        )
