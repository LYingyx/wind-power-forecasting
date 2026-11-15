from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


SPLITS = ("train", "val", "test")


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared model cache is missing: {path}. Run prepare_model_data.py first."
        )
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported model-cache schema: {manifest.get('schema_version')}")
    return manifest


def _load_npy(path: Path, *, mmap: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def _tensor(values: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    array = np.ascontiguousarray(values)
    tensor = torch.from_numpy(array)
    return tensor if dtype is None else tensor.to(dtype=dtype)


class PreparedCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest = _load_manifest(self.cache_dir)
        static_dir = self.cache_dir / "static"
        names = (
            "station_id",
            "station_name",
            "station_latitude",
            "station_longitude",
            "station_capacity_mw",
            "weather_latitude",
            "weather_longitude",
            "nwp_mean",
            "nwp_std",
            "nwp_channel_name",
        )
        self.static = {name: _load_npy(static_dir / f"{name}.npy") for name in names}
        station_count = int(self.manifest["station_count"])
        if self.static["station_capacity_mw"].shape != (station_count,):
            raise ValueError("Prepared station metadata has an inconsistent shape")


class _WeatherStore:
    def __init__(
        self,
        path: Path,
        preload: bool | str,
        preload_max_gib: float,
        verbose: bool,
    ) -> None:
        mmap = _load_npy(path, mmap=True)
        self.nbytes = int(mmap.nbytes)
        setting = str(preload).lower() if isinstance(preload, str) else preload
        should_preload = bool(setting is True or setting in {"1", "true", "yes"})
        if setting == "auto":
            should_preload = self.nbytes <= int(preload_max_gib * 1024**3)
        if setting in {False, "0", "false", "no", "mmap"}:
            should_preload = False
        self.values = np.array(mmap, copy=True) if should_preload else mmap
        self.backend = "ram" if should_preload else "mmap"
        if verbose:
            print(
                f"[weather-store] backend={self.backend}, "
                f"shape={list(self.values.shape)}, size={self.nbytes / 1024**3:.2f} GiB"
            )

    def take(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.values[indices], dtype=np.float32)

    def close(self) -> None:
        mmap = getattr(self.values, "_mmap", None)
        if mmap is not None:
            mmap.close()


class ForecastDataset(Dataset[int]):
    """RAM-resident targets and indexed access to unique weather cycles."""

    sample_array_names = (
        "origin_unix",
        "sample_index",
        "origin_phase",
        "nwp_age_minutes",
        "weather_cycle_index",
        "weather_time_offset_hours",
        "station_history",
        "station_future",
        "cluster_history",
        "cluster_future",
        "future_capacity_warning",
    )
    cycle_array_names = (
        "weather_valid_unix",
        "weather_station_target",
        "weather_station_capacity_warning",
        "cycle_sample_index",
    )

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        *,
        preload_weather: bool | str = "auto",
        preload_weather_max_gib: float = 16.0,
        include_weather: bool = True,
        response_state_cache_dir: str | Path | None = None,
        response_state_donor_mode: str = "none",
        verbose: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.cache = PreparedCache(cache_dir)
        self.split = split
        split_dir = self.cache.cache_dir / split
        self.arrays = {
            name: _load_npy(split_dir / f"{name}.npy")
            for name in self.sample_array_names
        }
        self.include_weather = bool(include_weather)
        self.cycle_arrays = (
            {
                name: _load_npy(split_dir / f"{name}.npy")
                for name in self.cycle_array_names
            }
            if self.include_weather
            else {}
        )
        self.weather = (
            _WeatherStore(
                split_dir / "weather.npy",
                preload_weather,
                preload_weather_max_gib,
                verbose,
            )
            if self.include_weather
            else None
        )
        self.length = int(len(self.arrays["origin_unix"]))
        self.teacher_arrays: dict[str, np.ndarray] = {}
        if response_state_donor_mode not in {"none", "strong_sample"}:
            raise ValueError("response_state_donor_mode must be none or strong_sample")
        self.response_state_donor_mode = response_state_donor_mode
        self.response_state_manifest: dict[str, Any] | None = None
        self.response_state_arrays: dict[str, np.ndarray] = {}
        self.past_weather: _WeatherStore | None = None
        if response_state_cache_dir is not None:
            state_root = Path(response_state_cache_dir).resolve()
            with (state_root / "manifest.json").open(encoding="utf-8") as stream:
                self.response_state_manifest = json.load(stream)
            if self.response_state_manifest.get("base_cache_fingerprint") != self.manifest.get(
                "preprocess_fingerprint"
            ):
                raise ValueError("Response-state sidecar and forecast cache have different provenance")
            state_split = state_root / split
            for name in (
                "past_weather_cycle_index",
                "past_weather_time_offset_hours",
                "past_station_power",
                "strong_sample_donor_index",
            ):
                self.response_state_arrays[name] = _load_npy(state_split / f"{name}.npy")
            self.past_weather = _WeatherStore(
                state_split / "past_weather.npy",
                preload_weather,
                preload_weather_max_gib,
                verbose,
            )
            if any(len(value) != self.length for value in self.response_state_arrays.values()):
                raise ValueError("Response-state sample arrays have inconsistent lengths")
        lengths = {name: len(values) for name, values in self.arrays.items()}
        if len(set(lengths.values())) != 1 or self.length == 0:
            raise ValueError(f"Prepared sample arrays have inconsistent lengths: {lengths}")
        if self.include_weather:
            cycle_count = int(len(self.cycle_arrays["cycle_sample_index"]))
            if self.weather is None or len(self.weather.values) != cycle_count:
                raise ValueError("Weather cycles and cycle metadata have different lengths")
            cycle_index = self.arrays["weather_cycle_index"]
            if cycle_index.min() < 0 or cycle_index.max() >= cycle_count:
                raise ValueError("Sample weather_cycle_index falls outside the cycle array")
        else:
            cycle_count = 0
        if verbose:
            print(
                f"[forecast-dataset:{split}] samples={self.length}, cycles={cycle_count}, "
                f"include_weather={self.include_weather}, "
                f"history={list(self.arrays['station_history'].shape[1:])}, "
                f"target={list(self.arrays['station_future'].shape[1:])}"
            )

    @property
    def static(self) -> dict[str, np.ndarray]:
        return self.cache.static

    @property
    def manifest(self) -> dict[str, Any]:
        return self.cache.manifest

    def __len__(self) -> int:
        return self.length

    def attach_teacher_archive(self, path: str | Path) -> None:
        """Attach aligned ensemble targets used only by the training collator."""
        with np.load(Path(path), allow_pickle=False) as archive:
            required = {
                "sample_index",
                "origin_unix",
                "teacher_station_factor_mean",
                "teacher_station_factor_std",
                "teacher_cluster_factor_mean",
                "teacher_cluster_factor_std",
            }
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"Teacher archive is missing fields: {sorted(missing)}")
            np.testing.assert_array_equal(
                archive["sample_index"], self.arrays["sample_index"]
            )
            np.testing.assert_array_equal(
                archive["origin_unix"], self.arrays["origin_unix"]
            )
            arrays = {
                name: np.asarray(archive[name], dtype=np.float32)
                for name in required.difference({"sample_index", "origin_unix"})
            }
        station_shape = self.arrays["station_future"].shape
        cluster_shape = self.arrays["cluster_future"].shape
        if arrays["teacher_station_factor_mean"].shape != station_shape:
            raise ValueError("Teacher station mean has the wrong shape")
        if arrays["teacher_station_factor_std"].shape != station_shape:
            raise ValueError("Teacher station standard deviation has the wrong shape")
        if arrays["teacher_cluster_factor_mean"].shape != cluster_shape:
            raise ValueError("Teacher cluster mean has the wrong shape")
        if arrays["teacher_cluster_factor_std"].shape != cluster_shape:
            raise ValueError("Teacher cluster standard deviation has the wrong shape")
        if any(not np.isfinite(values).all() for values in arrays.values()):
            raise ValueError("Teacher archive contains non-finite values")
        self.teacher_arrays = arrays

    def __getitem__(self, index: int) -> int:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return index

    def close(self) -> None:
        if self.weather is not None:
            self.weather.close()
        if self.past_weather is not None:
            self.past_weather.close()

    def __del__(self) -> None:
        weather = getattr(self, "weather", None)
        if weather is not None:
            weather.close()
        past_weather = getattr(self, "past_weather", None)
        if past_weather is not None:
            past_weather.close()


def apply_stratified_weather_permutation(
    dataset: ForecastDataset,
    *,
    seed: int,
    power_bins: int = 10,
) -> dict[str, Any]:
    """Replace each sample's weather cycle within an in-distribution stratum."""
    if not dataset.include_weather:
        raise ValueError("Weather permutation requires a weather-enabled dataset")
    if power_bins < 2:
        raise ValueError("power_bins must be at least two")

    origins = np.asarray(dataset.arrays["origin_unix"], dtype=np.int64)
    months = (
        origins.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
        % 12
    ) + 1
    phases = np.asarray(dataset.arrays["origin_phase"], dtype=np.int64)
    nwp_age = np.asarray(dataset.arrays["nwp_age_minutes"], dtype=np.int64)
    current_power = np.asarray(
        dataset.arrays["cluster_history"][:, -1], dtype=np.float64
    )
    quantile_edges = np.quantile(
        current_power, np.linspace(0.0, 1.0, power_bins + 1)
    )
    power_bin = np.searchsorted(
        quantile_edges[1:-1], current_power, side="right"
    )
    strata = np.stack((months, phases, nwp_age, power_bin), axis=1)
    _, inverse = np.unique(strata, axis=0, return_inverse=True)

    rng = np.random.default_rng(int(seed))
    donor_index = np.empty(len(dataset), dtype=np.int64)
    stratum_sizes: list[int] = []
    for stratum in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == stratum)
        if len(members) < 2:
            raise ValueError(
                "Weather-permutation stratum has fewer than two samples; "
                "use fewer power bins"
            )
        shuffled = members.copy()
        rng.shuffle(shuffled)
        shift = int(rng.integers(1, len(shuffled)))
        donor_index[shuffled] = np.roll(shuffled, shift)
        stratum_sizes.append(int(len(members)))

    if np.any(donor_index == np.arange(len(dataset))):
        raise RuntimeError("Controlled weather permutation contains fixed samples")
    donor_offsets = dataset.arrays["weather_time_offset_hours"][donor_index]
    np.testing.assert_allclose(
        donor_offsets,
        dataset.arrays["weather_time_offset_hours"],
        atol=0.0,
        rtol=0.0,
    )
    original_cycle = np.asarray(
        dataset.arrays["weather_cycle_index"], dtype=np.int64
    )
    permuted_cycle = original_cycle[donor_index].copy()
    np.testing.assert_array_equal(
        np.sort(permuted_cycle), np.sort(original_cycle)
    )
    same_cycle = int(np.sum(permuted_cycle == original_cycle))
    if same_cycle != 0:
        raise RuntimeError(
            "Controlled weather permutation retained one or more original cycles"
        )
    dataset.arrays["weather_cycle_index"] = permuted_cycle
    checksum = hashlib.sha256(donor_index.tobytes()).hexdigest()
    return {
        "seed": int(seed),
        "power_bins": int(power_bins),
        "stratification": "UTC_month x origin_phase x NWP_age x current_power_quantile",
        "strata": len(stratum_sizes),
        "minimum_stratum_size": min(stratum_sizes),
        "maximum_stratum_size": max(stratum_sizes),
        "fixed_samples": int(np.sum(donor_index == np.arange(len(dataset)))),
        "same_weather_cycles": same_cycle,
        "donor_index_sha256": checksum,
        "power_quantile_edges": quantile_edges.tolist(),
    }


class SpatialPretrainDataset(Dataset[int]):
    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        *,
        preload_weather: bool | str = "auto",
        preload_weather_max_gib: float = 16.0,
        verbose: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.cache = PreparedCache(cache_dir)
        self.split = split
        split_dir = self.cache.cache_dir / split
        self.weather = _WeatherStore(
            split_dir / "weather.npy",
            preload_weather,
            preload_weather_max_gib,
            verbose,
        )
        self.target = _load_npy(split_dir / "weather_station_target.npy")
        self.warning = _load_npy(split_dir / "weather_station_capacity_warning.npy")
        self.valid_unix = _load_npy(split_dir / "weather_valid_unix.npy")
        self.cycle_sample_index = _load_npy(split_dir / "cycle_sample_index.npy")
        self.length = int(len(self.target))
        if not all(
            len(values) == self.length
            for values in (self.weather.values, self.warning, self.valid_unix, self.cycle_sample_index)
        ):
            raise ValueError("Spatial-pretraining cycle arrays have inconsistent lengths")
        if verbose:
            print(
                f"[spatial-dataset:{split}] cycles={self.length}, "
                f"weather={list(self.weather.values.shape[1:])}, "
                f"target={list(self.target.shape[1:])}"
            )

    @property
    def static(self) -> dict[str, np.ndarray]:
        return self.cache.static

    @property
    def manifest(self) -> dict[str, Any]:
        return self.cache.manifest

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return index

    def close(self) -> None:
        self.weather.close()

    def __del__(self) -> None:
        weather = getattr(self, "weather", None)
        if weather is not None:
            weather.close()


class CycleGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle weather cycles while keeping their quarter-hour origins together."""

    def __init__(
        self,
        cycle_index: Sequence[int],
        cycles_per_batch: int,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        if cycles_per_batch < 1:
            raise ValueError("cycles_per_batch must be positive")
        values = np.asarray(cycle_index, dtype=np.int64)
        if values.ndim != 1 or not len(values):
            raise ValueError("cycle_index must be a non-empty vector")
        self.cycles_per_batch = cycles_per_batch
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self.unique_cycles = np.unique(values)
        self.members = {
            int(cycle): np.flatnonzero(values == cycle).astype(np.int64)
            for cycle in self.unique_cycles
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (len(self.unique_cycles) + self.cycles_per_batch - 1) // self.cycles_per_batch

    def __iter__(self) -> Iterator[list[int]]:
        cycles = self.unique_cycles.copy()
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(cycles)
        for start in range(0, len(cycles), self.cycles_per_batch):
            selected = cycles[start : start + self.cycles_per_batch]
            batch = np.concatenate([self.members[int(cycle)] for cycle in selected])
            if self.shuffle:
                rng.shuffle(batch)
            yield batch.tolist()


class ForecastCollator:
    def __init__(self, dataset: ForecastDataset) -> None:
        self.dataset = dataset

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        sample_index = np.asarray(indices, dtype=np.int64)
        batch: dict[str, torch.Tensor] = {}
        if self.dataset.include_weather:
            sample_cycles = np.asarray(
                self.dataset.arrays["weather_cycle_index"][sample_index], dtype=np.int64
            )
            unique_cycles, weather_batch_index = np.unique(
                sample_cycles, return_inverse=True
            )
            if self.dataset.weather is None:
                raise RuntimeError("Weather storage is unavailable")
            batch.update(
                weather=_tensor(
                    self.dataset.weather.take(unique_cycles), torch.float32
                ),
                weather_batch_index=_tensor(weather_batch_index, torch.long),
                weather_station_target=_tensor(
                    self.dataset.cycle_arrays["weather_station_target"][unique_cycles],
                    torch.float32,
                ),
                weather_station_capacity_warning=_tensor(
                    self.dataset.cycle_arrays[
                        "weather_station_capacity_warning"
                    ][unique_cycles],
                    torch.bool,
                ),
            )
        for name, values in self.dataset.arrays.items():
            selected = values[sample_index]
            if selected.dtype == np.bool_:
                dtype = torch.bool
            elif np.issubdtype(selected.dtype, np.floating):
                dtype = torch.float32
            else:
                dtype = torch.long
            batch[name] = _tensor(selected, dtype)
        for name, values in self.dataset.teacher_arrays.items():
            batch[name] = _tensor(values[sample_index], torch.float32)
        if self.dataset.response_state_manifest is not None:
            state_arrays = self.dataset.response_state_arrays
            state_index = sample_index
            if self.dataset.response_state_donor_mode == "strong_sample":
                state_index = np.asarray(
                    state_arrays["strong_sample_donor_index"][sample_index], dtype=np.int64
                )
            batch["state_station_history"] = _tensor(
                self.dataset.arrays["station_history"][state_index], torch.float32
            )
            batch["state_cluster_history"] = _tensor(
                self.dataset.arrays["cluster_history"][state_index], torch.float32
            )
            batch["past_station_power"] = _tensor(
                state_arrays["past_station_power"][state_index], torch.float32
            )
            past_cycles = np.asarray(
                state_arrays["past_weather_cycle_index"][state_index], dtype=np.int64
            )
            unique_past, past_batch_index = np.unique(past_cycles, return_inverse=True)
            if self.dataset.past_weather is None:
                raise RuntimeError("Historical response-state weather is unavailable")
            batch["past_weather"] = _tensor(
                self.dataset.past_weather.take(unique_past), torch.float32
            )
            batch["past_weather_batch_index"] = _tensor(past_batch_index, torch.long)
            batch["past_weather_time_offset_hours"] = _tensor(
                state_arrays["past_weather_time_offset_hours"][state_index], torch.float32
            )
            batch["response_state_sample_index"] = _tensor(state_index, torch.long)
        return batch


class SpatialPretrainCollator:
    def __init__(self, dataset: SpatialPretrainDataset) -> None:
        self.dataset = dataset

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        index = np.asarray(indices, dtype=np.int64)
        return {
            "weather": _tensor(self.dataset.weather.take(index), torch.float32),
            "station_target": _tensor(self.dataset.target[index], torch.float32),
            "capacity_warning": _tensor(self.dataset.warning[index], torch.bool),
            "weather_valid_unix": _tensor(self.dataset.valid_unix[index], torch.long),
            "cycle_sample_index": _tensor(
                self.dataset.cycle_sample_index[index], torch.long
            ),
        }


def _loader_kwargs(num_workers: int, prefetch_factor: int) -> dict[str, Any]:
    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("num_workers must be non-negative and prefetch_factor positive")
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def create_forecast_loader(
    dataset: ForecastDataset,
    *,
    cycles_per_batch: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> DataLoader[dict[str, torch.Tensor]]:
    if not dataset.include_weather:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return DataLoader(
            dataset,
            batch_size=4 * cycles_per_batch,
            shuffle=shuffle,
            generator=generator,
            collate_fn=ForecastCollator(dataset),
            **_loader_kwargs(num_workers, prefetch_factor),
        )
    sampler = CycleGroupedBatchSampler(
        dataset.arrays["weather_cycle_index"], cycles_per_batch, shuffle, seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=ForecastCollator(dataset),
        **_loader_kwargs(num_workers, prefetch_factor),
    )


def create_spatial_pretrain_loader(
    dataset: SpatialPretrainDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=SpatialPretrainCollator(dataset),
        **_loader_kwargs(num_workers, prefetch_factor),
    )
