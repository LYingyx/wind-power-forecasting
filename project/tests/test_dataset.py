from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wind_forcing.data import (
    ForecastDataset,
    apply_stratified_weather_permutation,
    create_forecast_loader,
)


CACHE = Path(__file__).parents[1] / "data" / "model_cache_pad5_f8"


@pytest.mark.skipif(not (CACHE / "manifest.json").is_file(), reason="prepared cache absent")
def test_real_cache_loader_deduplicates_weather_cycles() -> None:
    dataset = ForecastDataset(CACHE, "train", preload_weather="mmap", verbose=False)
    loader = create_forecast_loader(
        dataset, cycles_per_batch=1, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    assert batch["weather"].shape == (1, 8, 14, 24, 25)
    assert batch["station_history"].shape[1:] == (56, 24)
    assert batch["station_future"].shape[1:] == (56, 24)
    assert batch["weather_batch_index"].unique().tolist() == [0]
    capacity = dataset.static["station_capacity_mw"].reshape(1, -1, 1)
    reconstructed = (batch["station_future"].numpy() * capacity).sum(axis=1)
    reconstructed /= float(dataset.manifest["total_capacity_mw"])
    np.testing.assert_allclose(
        reconstructed, batch["cluster_future"].numpy(), atol=2e-6, rtol=1e-6
    )
    dataset.close()


@pytest.mark.skipif(not (CACHE / "manifest.json").is_file(), reason="prepared cache absent")
def test_history_loader_does_not_open_or_transfer_weather() -> None:
    dataset = ForecastDataset(
        CACHE, "train", preload_weather="mmap", include_weather=False, verbose=False
    )
    loader = create_forecast_loader(
        dataset, cycles_per_batch=1, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    assert dataset.weather is None
    assert "weather" not in batch
    assert "weather_batch_index" not in batch
    assert batch["station_history"].shape == (4, 56, 24)
    dataset.close()


@pytest.mark.skipif(not (CACHE / "manifest.json").is_file(), reason="prepared cache absent")
def test_weather_permutation_is_stratified_deranged_and_distribution_preserving() -> None:
    dataset = ForecastDataset(
        CACHE, "val", preload_weather="mmap", include_weather=True, verbose=False
    )
    original = dataset.arrays["weather_cycle_index"].copy()
    metadata = apply_stratified_weather_permutation(
        dataset, seed=7, power_bins=10
    )
    permuted = dataset.arrays["weather_cycle_index"]

    assert metadata["fixed_samples"] == 0
    assert metadata["same_weather_cycles"] == 0
    assert metadata["minimum_stratum_size"] >= 2
    assert np.all(permuted != original)
    np.testing.assert_array_equal(np.sort(permuted), np.sort(original))
    dataset.close()
