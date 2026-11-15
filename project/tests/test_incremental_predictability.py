from __future__ import annotations

import numpy as np

from scripts.analyze_incremental_predictability import (
    bilinear_weight_matrix,
    chronological_day_split,
    spatial_projection,
    time_low_pass,
)


def test_chronological_split_keeps_whole_days_disjoint() -> None:
    days = np.repeat(np.arange(20), 4)
    origins = (days * 86_400 + np.tile(np.arange(4), 20) * 900).astype(np.int64)
    groups = chronological_day_split(origins)
    group_days = {
        name: set(np.floor_divide(origins[indices], 86_400))
        for name, indices in groups.items()
    }
    assert group_days["fit"].isdisjoint(group_days["selection"])
    assert group_days["fit"].isdisjoint(group_days["confirmation"])
    assert group_days["selection"].isdisjoint(group_days["confirmation"])
    assert max(group_days["fit"]) < min(group_days["selection"])
    assert max(group_days["selection"]) < min(group_days["confirmation"])


def test_bilinear_weights_are_normalized_at_station_locations() -> None:
    weights = bilinear_weight_matrix(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.5, 1.0]),
        np.array([0.5, 0.0]),
    )
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    np.testing.assert_allclose(weights[:, 0], 0.25)
    np.testing.assert_allclose(weights[:, 1], [1.0, 0.0, 0.0, 0.0])


def test_low_pass_and_spatial_projection_preserve_selected_subspaces() -> None:
    time_signal = np.ones((2, 3, 24), dtype=np.float32)
    np.testing.assert_allclose(time_low_pass(time_signal, 1), time_signal, atol=1e-5)

    values = np.zeros((2, 3, 4), dtype=np.float32)
    values[:, 0] = 2.0
    components = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(spatial_projection(values, components), values)
