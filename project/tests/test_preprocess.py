from __future__ import annotations

import numpy as np

from wind_forcing.data.preprocess import compute_station_envelope_crop


def test_pad5_crop_uses_exact_station_envelope() -> None:
    latitude = np.arange(47.75, 39.49, -0.25)
    longitude = np.arange(120.25, 132.76, 0.25)
    crop = compute_station_envelope_crop(
        latitude,
        longitude,
        [42.8333, 45.9714],
        [122.2136, 125.3206],
        5,
    )
    assert (crop.row_start, crop.row_stop) == (2, 26)
    assert (crop.column_start, crop.column_stop) == (2, 27)
    assert (crop.height, crop.width) == (24, 25)
    assert (crop.north, crop.south, crop.west, crop.east) == (
        47.25,
        41.5,
        120.75,
        126.75,
    )
