from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_local_state_cluster_transfer_v2 import (
    paired_summary,
    passes_paired,
    residual_decomposition,
)


def test_residual_covariance_identity_closes_exactly() -> None:
    target = np.zeros((5, 3, 4), dtype=np.float64)
    prediction = np.asarray(
        [
            [[1, 2, 3, 4], [-1, 0, 1, 2], [2, -2, 2, -2]],
            [[2, 1, 4, 3], [-2, 1, 0, 3], [1, -1, 1, -1]],
            [[0, 1, 2, 3], [1, 2, 3, 4], [-2, 2, -2, 2]],
            [[3, 2, 1, 0], [0, -1, -2, -3], [3, 1, -1, -3]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [-1, -1, -1, -1]],
        ],
        dtype=np.float64,
    )
    result = residual_decomposition(prediction, target)
    reconstructed = (
        result["aggregate_bias_squared_mw2"]
        + result["diagonal_variance_mw2"]
        + result["cross_covariance_mw2"]
    )
    assert result["aggregate_mse_mw2"] == pytest.approx(reconstructed, abs=1e-12)
    assert result["closure_error_mw2"] == pytest.approx(0.0, abs=1e-12)


def test_frozen_decision_requires_ci_and_seven_wins() -> None:
    passing = paired_summary(
        [-2.0, -1.0, -3.0, -1.5, -2.5, -1.0, -2.0, -1.2, -1.8, -2.2],
        candidate="int",
        control="add",
        metric_name="mae",
        split="val",
    )
    assert passes_paired(passing)

    too_few_wins = dict(passing, wins=6)
    assert not passes_paired(too_few_wins)
    crossing = dict(passing, ci95=[-2.0, 0.01])
    assert not passes_paired(crossing)


def test_residual_decomposition_rejects_non_station_tensor() -> None:
    with pytest.raises(ValueError, match="sample,farm,lead"):
        residual_decomposition(np.zeros((2, 3)), np.zeros((2, 3)))
