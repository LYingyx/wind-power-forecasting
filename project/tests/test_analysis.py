from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_final_route import compare_groups
from scripts.analyze_conditional_forcing_validation import block_bootstrap_matrix
from scripts.analyze_incremental_mechanism_v2 import paired_utc_day_block_bootstrap
from scripts.analyze_paper_results import align_predictions
from scripts.analyze_state_forcing_regimes import grouped_bootstrap


def _prediction_archive(origins: list[int]) -> dict[str, np.ndarray]:
    origin_array = np.asarray(origins, dtype=np.int64)
    target = np.stack((origin_array, origin_array + 1), axis=1).astype(np.float32)
    return {
        "origin_unix": origin_array,
        "cluster_target_mw": target,
        "cluster_history_last_mw": origin_array.astype(np.float32),
        "cluster_prediction_mw": target + 1.0,
    }


def test_prediction_alignment_uses_sorted_common_origins() -> None:
    reference, candidate = align_predictions(
        _prediction_archive([30, 10, 20]),
        _prediction_archive([40, 20, 30]),
    )
    np.testing.assert_array_equal(reference["origin_unix"], [20, 30])
    np.testing.assert_array_equal(candidate["origin_unix"], [20, 30])
    np.testing.assert_allclose(
        reference["cluster_target_mw"], candidate["cluster_target_mw"]
    )


def test_prediction_alignment_rejects_duplicate_origins() -> None:
    with pytest.raises(ValueError, match="duplicate origins"):
        align_predictions(
            _prediction_archive([10, 10]),
            _prediction_archive([10, 20]),
        )


def test_final_route_comparison_reports_candidate_minus_reference(
    tmp_path,
) -> None:
    candidate = tmp_path / "candidate" / "seed42"
    reference = tmp_path / "reference" / "seed42"
    candidate.mkdir(parents=True)
    reference.mkdir(parents=True)
    candidate_archive = _prediction_archive([86400, 172800, 259200])
    reference_archive = _prediction_archive([86400, 172800, 259200])
    candidate_archive["cluster_prediction_mw"] = (
        candidate_archive["cluster_target_mw"] + 1.0
    )
    reference_archive["cluster_prediction_mw"] = (
        reference_archive["cluster_target_mw"] + 3.0
    )
    np.savez(candidate / "predictions_val.npz", **candidate_archive)
    np.savez(reference / "predictions_val.npz", **reference_archive)

    result = compare_groups(
        "candidate",
        candidate.parent,
        "reference",
        reference.parent,
        scientific_question="test",
        repeats=100,
        seed=1,
    )

    assert result["candidate_minus_reference_mae_mw"] == pytest.approx(-2.0)
    assert result["candidate_better"] is True


def test_incremental_bootstrap_point_estimate_remains_sample_weighted() -> None:
    delta, _, _, blocks = paired_utc_day_block_bootstrap(
        candidate_error=np.asarray([0.0, 0.0, 3.0]),
        reference_error=np.zeros(3),
        origins=np.asarray([86400, 86460, 172800]),
        repeats=100,
        seed=1,
    )

    assert delta == pytest.approx(1.0)
    assert blocks == 2


def test_horizon_bootstrap_point_estimate_remains_sample_weighted() -> None:
    mean, _, _, blocks = block_bootstrap_matrix(
        delta=np.asarray([[0.0, 0.0], [0.0, 0.0], [3.0, 6.0]]),
        origins=np.asarray([86400, 86460, 172800]),
        repeats=100,
        seed=1,
    )

    np.testing.assert_allclose(mean, [1.0, 2.0])
    assert blocks == 2


def test_grouped_bootstrap_reports_and_filters_empty_group_resamples() -> None:
    bootstrap_counts = np.tile(np.asarray([[1, 0]], dtype=np.int64), (100, 1))
    bootstrap_counts[-1] = [0, 1]
    point, low, high, samples, days, effective = grouped_bootstrap(
        values=np.asarray([2.0, 8.0]),
        mask=np.asarray([True, False]),
        day_inverse=np.asarray([0, 1]),
        bootstrap_counts=bootstrap_counts,
    )

    np.testing.assert_allclose(point, [2.0])
    np.testing.assert_allclose(low, [2.0])
    np.testing.assert_allclose(high, [2.0])
    assert samples == 1
    assert days == 1
    assert effective == 99
