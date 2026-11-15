from __future__ import annotations

import torch

from wind_forcing.training import MetricAccumulator, deterministic_loss


def settings() -> dict[str, object]:
    return {
        "station_loss_weight": 1.0,
        "cluster_loss_weight": 0.0,
        "auxiliary_weather_loss_weight": 0.0,
        "distillation_weight": 0.0,
        "baseline_auxiliary_loss_weight": 0.0,
        "baseline_anchor_loss_weight": 0.0,
        "ramp_loss_weight": 0.0,
        "station_loss_indices": [0],
    }


def test_heldout_target_never_changes_training_loss() -> None:
    output = {
        "station_factor": torch.zeros(2, 2, 3, requires_grad=True),
        "cluster_factor": torch.zeros(2, 3),
    }
    batch = {
        "station_future": torch.zeros(2, 2, 3),
        "cluster_future": torch.zeros(2, 3),
    }
    first, _ = deterministic_loss(output, batch, settings())
    changed = {key: value.clone() for key, value in batch.items()}
    changed["station_future"][:, 1] = 1.0
    changed["cluster_future"][:] = 1.0
    second, _ = deterministic_loss(output, changed, settings())
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_checkpoint_metric_can_be_restricted_to_training_farms() -> None:
    accumulator = MetricAccumulator(3, 30.0, metric_station_indices=[0])
    prediction = torch.zeros(2, 2, 3)
    target = torch.zeros_like(prediction)
    target[:, 1] = 1.0
    accumulator.update(
        prediction,
        torch.zeros(2, 3),
        target,
        torch.ones(2, 3),
        torch.tensor([10.0, 20.0]),
        torch.zeros(2),
    )
    assert accumulator.compute()["station_mae_cf"] == 0.0
