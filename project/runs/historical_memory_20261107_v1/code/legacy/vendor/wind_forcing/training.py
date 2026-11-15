from __future__ import annotations

from collections import deque
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True) for key, value in batch.items()
    }


def model_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    accepted = (
        "weather",
        "weather_batch_index",
        "weather_time_offset_hours",
        "station_history",
        "cluster_history",
        "origin_phase",
        "state_station_history",
        "state_cluster_history",
        "past_weather",
        "past_weather_batch_index",
        "past_weather_time_offset_hours",
        "past_station_power",
    )
    return {key: batch[key] for key in accepted if key in batch}


def autocast_context(device: torch.device, precision: str):
    precision = precision.lower()
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    if precision not in {"bf16", "fp16"}:
        raise ValueError("precision must be float32, bf16, or fp16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _smooth_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction.float(), target.float(), beta=0.05)


def _configured_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    settings: Mapping[str, Any],
) -> torch.Tensor:
    """Use an explicit loss name; never silently call Smooth L1."""
    mode = str(settings.get("loss_mode", "smooth_l1")).lower()
    if mode == "mae":
        return (prediction.float() - target.float()).abs().mean()
    if mode == "smooth_l1":
        return _smooth_l1(prediction, target)
    raise ValueError(f"Unsupported loss_mode={mode!r}; choose mae or smooth_l1")


def _weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    values = F.smooth_l1_loss(
        prediction.float(), target.float(), beta=0.05, reduction="none"
    )
    weights = weight.float().expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def _horizon_anchor_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "scalar":
        return _smooth_l1(prediction, target)
    if mode != "horizon_taper":
        raise ValueError("anchor_schedule_mode must be scalar or horizon_taper")
    steps = prediction.shape[-1]
    if target.shape[-1] != steps:
        raise ValueError("Anchor prediction and target horizons are inconsistent")
    lead_hours = torch.arange(
        1, steps + 1, dtype=torch.float32, device=prediction.device
    ) / 4.0
    weights = torch.where(
        lead_hours <= 3.0,
        torch.ones_like(lead_hours),
        (4.0 - lead_hours).clamp(0.0, 1.0),
    )
    values = F.smooth_l1_loss(
        prediction.float(), target.float(), beta=0.05, reduction="none"
    )
    shape = (1,) * (values.ndim - 1) + (steps,)
    return (values * weights.reshape(shape)).mean()


def spatial_pretrain_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    capacity_mw: torch.Tensor,
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    station_target = batch["station_target"]
    cluster_target = (
        station_target * capacity_mw.reshape(1, -1, 1)
    ).sum(dim=1) / capacity_mw.sum()
    station = _configured_loss(output["hourly_station_factor"], station_target, settings)
    cluster = _configured_loss(output["cluster_factor"], cluster_target, settings)
    loss = (
        float(settings["station_loss_weight"]) * station
        + float(settings["cluster_loss_weight"]) * cluster
    )
    return loss, {"station": float(station.detach()), "cluster": float(cluster.detach())}


def deterministic_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    cluster = _configured_loss(output["cluster_factor"], batch["cluster_future"], settings)
    station_prediction = output.get("station_factor")
    auxiliary_prediction = output.get("hourly_station_factor_unique")
    station_weight = float(settings["station_loss_weight"])
    auxiliary_weight = float(settings["auxiliary_weather_loss_weight"])
    station_indices_value = settings.get("station_loss_indices")
    station_indices: torch.Tensor | None = None
    if station_indices_value is not None:
        station_indices = torch.as_tensor(
            station_indices_value, dtype=torch.long, device=cluster.device
        )
        if station_indices.ndim != 1 or station_indices.numel() == 0:
            raise ValueError("station_loss_indices must be a non-empty 1-D sequence")
        if station_prediction is not None and (
            int(station_indices.min()) < 0
            or int(station_indices.max()) >= station_prediction.shape[1]
            or station_indices.unique().numel() != station_indices.numel()
        ):
            raise ValueError("station_loss_indices contain invalid or duplicate indices")

    def station_view(value: torch.Tensor) -> torch.Tensor:
        return value if station_indices is None else value.index_select(1, station_indices)

    if station_prediction is None:
        if station_weight != 0.0:
            raise ValueError("station_loss_weight must be zero for direct-cluster models")
        station = cluster.detach() * 0.0
    else:
        station = _configured_loss(
            station_view(station_prediction), station_view(batch["station_future"]), settings
        )
    if auxiliary_prediction is None:
        if auxiliary_weight != 0.0:
            raise ValueError(
                "auxiliary_weather_loss_weight must be zero when no hourly weather head exists"
            )
        auxiliary = cluster.detach() * 0.0
    else:
        auxiliary = _configured_loss(
            auxiliary_prediction, batch["weather_station_target"]
            , settings
        )
    supervised = (
        station_weight * station
        + float(settings["cluster_loss_weight"]) * cluster
    )
    distillation_weight = float(settings.get("distillation_weight", 0.0))
    if distillation_weight < 0.0:
        raise ValueError("distillation_weight must be non-negative")
    teacher_station = batch.get("teacher_station_factor_mean")
    teacher_cluster = batch.get("teacher_cluster_factor_mean")
    if distillation_weight == 0.0:
        distillation = cluster.detach() * 0.0
    else:
        if teacher_station is None or teacher_cluster is None:
            raise ValueError("Positive distillation_weight requires teacher targets")
        if station_prediction is None:
            raise ValueError("Station teacher targets require station predictions")
        uncertainty_aware = bool(
            settings.get("distillation_uncertainty_aware", False)
        )
        if uncertainty_aware:
            teacher_station_std = batch.get("teacher_station_factor_std")
            teacher_cluster_std = batch.get("teacher_cluster_factor_std")
            if teacher_station_std is None or teacher_cluster_std is None:
                raise ValueError("Uncertainty-aware KD requires teacher dispersion")
            station_scale = float(settings.get("distillation_station_scale", 0.0))
            cluster_scale = float(settings.get("distillation_cluster_scale", 0.0))
            if station_scale <= 0.0 or cluster_scale <= 0.0:
                raise ValueError("Uncertainty-aware KD scales must be positive")
            station_kd = _weighted_smooth_l1(
                station_view(station_prediction),
                station_view(teacher_station),
                1.0 / (1.0 + station_view(teacher_station_std) / station_scale),
            )
            cluster_kd = _weighted_smooth_l1(
                output["cluster_factor"],
                teacher_cluster,
                1.0 / (1.0 + teacher_cluster_std / cluster_scale),
            )
        else:
            station_kd = _smooth_l1(
                station_view(station_prediction), station_view(teacher_station)
            )
            cluster_kd = _smooth_l1(output["cluster_factor"], teacher_cluster)
        distillation_station_weight = float(
            settings.get("distillation_station_loss_weight", station_weight)
        )
        distillation_cluster_weight = float(
            settings.get(
                "distillation_cluster_loss_weight",
                float(settings["cluster_loss_weight"]),
            )
        )
        if distillation_station_weight < 0.0 or distillation_cluster_weight < 0.0:
            raise ValueError("Distillation target weights must be non-negative")
        if distillation_station_weight == 0.0 and distillation_cluster_weight == 0.0:
            raise ValueError(
                "Positive distillation_weight requires at least one KD target"
            )
        distillation = (
            distillation_station_weight * station_kd
            + distillation_cluster_weight * cluster_kd
        )
    loss = (
        supervised
        + distillation_weight * distillation
        + auxiliary_weight * auxiliary
    )
    baseline_weight = float(settings.get("baseline_auxiliary_loss_weight", 0.0))
    baseline_station = output.get("baseline_station_factor")
    baseline_cluster = output.get("baseline_cluster_factor")
    if baseline_station is None or baseline_cluster is None:
        if baseline_weight != 0.0:
            raise ValueError(
                "baseline_auxiliary_loss_weight requires explicit baseline predictions"
            )
        baseline = cluster.detach() * 0.0
    else:
        baseline = (
            station_weight
            * _smooth_l1(
                station_view(baseline_station), station_view(batch["station_future"])
            )
            + float(settings["cluster_loss_weight"])
            * _smooth_l1(baseline_cluster, batch["cluster_future"])
        )
    loss = loss + baseline_weight * baseline
    anchor_weight = float(settings.get("baseline_anchor_loss_weight", 0.0))
    if anchor_weight < 0.0:
        raise ValueError("baseline_anchor_loss_weight must be non-negative")
    anchor_station = output.get("anchor_baseline_station_factor")
    anchor_cluster = output.get("anchor_baseline_cluster_factor")
    if anchor_station is None or anchor_cluster is None:
        if anchor_weight != 0.0:
            raise ValueError(
                "baseline_anchor_loss_weight requires pretrained anchor predictions"
            )
        baseline_anchor = cluster.detach() * 0.0
    else:
        if baseline_station is None or baseline_cluster is None:
            raise ValueError("Pretrained anchor requires explicit baseline predictions")
        anchor_schedule_mode = str(settings.get("anchor_schedule_mode", "scalar"))
        baseline_anchor = (
            station_weight
            * _horizon_anchor_smooth_l1(
                station_view(baseline_station),
                station_view(anchor_station),
                anchor_schedule_mode,
            )
            + float(settings["cluster_loss_weight"])
            * _horizon_anchor_smooth_l1(
                baseline_cluster, anchor_cluster, anchor_schedule_mode
            )
        )
    loss = loss + anchor_weight * baseline_anchor
    ramp_weight = float(settings.get("ramp_loss_weight", 0.0))
    if ramp_weight < 0.0:
        raise ValueError("ramp_loss_weight must be non-negative")
    if ramp_weight == 0.0:
        ramp = cluster.detach() * 0.0
    else:
        cluster_path = torch.cat(
            (batch["cluster_history"][:, -1:], output["cluster_factor"]), dim=-1
        )
        cluster_target_path = torch.cat(
            (batch["cluster_history"][:, -1:], batch["cluster_future"]), dim=-1
        )
        cluster_ramp = _smooth_l1(
            cluster_path[:, 1:] - cluster_path[:, :-1],
            cluster_target_path[:, 1:] - cluster_target_path[:, :-1],
        )
        if station_prediction is None:
            station_ramp = cluster.detach() * 0.0
        else:
            station_path = torch.cat(
                (
                    station_view(batch["station_history"][:, :, -1:]),
                    station_view(station_prediction),
                ),
                dim=-1,
            )
            station_target_path = torch.cat(
                (
                    station_view(batch["station_history"][:, :, -1:]),
                    station_view(batch["station_future"]),
                ),
                dim=-1,
            )
            station_ramp = _smooth_l1(
                station_path[:, :, 1:] - station_path[:, :, :-1],
                station_target_path[:, :, 1:] - station_target_path[:, :, :-1],
            )
        ramp = station_weight * station_ramp + float(
            settings["cluster_loss_weight"]
        ) * cluster_ramp
    loss = loss + ramp_weight * ramp
    multirate_target_weight = float(
        settings.get("multirate_target_loss_weight", 0.0)
    )
    multirate_consistency_weight = float(
        settings.get("multirate_consistency_loss_weight", 0.0)
    )
    multirate_station = output.get("multirate_station_factor")
    multirate_cluster = output.get("multirate_cluster_factor")
    if multirate_station is None or multirate_cluster is None:
        if multirate_target_weight != 0.0 or multirate_consistency_weight != 0.0:
            raise ValueError(
                "multirate loss weights require multirate model predictions"
            )
        multirate_target = cluster.detach() * 0.0
        multirate_consistency = cluster.detach() * 0.0
    else:
        station_hourly_target = batch["station_future"][:, :, 3::4]
        cluster_hourly_target = batch["cluster_future"][:, 3::4]
        multirate_target = (
            station_weight
            * _smooth_l1(
                station_view(multirate_station), station_view(station_hourly_target)
            )
            + float(settings["cluster_loss_weight"])
            * _smooth_l1(multirate_cluster, cluster_hourly_target)
        )
        if station_prediction is None:
            raise ValueError("multirate consistency requires station predictions")
        multirate_consistency = (
            station_weight
            * _smooth_l1(
                station_view(multirate_station),
                station_view(station_prediction[:, :, 3::4]),
            )
            + float(settings["cluster_loss_weight"])
            * _smooth_l1(multirate_cluster, output["cluster_factor"][:, 3::4])
        )
        loss = (
            loss
            + multirate_target_weight * multirate_target
            + multirate_consistency_weight * multirate_consistency
        )
    identification_weight = float(settings.get("response_identification_loss_weight", 0.0))
    identification_prediction = output.get("response_identification_station_factor")
    if identification_prediction is None:
        if identification_weight != 0.0:
            raise ValueError(
                "response_identification_loss_weight requires historical query predictions"
            )
        identification = cluster.detach() * 0.0
    else:
        if "past_station_power" not in batch:
            raise ValueError("Historical query predictions require past_station_power")
        identification = _smooth_l1(
            station_view(identification_prediction),
            station_view(batch["past_station_power"][:, :, 3:]),
        )
        loss = loss + identification_weight * identification
    return loss, {
        "station": float(station.detach()),
        "cluster": float(cluster.detach()),
        "auxiliary": float(auxiliary.detach()),
        "distillation": float(distillation.detach()),
        "baseline": float(baseline.detach()),
        "baseline_anchor": float(baseline_anchor.detach()),
        "ramp": float(ramp.detach()),
        "multirate_target": float(multirate_target.detach()),
        "multirate_consistency": float(multirate_consistency.detach()),
        "response_identification": float(identification.detach()),
    }


class MetricAccumulator:
    def __init__(
        self,
        forecast_steps: int,
        total_capacity_mw: float,
        metric_station_indices: Sequence[int] | None = None,
    ) -> None:
        self.forecast_steps = forecast_steps
        self.total_capacity_mw = float(total_capacity_mw)
        self.samples = 0
        self.cluster_abs_sum = 0.0
        self.cluster_sq_sum = 0.0
        self.station_abs_sum = 0.0
        self.station_values = 0
        self.station_cf_abs_sum = 0.0
        self.station_cf_values = 0
        self.per_station_cf_abs_sum: np.ndarray | None = None
        self.per_station_mw_abs_sum: np.ndarray | None = None
        self.per_station_values = 0
        self.lead_abs_sum = np.zeros(forecast_steps, dtype=np.float64)
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.persistence_abs_sum = 0.0
        self.energy_abs_sum = 0.0
        self.cluster_ramp_abs_sum = 0.0
        self.metric_station_indices = (
            None
            if metric_station_indices is None
            else tuple(int(value) for value in metric_station_indices)
        )
        if self.metric_station_indices is not None and (
            not self.metric_station_indices
            or len(set(self.metric_station_indices)) != len(self.metric_station_indices)
            or min(self.metric_station_indices) < 0
        ):
            raise ValueError(
                "metric_station_indices must be non-empty, unique, and non-negative"
            )

    def update(
        self,
        station_prediction: torch.Tensor | None,
        cluster_prediction: torch.Tensor,
        station_target: torch.Tensor,
        cluster_target: torch.Tensor,
        capacity_mw: torch.Tensor,
        cluster_history_last: torch.Tensor,
    ) -> None:
        cluster_error_mw = (
            cluster_prediction.float() - cluster_target.float()
        ) * self.total_capacity_mw
        cluster_target_mw = cluster_target.float() * self.total_capacity_mw
        persistence_error_mw = (
            cluster_history_last.float().unsqueeze(-1) - cluster_target.float()
        ) * self.total_capacity_mw
        self.samples += int(cluster_error_mw.shape[0])
        self.cluster_abs_sum += float(cluster_error_mw.abs().sum())
        self.cluster_sq_sum += float(cluster_error_mw.square().sum())
        self.target_sum += float(cluster_target_mw.sum())
        self.target_sq_sum += float(cluster_target_mw.square().sum())
        self.persistence_abs_sum += float(persistence_error_mw.abs().sum())
        self.energy_abs_sum += float((cluster_error_mw.sum(dim=1) * 0.25).abs().sum())
        prediction_path = torch.cat(
            (cluster_history_last.float().unsqueeze(-1), cluster_prediction.float()),
            dim=-1,
        )
        target_path = torch.cat(
            (cluster_history_last.float().unsqueeze(-1), cluster_target.float()),
            dim=-1,
        )
        ramp_error_mw = (
            (prediction_path[:, 1:] - prediction_path[:, :-1])
            - (target_path[:, 1:] - target_path[:, :-1])
        ) * self.total_capacity_mw
        self.cluster_ramp_abs_sum += float(ramp_error_mw.abs().sum())
        if station_prediction is not None:
            if self.metric_station_indices is not None:
                if max(self.metric_station_indices) >= station_prediction.shape[1]:
                    raise ValueError("metric_station_indices exceed the station dimension")
                selected = torch.as_tensor(
                    self.metric_station_indices,
                    dtype=torch.long,
                    device=station_prediction.device,
                )
                station_prediction = station_prediction.index_select(1, selected)
                station_target = station_target.index_select(1, selected)
                capacity_mw = capacity_mw.index_select(0, selected)
            station_error_cf = station_prediction.float() - station_target.float()
            station_error_mw = station_error_cf * capacity_mw.reshape(1, -1, 1)
            self.station_abs_sum += float(station_error_mw.abs().sum())
            self.station_values += station_error_mw.numel()
            self.station_cf_abs_sum += float(station_error_cf.abs().sum())
            self.station_cf_values += station_error_cf.numel()
            cf_by_station = (
                station_error_cf.abs().sum(dim=(0, 2)).detach().cpu().double().numpy()
            )
            mw_by_station = (
                station_error_mw.abs().sum(dim=(0, 2)).detach().cpu().double().numpy()
            )
            if self.per_station_cf_abs_sum is None:
                self.per_station_cf_abs_sum = np.zeros_like(cf_by_station)
                self.per_station_mw_abs_sum = np.zeros_like(mw_by_station)
            self.per_station_cf_abs_sum += cf_by_station
            assert self.per_station_mw_abs_sum is not None
            self.per_station_mw_abs_sum += mw_by_station
            self.per_station_values += int(station_prediction.shape[0] * station_prediction.shape[2])
        self.lead_abs_sum += (
            cluster_error_mw.abs().sum(dim=0).detach().cpu().double().numpy()
        )

    def compute(self) -> dict[str, Any]:
        cluster_values = self.samples * self.forecast_steps
        cluster_mae = self.cluster_abs_sum / cluster_values
        target_mean = self.target_sum / cluster_values
        target_total_sq = self.target_sq_sum - cluster_values * target_mean**2
        r2 = (
            1.0 - self.cluster_sq_sum / target_total_sq
            if target_total_sq > 0.0
            else float("nan")
        )
        lead_mae = self.lead_abs_sum / self.samples
        persistence_mae = self.persistence_abs_sum / cluster_values
        return {
            "cluster_mae_mw": cluster_mae,
            "cluster_rmse_mw": math.sqrt(self.cluster_sq_sum / cluster_values),
            "cluster_nmae": cluster_mae / self.total_capacity_mw,
            "cluster_r2": r2,
            "cluster_energy_mae_mwh": self.energy_abs_sum / self.samples,
            "cluster_ramp_mae_mw": self.cluster_ramp_abs_sum / cluster_values,
            "persistence_mae_mw": persistence_mae,
            "persistence_skill_mae": 1.0 - cluster_mae / persistence_mae,
            "station_mae_mw": (
                self.station_abs_sum / self.station_values
                if self.station_values
                else None
            ),
            "station_mae_cf": (
                self.station_cf_abs_sum / self.station_cf_values
                if self.station_cf_values
                else None
            ),
            "per_station_mae_cf": (
                (self.per_station_cf_abs_sum / self.per_station_values).tolist()
                if self.per_station_cf_abs_sum is not None and self.per_station_values
                else None
            ),
            "per_station_mae_mw": (
                (self.per_station_mw_abs_sum / self.per_station_values).tolist()
                if self.per_station_mw_abs_sum is not None and self.per_station_values
                else None
            ),
            "lead_cluster_mae_mw": lead_mae.tolist(),
            "horizon_0_1h_mae_mw": float(lead_mae[:4].mean()),
            "horizon_1_3h_mae_mw": float(lead_mae[4:12].mean()),
            "horizon_3_6h_mae_mw": float(lead_mae[12:24].mean()),
            "samples": self.samples,
        }


def save_prediction_archive(
    path: str | Path, batches: list[dict[str, np.ndarray]]
) -> None:
    if not batches:
        raise ValueError("No prediction batches were collected")
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches):
        raise ValueError("Prediction batches have inconsistent fields")
    values = {key: np.concatenate([batch[key] for batch in batches]) for key in keys}
    order = np.argsort(values["origin_unix"], kind="stable")
    values = {key: value[order] for key, value in values.items()}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, output)


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, output)


def save_json(path: str | Path, values: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(values, stream, ensure_ascii=True, indent=2)
    os.replace(temporary, output)


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update a complete inference model without modifying the training model."""
    if not 0.0 < decay < 1.0:
        raise ValueError("EMA decay must be strictly between zero and one")
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    if ema_state.keys() != model_state.keys():
        raise ValueError("EMA and training model states do not match")
    for key, ema_value in ema_state.items():
        value = model_state[key].detach()
        if ema_value.is_floating_point() or ema_value.is_complex():
            ema_value.lerp_(value, 1.0 - decay)
        else:
            ema_value.copy_(value)


def detached_cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def average_model_states(
    states: Iterable[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average floating checkpoint tensors and retain the latest discrete state."""
    materialized = list(states)
    if not materialized:
        raise ValueError("At least one model state is required for averaging")
    keys = materialized[0].keys()
    if any(state.keys() != keys for state in materialized[1:]):
        raise ValueError("Model states have inconsistent keys")
    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [state[key].detach().cpu() for state in materialized]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"Model-state tensor shape mismatch for {key}")
        if values[0].is_floating_point() or values[0].is_complex():
            accumulator = values[0].to(dtype=torch.float64)
            for value in values[1:]:
                accumulator = accumulator + value.to(dtype=torch.float64)
            averaged[key] = (accumulator / len(values)).to(dtype=values[0].dtype)
        else:
            averaged[key] = values[-1].clone()
    return averaged


def _limited_batches(
    loader: Iterable[dict[str, torch.Tensor]], max_batches: int | None
):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def _batch_count(loader, max_batches: int | None) -> int:
    total = len(loader)
    return total if max_batches is None else min(total, max_batches)


def _progress(
    loader,
    max_batches: int | None,
    description: str,
):
    prefix = os.environ.get("WIND_PROGRESS_PREFIX", "").strip()
    display_description = f"{prefix} {description}" if prefix else description
    position_value = os.environ.get("WIND_PROGRESS_POSITION", "").strip()
    position = int(position_value) if position_value else None
    leave_value = os.environ.get("WIND_PROGRESS_LEAVE", "1").strip().lower()
    leave = leave_value not in {"0", "false", "no"}
    return tqdm(
        _limited_batches(loader, max_batches),
        total=_batch_count(loader, max_batches),
        desc=display_description,
        unit="batch",
        dynamic_ncols=True,
        leave=leave,
        position=position,
        mininterval=0.5,
    )


@torch.no_grad()
def evaluate_spatial(
    model: nn.Module,
    loader,
    device: torch.device,
    precision: str,
    settings: Mapping[str, Any],
    max_batches: int | None = None,
    progress_desc: str = "Spatial validation",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cycles = 0
    station_abs = 0.0
    cluster_abs = 0.0
    progress = _progress(loader, max_batches, progress_desc)
    for batch in progress:
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            output = model(batch["weather"])
            loss, _ = spatial_pretrain_loss(
                output, batch, model.capacity_mw, settings
            )
        cycles = int(batch["weather"].shape[0])
        target = batch["station_target"]
        cluster_target = (
            target * model.capacity_mw.reshape(1, -1, 1)
        ).sum(dim=1) / model.total_capacity_mw
        station_abs += float(
            ((output["hourly_station_factor"] - target).abs() * model.capacity_mw.reshape(1, -1, 1)).sum()
        )
        cluster_abs += float(
            (
                (output["cluster_factor"] - cluster_target).abs()
                * model.total_capacity_mw
            ).sum()
        )
        total_loss += float(loss) * cycles
        total_cycles += cycles
        progress.set_postfix(loss=f"{total_loss / total_cycles:.5f}")
    if total_cycles == 0:
        raise ValueError("Spatial validation loader produced no batches")
    station_values = total_cycles * model.capacity_mw.numel() * batch["station_target"].shape[-1]
    cluster_values = total_cycles * batch["station_target"].shape[-1]
    return {
        "loss": total_loss / total_cycles,
        "station_mae_mw": station_abs / station_values,
        "cluster_mae_mw": cluster_abs / cluster_values,
    }


def fit_spatial_pretrain(
    model: nn.Module,
    train_loader,
    val_loader,
    settings: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    *,
    precision: str = "bf16",
    epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, list[dict[str, float]]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epoch_count = int(settings["epochs"] if epochs is None else epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_count)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision.lower() == "fp16"
    )
    primary_metric = str(settings.get("checkpoint_metric", "cluster_mae_mw"))
    if primary_metric not in {"cluster_mae_mw", "station_mae_cf", "station_mae_mw"}:
        raise ValueError(f"Unsupported deterministic checkpoint metric: {primary_metric}")
    best = float("inf")
    stale = 0
    patience = int(settings.get("early_stopping_patience", 12))
    history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
    for epoch in range(epoch_count):
        model.train()
        started = time.perf_counter()
        total = 0.0
        cycles = 0
        progress = _progress(
            train_loader,
            max_train_batches,
            f"Spatial train {epoch + 1}/{epoch_count}",
        )
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                output = model(batch["weather"])
                loss, _ = spatial_pretrain_loss(
                    output, batch, model.capacity_mw, settings
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), float(settings.get("gradient_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            size = int(batch["weather"].shape[0])
            total += float(loss.detach()) * size
            cycles += size
            progress.set_postfix(
                loss=f"{total / cycles:.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        if cycles == 0:
            raise ValueError("Spatial training loader produced no batches")
        scheduler.step()
        validation = evaluate_spatial(
            model,
            val_loader,
            device,
            precision,
            settings,
            max_val_batches,
            progress_desc=f"Spatial val   {epoch + 1}/{epoch_count}",
        )
        train_record = {"loss": total / cycles, "seconds": time.perf_counter() - started}
        history["train"].append(train_record)
        history["val"].append(validation)
        improved = float(validation[primary_metric]) < best
        if improved:
            best = float(validation[primary_metric])
            stale = 0
            save_checkpoint(
                checkpoint_path,
                {
                    **checkpoint_metadata,
                    "model_state": model.state_dict(),
                    "forcing_encoder_state": model.forcing_encoder.state_dict(),
                    "epoch": epoch + 1,
                    "val_metrics": validation,
                    "checkpoint_metric": primary_metric,
                },
            )
        else:
            stale += 1
        station_mae = validation["station_mae_mw"]
        station_display = (
            f"{station_mae:.2f}MW" if station_mae is not None else "n/a"
        )
        print(
            f"[spatial:{epoch + 1:03d}/{epoch_count:03d}] "
            f"train={train_record['loss']:.6f} "
            f"val={validation['loss']:.6f} "
            f"station_mae={station_display} "
            f"cluster_mae={validation['cluster_mae_mw']:.2f}MW "
            f"best_{primary_metric}={best:.6f} patience={stale}/{patience} "
            f"checkpoint={'saved' if improved else 'unchanged'} "
            f"seconds={train_record['seconds']:.1f}",
            flush=True,
        )
        if stale >= patience:
            print(
                f"[early-stop] spatial epoch={epoch + 1}, "
                f"best_{primary_metric}={best:.6f}, patience={patience}",
                flush=True,
            )
            break
    return history


@torch.no_grad()
def evaluate_deterministic(
    model: nn.Module,
    loader,
    device: torch.device,
    precision: str,
    settings: Mapping[str, Any],
    max_batches: int | None = None,
    progress_desc: str = "Forecast validation",
    prediction_batches: list[dict[str, np.ndarray]] | None = None,
    export_station_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator(
        model.forecast_steps,
        float(model.total_capacity_mw),
        settings.get("metric_station_indices"),
    )
    total_loss = 0.0
    samples = 0
    inference_seconds = 0.0
    progress = _progress(loader, max_batches, progress_desc)
    for batch in progress:
        batch = move_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        with autocast_context(device, precision):
            output = model(**model_inputs(batch))
            loss, _ = deterministic_loss(output, batch, settings)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - inference_started
        size = int(batch["station_history"].shape[0])
        total_loss += float(loss) * size
        samples += size
        accumulator.update(
            output["station_factor"],
            output["cluster_factor"],
            batch["station_future"],
            batch["cluster_future"],
            model.capacity_mw,
            batch["cluster_history"][:, -1],
        )
        if prediction_batches is not None:
            total_capacity = float(model.total_capacity_mw)
            record = {
                "origin_unix": batch["origin_unix"].detach().cpu().numpy(),
                "sample_index": batch["sample_index"].detach().cpu().numpy(),
                "cluster_prediction_mw": (
                    output["cluster_factor"].detach().float().cpu().numpy()
                    * total_capacity
                ),
                "cluster_target_mw": (
                    batch["cluster_future"].detach().float().cpu().numpy()
                    * total_capacity
                ),
                "cluster_history_last_mw": (
                    batch["cluster_history"][:, -1].detach().float().cpu().numpy()
                    * total_capacity
                ),
            }
            station_prediction = output.get("station_factor")
            baseline_cluster = output.get("baseline_cluster_factor")
            if baseline_cluster is not None:
                record["baseline_cluster_prediction_mw"] = (
                    baseline_cluster.detach().float().cpu().numpy()
                    * total_capacity
                )
                record["realized_cluster_correction_mw"] = (
                    (
                        output["cluster_factor"].detach().float()
                        - baseline_cluster.detach().float()
                    )
                    .cpu()
                    .numpy()
                    * total_capacity
                )
            component_keys = {
                "weather_correction_factor": "cluster_raw_correction_mw",
                "near_specialist_correction_factor": (
                    "cluster_near_specialist_correction_mw"
                ),
                "full_specialist_correction_factor": (
                    "cluster_full_specialist_correction_mw"
                ),
                "weather_main_correction_factor": "cluster_weather_main_correction_mw",
                "state_main_correction_factor": "cluster_state_main_correction_mw",
                "state_weather_interaction_correction_factor": (
                    "cluster_state_weather_interaction_correction_mw"
                ),
            }
            for output_key, record_key in component_keys.items():
                component = output.get(output_key)
                if component is not None:
                    capacity = model.capacity_mw.reshape(1, -1, 1)
                    record[record_key] = (
                        component.detach().float().cpu() * capacity.cpu()
                    ).sum(dim=1).numpy()
            diagnostic_keys = {
                "interaction_update_abs_mean": "interaction_update_abs_mean",
                "weather_attention_shift_l1": "weather_attention_shift_l1",
                "weather_view_gate": "weather_view_gate",
                "baseline_anchor_abs_drift_factor": (
                    "baseline_anchor_abs_drift_factor"
                ),
            }
            for output_key, record_key in diagnostic_keys.items():
                diagnostic = output.get(output_key)
                if diagnostic is not None:
                    record[record_key] = (
                        diagnostic.detach().float().cpu().numpy()
                    )
            history_attention = output.get("history_response_attention")
            if history_attention is not None:
                # Persist one compact lead-by-history matrix per sample instead
                # of the prohibitively large station/head-resolved tensor.
                record["history_response_attention_mean"] = (
                    history_attention.detach()
                    .float()
                    .mean(dim=(1, 2), keepdim=False)
                    .cpu()
                    .numpy()
                )
            anchor_cluster = output.get("anchor_baseline_cluster_factor")
            if anchor_cluster is not None:
                record["anchor_baseline_cluster_prediction_mw"] = (
                    anchor_cluster.detach().float().cpu().numpy()
                    * total_capacity
                )
            for output_key, record_key in (
                ("near_specialist_cluster_factor", "near_specialist_prediction_mw"),
                ("full_specialist_cluster_factor", "full_specialist_prediction_mw"),
            ):
                specialist = output.get(output_key)
                if specialist is not None:
                    record[record_key] = (
                        specialist.detach().float().cpu().numpy() * total_capacity
                    )
            if export_station_predictions and station_prediction is not None:
                capacity = model.capacity_mw.reshape(1, -1, 1)
                record["station_prediction_mw"] = (
                    station_prediction.detach().float().cpu() * capacity.cpu()
                ).numpy()
                record["station_target_mw"] = (
                    batch["station_future"].detach().float().cpu() * capacity.cpu()
                ).numpy()
            prediction_batches.append(record)
        progress.set_postfix(loss=f"{total_loss / samples:.5f}")
    if samples == 0:
        raise ValueError("Deterministic evaluation loader produced no batches")
    return {
        "loss": total_loss / samples,
        **accumulator.compute(),
        "inference_seconds": inference_seconds,
        "samples_per_second": samples / max(inference_seconds, 1e-9),
    }


def fit_deterministic(
    model: nn.Module,
    train_loader,
    val_loader,
    settings: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    *,
    precision: str = "bf16",
    epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    model.to(device)
    learning_rate = float(settings["learning_rate"])
    weight_decay = float(settings["weight_decay"])
    parameter_group_builder = getattr(model, "optimizer_parameter_groups", None)
    parameters = (
        parameter_group_builder(learning_rate, weight_decay)
        if callable(parameter_group_builder)
        else model.parameters()
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    epoch_count = int(settings["epochs"] if epochs is None else epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_count)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision.lower() == "fp16"
    )
    primary_metric = str(settings.get("checkpoint_metric", "cluster_mae_mw"))
    if primary_metric not in {"cluster_mae_mw", "station_mae_cf", "station_mae_mw"}:
        raise ValueError(f"Unsupported deterministic checkpoint metric: {primary_metric}")
    best = float("inf")
    stale = 0
    patience = int(settings.get("early_stopping_patience", 12))
    ema_decay = float(settings.get("ema_decay", 0.0))
    if ema_decay != 0.0 and not 0.0 < ema_decay < 1.0:
        raise ValueError("ema_decay must be zero or strictly between zero and one")
    ema_model = None
    if ema_decay > 0.0:
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    checkpoint_average_window = int(settings.get("checkpoint_average_window", 0))
    if checkpoint_average_window < 0:
        raise ValueError("checkpoint_average_window must be non-negative")
    recent_states: deque[tuple[int, dict[str, torch.Tensor]]] = deque(
        maxlen=max(1, checkpoint_average_window)
    )
    history: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for epoch in range(epoch_count):
        sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        started = time.perf_counter()
        total = 0.0
        samples = 0
        progress = _progress(
            train_loader,
            max_train_batches,
            f"Forecast train {epoch + 1}/{epoch_count}",
        )
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                output = model(**model_inputs(batch))
                loss, _ = deterministic_loss(output, batch, settings)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), float(settings.get("gradient_clip", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                update_ema_model(ema_model, model, ema_decay)
            size = int(batch["station_history"].shape[0])
            total += float(loss.detach()) * size
            samples += size
            progress.set_postfix(
                loss=f"{total / samples:.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        if samples == 0:
            raise ValueError("Forecast training loader produced no batches")
        scheduler.step()
        validation_settings = dict(settings)
        validation_settings["distillation_weight"] = 0.0
        validation = evaluate_deterministic(
            model,
            val_loader,
            device,
            precision,
            validation_settings,
            max_val_batches,
            progress_desc=f"Forecast val   {epoch + 1}/{epoch_count}",
        )
        train_record = {"loss": total / samples, "seconds": time.perf_counter() - started}
        history["train"].append(train_record)
        history["val"].append(validation)
        if checkpoint_average_window > 0:
            recent_states.append((epoch + 1, detached_cpu_state(model)))
        improved = float(validation[primary_metric]) < best
        if improved:
            best = float(validation[primary_metric])
            stale = 0
            checkpoint_payload: dict[str, Any] = {
                    **checkpoint_metadata,
                    "model_state": model.state_dict(),
                    "forcing_encoder_state": (
                        model.forcing_encoder.state_dict()
                        if hasattr(model, "forcing_encoder")
                        else {}
                    ),
                    "epoch": epoch + 1,
                    "val_metrics": validation,
                    "checkpoint_metric": primary_metric,
                }
            if ema_model is not None:
                checkpoint_payload["ema_model_state"] = ema_model.state_dict()
                checkpoint_payload["ema_decay"] = ema_decay
            if checkpoint_average_window > 0:
                checkpoint_payload["checkpoint_average_model_state"] = (
                    average_model_states(state for _, state in recent_states)
                )
                checkpoint_payload["checkpoint_average_epochs"] = [
                    state_epoch for state_epoch, _ in recent_states
                ]
                checkpoint_payload["checkpoint_average_window"] = (
                    checkpoint_average_window
                )
            save_checkpoint(checkpoint_path, checkpoint_payload)
        else:
            stale += 1
        station_mae = validation["station_mae_mw"]
        station_display = (
            f"{station_mae:.2f}MW" if station_mae is not None else "n/a"
        )
        print(
            f"[forecast:{epoch + 1:03d}/{epoch_count:03d}] "
            f"train={train_record['loss']:.6f} "
            f"val={validation['loss']:.6f} "
            f"station_mae={station_display} "
            f"cluster_mae={validation['cluster_mae_mw']:.2f}MW "
            f"cluster_rmse={validation['cluster_rmse_mw']:.2f}MW "
            f"best_{primary_metric}={best:.6f} patience={stale}/{patience} "
            f"checkpoint={'saved' if improved else 'unchanged'} "
            f"seconds={train_record['seconds']:.1f}",
            flush=True,
        )
        if stale >= patience:
            print(
                f"[early-stop] forecast epoch={epoch + 1}, "
                f"best_{primary_metric}={best:.6f}, patience={patience}",
                flush=True,
            )
            break
    return history
