from __future__ import annotations

import json
import itertools
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from wind_forcing.training import autocast_context, move_batch, save_checkpoint


def aggregate_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = (
        "cluster_history",
        "weather",
        "future_hourly_weather_latent",
        "weather_batch_index",
        "weather_time_offset_hours",
        "past_hourly_weather_latent",
        "past_cluster_power",
        "past_weather_time_offset_hours",
        "current_weather_latent",
        "current_weather_time_offset_hours",
        "past_realization_factor",
        "origin_phase",
    )
    return {key: batch[key] for key in keys if key in batch}


def aggregate_loss(
    output: Mapping[str, torch.Tensor | None],
    batch: Mapping[str, torch.Tensor],
    settings: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = output["cluster_factor"]
    assert prediction is not None
    quantile = float(settings.get("weather_quantile", 0.0))
    if quantile:
        if not 0.0 < quantile < 1.0:
            raise ValueError("weather_quantile must lie strictly between zero and one")
        residual = batch["cluster_future"].float() - prediction.float()
        supervised = torch.maximum(quantile * residual, (quantile - 1.0) * residual).mean()
    else:
        supervised = F.smooth_l1_loss(
            prediction.float(), batch["cluster_future"].float(), beta=0.05
        )
    current_weight = float(settings.get("current_weather_loss_weight", 0.0))
    current_prediction = output.get("current_weather_prediction_factor")
    if current_weight:
        if current_prediction is None:
            raise ValueError("Current-weather loss requires a current prediction")
        current_target = batch["cluster_history"][:, -1:].float()
        if quantile:
            current_residual = current_target - current_prediction.float()
            current_loss = torch.maximum(
                quantile * current_residual, (quantile - 1.0) * current_residual
            ).mean()
        else:
            current_loss = F.smooth_l1_loss(
                current_prediction.float(), current_target, beta=0.05
            )
    else:
        current_loss = supervised.detach() * 0.0
    anchor_weight = float(settings.get("baseline_anchor_loss_weight", 0.0))
    baseline = output.get("baseline_cluster_factor")
    anchor = output.get("anchor_baseline_cluster_factor")
    if anchor_weight:
        if baseline is None or anchor is None:
            raise ValueError("Anchor loss requires baseline and anchor predictions")
        values = F.smooth_l1_loss(baseline.float(), anchor.float(), beta=0.05, reduction="none")
        steps = values.shape[-1]
        lead = torch.arange(1, steps + 1, device=values.device, dtype=values.dtype) / 4.0
        if str(settings.get("anchor_schedule_mode", "horizon_taper")) == "horizon_taper":
            weights = torch.where(lead <= 3.0, 1.0, (4.0 - lead).clamp(0.0, 1.0))
        else:
            weights = torch.ones_like(lead)
        anchor_value = (values * weights.reshape(1, -1)).mean()
    else:
        anchor_value = supervised.detach() * 0.0
    ramp_weight = float(settings.get("ramp_loss_weight", 0.0))
    if ramp_weight:
        prediction_path = torch.cat((batch["cluster_history"][:, -1:], prediction), dim=1)
        target_path = torch.cat((batch["cluster_history"][:, -1:], batch["cluster_future"]), dim=1)
        ramp = F.smooth_l1_loss(
            prediction_path[:, 1:] - prediction_path[:, :-1],
            target_path[:, 1:] - target_path[:, :-1],
            beta=0.05,
        )
    else:
        ramp = supervised.detach() * 0.0
    identification_weight = float(settings.get("response_identification_loss_weight", 0.0))
    if identification_weight:
        identification_prediction = output.get("identification_prediction")
        identification_target = output.get("identification_target")
        if identification_prediction is None or identification_target is None:
            raise ValueError("Response-identification loss requires historical query outputs")
        identification = F.smooth_l1_loss(
            identification_prediction.float(),
            identification_target.float(),
            beta=0.05,
        )
    else:
        identification = supervised.detach() * 0.0
    loss = (
        float(settings.get("cluster_loss_weight", 1.0)) * supervised
        + anchor_weight * anchor_value
        + ramp_weight * ramp
        + identification_weight * identification
        + current_weight * current_loss
    )
    return loss, {
        "cluster": float(supervised.detach()),
        "anchor": float(anchor_value.detach()),
        "ramp": float(ramp.detach()),
        "identification": float(identification.detach()),
        "current_weather": float(current_loss.detach()),
    }


class AggregateMetrics:
    def __init__(self, forecast_steps: int, total_capacity_mw: float) -> None:
        self.steps = forecast_steps
        self.capacity = float(total_capacity_mw)
        self.samples = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.persistence_abs_sum = 0.0
        self.energy_abs_sum = 0.0
        self.ramp_abs_sum = 0.0
        self.lead_abs = np.zeros(forecast_steps, dtype=np.float64)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        history_last: torch.Tensor,
    ) -> None:
        error = (prediction.float() - target.float()) * self.capacity
        target_mw = target.float() * self.capacity
        persistence = (history_last[:, None].float() - target.float()) * self.capacity
        self.samples += len(prediction)
        self.abs_sum += float(error.abs().sum())
        self.sq_sum += float(error.square().sum())
        self.target_sum += float(target_mw.sum())
        self.target_sq_sum += float(target_mw.square().sum())
        self.persistence_abs_sum += float(persistence.abs().sum())
        self.energy_abs_sum += float((error.sum(dim=1) * 0.25).abs().sum())
        prediction_path = torch.cat((history_last[:, None], prediction), dim=1)
        target_path = torch.cat((history_last[:, None], target), dim=1)
        ramp_error = (
            prediction_path[:, 1:] - prediction_path[:, :-1]
            - target_path[:, 1:] + target_path[:, :-1]
        ) * self.capacity
        self.ramp_abs_sum += float(ramp_error.abs().sum())
        self.lead_abs += error.abs().sum(dim=0).cpu().double().numpy()

    def compute(self) -> dict[str, Any]:
        count = self.samples * self.steps
        mae = self.abs_sum / count
        target_mean = self.target_sum / count
        target_total_sq = self.target_sq_sum - count * target_mean**2
        persistence = self.persistence_abs_sum / count
        lead = self.lead_abs / self.samples
        return {
            "cluster_mae_mw": mae,
            "cluster_rmse_mw": math.sqrt(self.sq_sum / count),
            "cluster_nmae": mae / self.capacity,
            "cluster_r2": 1.0 - self.sq_sum / target_total_sq,
            "cluster_energy_mae_mwh": self.energy_abs_sum / self.samples,
            "cluster_ramp_mae_mw": self.ramp_abs_sum / count,
            "persistence_mae_mw": persistence,
            "persistence_skill_mae": 1.0 - mae / persistence,
            "lead_cluster_mae_mw": lead.tolist(),
            "horizon_0_1h_mae_mw": float(lead[:4].mean()),
            "horizon_1_3h_mae_mw": float(lead[4:12].mean()),
            "horizon_3_6h_mae_mw": float(lead[12:].mean()),
            "samples": self.samples,
        }


def _progress(loader, description: str, max_batches: int | None = None):
    prefix = os.environ.get("WIND_PROGRESS_PREFIX", "").strip()
    position = os.environ.get("WIND_PROGRESS_POSITION", "").strip()
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    iterator = loader if max_batches is None else itertools.islice(loader, max_batches)
    return tqdm(
        iterator,
        total=total,
        desc=f"{prefix} {description}".strip(),
        unit="batch",
        dynamic_ncols=True,
        position=int(position) if position else None,
        leave=True,
        mininterval=0.5,
    )


@torch.no_grad()
def evaluate_aggregate(
    model: nn.Module,
    loader,
    device: torch.device,
    precision: str,
    settings: Mapping[str, Any],
    *,
    max_batches: int | None = None,
    description: str = "Aggregate val",
    collect_predictions: bool = False,
    collect_diagnostics: bool = False,
    correction_state_intervention: str = "none",
    correction_state_lookup: Mapping[int, np.ndarray] | None = None,
    response_intervention: str = "none",
    response_power_lookup: Mapping[int, np.ndarray] | None = None,
    response_weather_lookup: Mapping[int, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    if correction_state_intervention not in {"none", "cyclic_shift", "global_permutation", "reference"}:
        raise ValueError(
            "correction_state_intervention must be none, cyclic_shift, global_permutation, or reference"
        )
    if correction_state_intervention == "global_permutation" and correction_state_lookup is None:
        raise ValueError("global_permutation requires a global state-history lookup")
    valid_response = {
        "none", "power_shuffle", "weather_shuffle", "pair_shift", "response_mean", "support_only"
    }
    if response_intervention not in valid_response:
        raise ValueError(f"Unsupported response-state intervention: {response_intervention}")
    if response_intervention == "power_shuffle" and response_power_lookup is None:
        raise ValueError("power_shuffle requires a global historical-power lookup")
    if response_intervention == "weather_shuffle" and response_weather_lookup is None:
        raise ValueError("weather_shuffle requires a global historical-weather lookup")
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = AggregateMetrics(model.forecast_steps, float(model.total_capacity_mw))
    loss_sum = 0.0
    samples = 0
    inference_seconds = 0.0
    state_attention_sum: torch.Tensor | None = None
    state_attention_samples = 0
    identification_abs_sum = 0.0
    identification_count = 0
    records: dict[str, list[np.ndarray]] = {
        "origin_unix": [], "sample_index": [], "cluster_prediction_mw": [],
        "cluster_target_mw": [], "cluster_history_last_mw": [],
    }
    diagnostic_keys = (
        "dynamics_signal", "equilibrium_factor", "response_rate_per_hour",
        "weather_envelope_factor", "realization_factor", "weather_absolute_factor",
        "current_weather_prediction_factor", "current_discrepancy_factor",
        "discrepancy_persistence",
    )
    for batch in _progress(loader, description, max_batches):
        batch = move_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with autocast_context(device, precision):
            model_inputs = aggregate_inputs(batch)
            if correction_state_intervention == "cyclic_shift":
                if len(batch["cluster_history"]) < 2:
                    raise ValueError(
                        "cyclic_shift requires evaluation batches with at least two samples"
                    )
                model_inputs["correction_state_history"] = torch.roll(
                    batch["cluster_history"], shifts=1, dims=0
                )
            elif correction_state_intervention == "global_permutation":
                assert correction_state_lookup is not None
                replacement = np.stack(
                    [correction_state_lookup[int(index)] for index in batch["sample_index"].cpu().tolist()]
                )
                model_inputs["correction_state_history"] = torch.from_numpy(replacement).to(
                    device=device, dtype=batch["cluster_history"].dtype
                )
            if response_intervention == "power_shuffle":
                assert response_power_lookup is not None
                replacement = np.stack(
                    [response_power_lookup[int(index)] for index in batch["sample_index"].cpu().tolist()]
                )
                model_inputs["response_past_power_override"] = torch.from_numpy(replacement).to(
                    device=device, dtype=batch["past_cluster_power"].dtype
                )
            elif response_intervention == "weather_shuffle":
                assert response_weather_lookup is not None
                replacement = np.stack(
                    [response_weather_lookup[int(index)] for index in batch["sample_index"].cpu().tolist()]
                )
                model_inputs["response_past_weather_override"] = torch.from_numpy(replacement).to(
                    device=device, dtype=batch["past_hourly_weather_latent"].dtype
                )
            elif response_intervention == "pair_shift":
                model_inputs["response_pair_shift"] = True
            elif response_intervention == "support_only":
                model_inputs["response_support_only"] = True
            output = model(**model_inputs)
            loss, _ = aggregate_loss(output, batch, settings)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started
        prediction = output["cluster_factor"]
        assert prediction is not None
        state_attention = output.get("state_attention")
        identification_prediction = output.get("identification_prediction")
        identification_target = output.get("identification_target")
        if identification_prediction is not None and identification_target is not None:
            identification_abs_sum += float(
                (identification_prediction.float() - identification_target.float()).abs().sum()
            )
            identification_count += int(identification_target.numel())
        if collect_diagnostics and state_attention is not None:
            detached_attention = state_attention.detach().float().sum(dim=0).cpu()
            state_attention_sum = (
                detached_attention
                if state_attention_sum is None
                else state_attention_sum + detached_attention
            )
            state_attention_samples += len(state_attention)
        size = len(prediction)
        loss_sum += float(loss) * size
        samples += size
        metrics.update(prediction, batch["cluster_future"], batch["cluster_history"][:, -1])
        if collect_predictions:
            capacity = float(model.total_capacity_mw)
            records["origin_unix"].append(batch["origin_unix"].cpu().numpy())
            records["sample_index"].append(batch["sample_index"].cpu().numpy())
            records["cluster_prediction_mw"].append(prediction.float().cpu().numpy() * capacity)
            records["cluster_target_mw"].append(batch["cluster_future"].float().cpu().numpy() * capacity)
            records["cluster_history_last_mw"].append(batch["cluster_history"][:, -1].float().cpu().numpy() * capacity)
            if collect_diagnostics:
                for key in diagnostic_keys:
                    value = output.get(key)
                    if value is not None:
                        records.setdefault(key, []).append(value.detach().float().cpu().numpy())
    if not samples:
        raise ValueError("Aggregate loader yielded no samples")
    result = {
        "loss": loss_sum / samples,
        **metrics.compute(),
        "inference_seconds": inference_seconds,
        "samples_per_second": samples / max(inference_seconds, 1e-9),
        "latency_ms_per_origin": 1000.0 * inference_seconds / samples,
        "peak_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda" else None
        ),
        "correction_state_intervention": correction_state_intervention,
        "response_state_intervention": response_intervention,
    }
    if identification_count:
        result["identification_delta_mae_factor"] = identification_abs_sum / identification_count
        result["identification_delta_mae_mw"] = (
            identification_abs_sum / identification_count * float(model.total_capacity_mw)
        )
    if state_attention_sum is not None and state_attention_samples:
        result["mean_state_attention"] = (
            state_attention_sum / state_attention_samples
        ).numpy().tolist()
    archive = None
    if collect_predictions:
        archive = {key: np.concatenate(value) for key, value in records.items() if value}
        order = np.argsort(archive["origin_unix"], kind="stable")
        archive = {key: value[order] for key, value in archive.items()}
    return result, archive


def fit_aggregate(
    model: nn.Module,
    train_loader,
    val_loader,
    settings: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    *,
    precision: str,
    epochs: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Aggregate training has no trainable parameters")
    learning_rate = float(settings["learning_rate"])
    if hasattr(model, "optimizer_parameter_groups"):
        optimizer_parameters = model.optimizer_parameter_groups(
            learning_rate,
            float(settings.get("decoder_lr_multiplier", 0.1)),
        )
    else:
        optimizer_parameters = parameters
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=learning_rate,
        weight_decay=float(settings["weight_decay"]),
    )
    epoch_count = int(settings["epochs"] if epochs is None else epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_count)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    patience = int(settings.get("early_stopping_patience", 10))
    best = float("inf")
    stale = 0
    history: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for epoch in range(epoch_count):
        sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        samples = 0
        steps = 0
        started = time.perf_counter()
        progress = _progress(train_loader, f"Aggregate train {epoch + 1}/{epoch_count}", max_train_batches)
        for batch_index, batch in enumerate(progress):
            batch = move_batch(batch, device)
            with autocast_context(device, precision):
                output = model(**aggregate_inputs(batch))
                loss, _ = aggregate_loss(output, batch, settings)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            size = len(batch["cluster_history"])
            total += float(loss.detach()) * size
            samples += size
            steps += 1
            final_batch = batch_index + 1 == len(progress)
            if steps % accumulation == 0 or final_batch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, float(settings.get("gradient_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(
                loss=f"{total / samples:.5f}",
                lr="/".join(f"{group['lr']:.1e}" for group in optimizer.param_groups),
            )
        scheduler.step()
        validation, _ = evaluate_aggregate(
            model, val_loader, device, precision, settings,
            max_batches=max_val_batches,
            description=f"Aggregate val {epoch + 1}/{epoch_count}",
        )
        train_record = {"loss": total / samples, "seconds": time.perf_counter() - started}
        history["train"].append(train_record)
        history["val"].append(validation)
        selection_mode = str(settings.get("checkpoint_selection", "validation"))
        if selection_mode not in {"validation", "last"}:
            raise ValueError("checkpoint_selection must be validation or last")
        improved = selection_mode == "last" or float(validation["cluster_mae_mw"]) < best
        if improved:
            best = float(validation["cluster_mae_mw"])
            stale = 0
            save_checkpoint(
                checkpoint_path,
                {
                    **checkpoint_metadata,
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_metrics": validation,
                    "checkpoint_metric": "cluster_mae_mw",
                },
            )
        elif selection_mode == "validation":
            stale += 1
        print(
            f"[aggregate:{epoch + 1:03d}/{epoch_count:03d}] train={train_record['loss']:.6f} "
            f"val_mae={validation['cluster_mae_mw']:.3f}MW val_rmse={validation['cluster_rmse_mw']:.3f}MW "
            f"best={best:.3f}MW patience={stale}/{patience} checkpoint={'saved' if improved else 'unchanged'}",
            flush=True,
        )
        if selection_mode == "validation" and stale >= patience:
            print(f"[early-stop] aggregate epoch={epoch + 1} best={best:.3f}MW", flush=True)
            break
    return history


def save_archive(path: str | Path, archive: Mapping[str, np.ndarray]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **archive)
    os.replace(temporary, output)


def save_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2)
    os.replace(temporary, output)
