"""Single-site model construction and state utilities for the CBH audit.

The learned parameters of every model created here belong to exactly one site
and one seed.  Global static statistics are retained as fixed covariates so the
shared/independent comparison does not silently change feature definitions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap


ROOT = Path("/storage/liying/new_try/wind_cluster_forced_evolution")
SOURCE = ROOT / "artifacts/p0i_attribution_closure_20260915_v1/snapshot/src"
CACHE = ROOT / "data/model_cache_pad5_f8"
OUT = ROOT / "artifacts/cbh_per_station_independent_20260918_v1"
SEEDS = tuple(range(42, 52))


BASELINE_CONFIG: dict[str, Any] = {
    "architecture": "forced_evolution",
    "weather_channels": 14,
    "weather_hidden_channels": 48,
    "d_model": 64,
    "station_query_mode": "learned_cells",
    "station_neighborhood_cells": 3,
    "include_wind_geometry": True,
    "include_region_context": False,
    "forcing_path_mode": "difference",
    "time_query_mode": "full_attention",
    "fusion_mode": "history_only",
    "output_level": "hierarchical",
    "history_tcn_layers": 3,
    "trajectory_tcn_layers": 3,
    "attention_heads": 4,
    "dropout": 0.1,
}

CBH_CONFIG: dict[str, Any] = {
    "architecture": "p0i_condition_transfer",
    "baseline_family": "covariate_transformer",
    "d_model": 64,
    "weather_channels": 14,
    "history_visible_steps": 24,
    "dropout": 0.05,
    "alignment_variant": "P0I",
    "native_scope": "native7",
    "transfer_variant": "CBH",
    "condition_mode": "full",
    "history_baseline_model_config": BASELINE_CONFIG,
    "initialize_history_baseline_from_checkpoint": True,
    "freeze_history_baseline": True,
    "output_clamp_min": None,
}

TRAINING_PROTOCOL: dict[str, Any] = {
    "precision": "bf16",
    "cycles_per_batch": 12,
    "validation_cycles_per_batch": 24,
    "baseline_batch_size": 48,
    "baseline_validation_batch_size": 96,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "epochs": 80,
    "early_stopping_patience": 10,
    "gradient_clip": 1.0,
    "loss": "mae",
    "checkpoint_metric": "single_site_mae_cf",
    "output_clipping": False,
    "ema": False,
    "checkpoint_averaging": False,
    "ramp_loss": False,
    "auxiliary_loss": False,
}


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(item.dtype).encode())
        digest.update(str(tuple(item.shape)).encode())
        digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def atomic_torch_save(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)


def load_static() -> dict[str, np.ndarray]:
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
    return {
        name: np.load(CACHE / "static" / f"{name}.npy", allow_pickle=False)
        for name in names
    }


def single_site_static(full: Mapping[str, np.ndarray], site: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in full.items():
        result[name] = value[site : site + 1].copy() if name.startswith("station_") else value.copy()
    return result


def canonical_static_vectors(full: Mapping[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the static vectors used by the original 56-site models."""
    lat = torch.as_tensor(full["station_latitude"], dtype=torch.float32)
    lon = torch.as_tensor(full["station_longitude"], dtype=torch.float32)
    capacity = torch.as_tensor(full["station_capacity_mw"], dtype=torch.float32)
    weather_lat = torch.as_tensor(full["weather_latitude"], dtype=torch.float32)
    weather_lon = torch.as_tensor(full["weather_longitude"], dtype=torch.float32)
    station_query_static = torch.stack(
        (
            (lat - 0.5 * (weather_lat.min() + weather_lat.max()))
            / (weather_lat.max() - weather_lat.min()),
            (lon - 0.5 * (weather_lon.min() + weather_lon.max()))
            / (weather_lon.max() - weather_lon.min()),
            capacity / capacity.sum(),
            torch.log(capacity) / torch.log(capacity).mean(),
        ),
        dim=-1,
    )
    p0i_site = torch.stack(
        (
            (lat - lat.mean()) / lat.std(),
            (lon - lon.mean()) / lon.std(),
            capacity / capacity.mean(),
        ),
        dim=-1,
    )
    return station_query_static, p0i_site


def build_baseline(site: int, seed: int, full_static: Mapping[str, np.ndarray]) -> nn.Module:
    from wind_forcing.models.forced_evolution import build_forced_evolution_model

    seed_all(seed)
    model = build_forced_evolution_model(
        BASELINE_CONFIG,
        single_site_static(full_static, site),
        history_steps=24,
        forecast_steps=24,
    )
    query_static, _ = canonical_static_vectors(full_static)
    with torch.no_grad():
        model.forcing_encoder.station_query.station_static.copy_(query_static[site : site + 1])
    return model


def build_cbh(
    site: int,
    seed: int,
    full_static: Mapping[str, np.ndarray],
    baseline_state: Mapping[str, torch.Tensor],
) -> nn.Module:
    from wind_forcing.models import build_forecast_model
    import wind_forcing.models.p0i_transfer as p0i_transfer

    # The frozen source contains a runtime monotonicity assertion implemented
    # as Python control flow over a Tensor.  torch.func.vmap cannot trace that
    # assertion, so validate offsets in the data audit and use the identical
    # interpolation arithmetic without the Python branch here.
    def vmap_safe_interpolate(nodes, times, leads):
        right = torch.searchsorted(times.contiguous(), leads.contiguous()).clamp(
            1, times.shape[1] - 1
        )
        left = right - 1
        t0, t1 = times.gather(1, left), times.gather(1, right)
        alpha = ((leads - t0) / (t1 - t0)).clamp(0, 1)
        shape = (nodes.shape[0], nodes.shape[1], leads.shape[1], nodes.shape[3])
        v0 = nodes.gather(2, left[:, None, :, None].expand(shape))
        v1 = nodes.gather(2, right[:, None, :, None].expand(shape))
        return v0 + alpha[:, None, :, None] * (v1 - v0)

    p0i_transfer.interpolate = vmap_safe_interpolate

    seed_all(seed)
    model = build_forecast_model(
        CBH_CONFIG,
        single_site_static(full_static, site),
        history_steps=24,
        forecast_steps=24,
    )
    model.load_history_checkpoint(
        {"model_config": BASELINE_CONFIG, "model_state": baseline_state}
    )
    query_static, p0i_site = canonical_static_vectors(full_static)
    with torch.no_grad():
        model.baseline_model.forcing_encoder.station_query.station_static.copy_(
            query_static[site : site + 1]
        )
        model.site.copy_(p0i_site[site : site + 1])
        # Local weather is pre-extracted with the original bilinear weights.
        # The canonical station_weather method is retained, but reads that
        # value exactly once instead of gathering the full 24x25 grid again.
        model.indices.zero_()
        model.weights.zero_()
        model.weights[..., 0] = 1.0
    model.baseline_model.requires_grad_(False)
    model.baseline_model.eval()
    return model


class FunctionalEnsemble:
    """A torch.func ensemble with one independent parameter slice per seed."""

    def __init__(self, models: Sequence[nn.Module], device: torch.device, kind: str) -> None:
        if not models:
            raise ValueError("models cannot be empty")
        self.kind = str(kind)
        self.size = len(models)
        trainable = {name: value.requires_grad for name, value in models[0].named_parameters()}
        params, buffers = stack_module_state(list(models))
        self.params = {
            name: value.detach().to(device).requires_grad_(trainable[name])
            for name, value in params.items()
        }
        self.buffers = {name: value.detach().to(device) for name, value in buffers.items()}
        self.base = copy.deepcopy(models[0]).to("meta")
        self.device = device
        self.parameter_names = tuple(dict(models[0].named_parameters()))
        self.buffer_names = tuple(dict(models[0].named_buffers()))
        self.trainable_names = tuple(name for name in self.parameter_names if trainable[name])
        if self.kind == "baseline":
            self._vmapped = vmap(self._call_baseline, in_dims=(0, 0, 0, 0, 0), randomness="different")
        elif self.kind == "cbh":
            self._vmapped = vmap(
                self._call_cbh,
                in_dims=(0, 0, 0, 0, 0, 0, 0, 0),
                randomness="different",
            )
        else:
            raise ValueError(f"Unknown ensemble kind: {self.kind}")

    def _call_baseline(self, params, buffers, history, cluster, phase):
        output = functional_call(
            self.base,
            (params, buffers),
            (),
            {"station_history": history, "cluster_history": cluster, "origin_phase": phase},
        )
        return output["station_factor"]

    def _call_cbh(self, params, buffers, history, cluster, phase, weather, weather_index, offsets):
        output = functional_call(
            self.base,
            (params, buffers),
            (),
            {
                "station_history": history,
                "cluster_history": cluster,
                "origin_phase": phase,
                "weather": weather,
                "weather_batch_index": weather_index,
                "weather_time_offset_hours": offsets,
            },
        )
        return output["station_factor"]

    def train(self, mode: bool = True) -> None:
        self.base.train(mode)
        if self.kind == "cbh":
            self.base.baseline_model.eval()

    def __call__(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if self.kind == "baseline":
            prediction = self._vmapped(
                self.params,
                self.buffers,
                batch["station_history"],
                batch["cluster_history"],
                batch["origin_phase"],
            )
        else:
            prediction = self._vmapped(
                self.params,
                self.buffers,
                batch["station_history"],
                batch["cluster_history"],
                batch["origin_phase"],
                batch["weather"],
                batch["weather_batch_index"],
                batch["weather_time_offset_hours"],
            )
        return {"station_factor": prediction}

    def optimizer_parameters(self) -> list[torch.Tensor]:
        return [self.params[name] for name in self.trainable_names]

    def clip_grad_norm_per_model(self, maximum: float) -> torch.Tensor:
        squared = torch.zeros(self.size, device=self.device, dtype=torch.float32)
        for name in self.trainable_names:
            gradient = self.params[name].grad
            if gradient is not None:
                squared += gradient.float().square().flatten(1).sum(1)
        norms = squared.sqrt()
        scale = (float(maximum) / (norms + 1e-6)).clamp(max=1.0)
        for name in self.trainable_names:
            gradient = self.params[name].grad
            if gradient is not None:
                gradient.mul_(scale.reshape((self.size,) + (1,) * (gradient.ndim - 1)).to(gradient))
        return norms

    def state_for(self, index: int) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name in self.parameter_names:
            result[name] = self.params[name][index].detach().cpu().clone()
        for name in self.buffer_names:
            result[name] = self.buffers[name][index].detach().cpu().clone()
        if set(result) != set(self.base.state_dict()):
            missing = set(self.base.state_dict()).difference(result)
            extra = set(result).difference(self.base.state_dict())
            raise RuntimeError(f"Functional state mismatch: missing={missing}, extra={extra}")
        return result


def ensemble_from_states(
    templates: Sequence[nn.Module],
    states: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device,
    kind: str,
) -> FunctionalEnsemble:
    if len(templates) != len(states):
        raise ValueError("template/state count mismatch")
    for model, state in zip(templates, states):
        model.load_state_dict(state, strict=True)
    return FunctionalEnsemble(templates, device, kind)
