"""Exact single-station specialization of the frozen original CBH correction graph."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

from paths import CACHE, ORIGINAL, ORIGINAL_RUNS, V2
from artifact_utils import sha256


sys.path.insert(0, str(V2 / "code"))
from generators import Generator  # noqa: E402
from common import interpolate  # noqa: E402

sys.path.insert(0, str(ORIGINAL / "snapshot/src"))
from wind_forcing.config import load_config  # noqa: E402
from wind_forcing.data.dataset import PreparedCache  # noqa: E402
from wind_forcing.models import build_forecast_model  # noqa: E402


STATIC = PreparedCache(CACHE).static


def source_checkpoint(seed: int) -> Path:
    return ORIGINAL_RUNS / "results" / f"CBH_seed{seed}" / "best_model.pt"


class CBHSelected(nn.Module):
    """CBH's effective graph for one output station, initialized from its checkpoint.

    The frozen multi-station history-only trajectory is supplied as ``baseline``.
    Every trainable module on the selected station's original CBH path is reused
    with its exact checkpoint weights.
    """

    MODULES = (
        "history_projection",
        "history_encoder",
        "context_projection",
        "native_projection",
        "align_v",
        "align_out",
        "condition_projection",
        "transfer_adapter",
        "decoder",
        "readout",
    )

    def __init__(self, station_index: int, seed: int) -> None:
        super().__init__()
        config_path = ORIGINAL / "configs" / f"CBH_seed{seed}.yaml"
        checkpoint_path = source_checkpoint(seed)
        config = load_config(config_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert payload["model_config"] == config["model"]
        full = build_forecast_model(
            payload["model_config"], STATIC, history_steps=24, forecast_steps=24
        )
        full.load_state_dict(payload["model_state"], strict=True)
        assert full.transfer_variant == "CBH" and full.summary_mode == "mean"
        assert full.condition_mode == "full" and full.variant == "P0I"
        for name in self.MODULES:
            setattr(self, name, getattr(full, name))
        self.condition_bias = nn.Parameter(full.condition_bias[:, 0].detach().clone())
        self.register_buffer("site", full.site[station_index].detach().clone())
        self.station_index = int(station_index)
        self.seed = int(seed)
        self.source_checkpoint_sha256 = sha256(checkpoint_path)
        self.source_checkpoint_epoch = int(payload["epoch"])
        del full

    def forward(
        self,
        history: torch.Tensor,
        weather_15min: torch.Tensor,
        phase: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(history)
        origin = history[:, -1:]
        lead = torch.arange(1, 25, device=history.device, dtype=history.dtype)[None].expand(batch, -1) / 4
        clock = torch.stack(
            (lead / 6, torch.sin(2 * math.pi * lead / 24), torch.cos(2 * math.pi * lead / 24)), -1
        )
        context = torch.cat(
            (
                clock,
                self.site[None, None].expand(batch, 24, -1),
                phase[:, None, None].to(history).expand(-1, 24, 1) / 3,
            ),
            -1,
        )
        history_time = torch.arange(-23, 1, device=history.device, dtype=history.dtype) / 24
        memory = self.history_encoder(
            self.history_projection(torch.stack((history, history_time.expand_as(history)), -1))
        )
        # Original P0I has one permitted interpolated weather key per target.
        # Its masked softmax is therefore exactly one and reduces to V/out.
        weather = self.align_out(self.align_v(self.native_projection(weather_15min)))
        delta = baseline - torch.cat((origin, baseline[:, :-1]), -1)
        summary = memory.mean(1)[:, None].expand(-1, 24, -1)
        condition_raw = torch.cat(
            (summary, baseline[..., None], delta[..., None], lead[..., None] / 6), -1
        )
        condition = self.condition_projection(condition_raw) + self.condition_bias
        query = (
            self.context_projection(context)
            + weather
            + self.transfer_adapter(torch.cat((weather, condition), -1))
        )
        correction = self.readout(self.decoder(query, memory))[..., 0]
        return baseline + correction


class CBHAdapterSR(nn.Module):
    def __init__(
        self,
        method: str,
        scale: torch.Tensor,
        seed: int,
        station_index: int,
    ) -> None:
        super().__init__()
        self.name = f"CBH_ADAPT_{method}"
        self.method = method
        self.predictor = CBHSelected(station_index, seed)
        self.generator = None if method == "LINEAR" else Generator(method, scale, seed)

    def refine(
        self,
        weather: torch.Tensor,
        times: torch.Tensor,
        noise: torch.Tensor,
        steps: int = 8,
    ) -> torch.Tensor:
        assert self.generator is not None
        batch = len(weather)
        left = weather[:, :-1].reshape(-1, 14)
        right = weather[:, 1:].reshape(-1, 14)
        context = weather.mean(1)[:, None].expand(-1, 6, -1).reshape(-1, 14)
        span = (times[:, 1:] - times[:, :-1]).reshape(-1)
        middle = self.generator.sample(
            left, right, context, span, noise.reshape(-1, 3, 14), steps
        ).reshape(batch, 6, 3, 14)
        # Observed hourly nodes bypass the generator exactly.
        grid = torch.cat(
            (torch.cat((weather[:, :-1, None], middle), 2).reshape(batch, 24, 14), weather[:, -1:]),
            1,
        )
        query = torch.arange(1, 25, device=weather.device, dtype=times.dtype)[None] / 4
        index = ((query - times[:, :1]) * 4).round().long().clamp(0, 24)
        return grid.gather(1, index[..., None].expand(-1, -1, 14))

    def forward(
        self,
        history: torch.Tensor,
        weather: torch.Tensor,
        times: torch.Tensor,
        phase: torch.Tensor,
        baseline: torch.Tensor,
        noise: torch.Tensor,
        auxiliary_noise: torch.Tensor,
        auxiliary_time: torch.Tensor,
        auxiliary_extra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.method == "LINEAR":
            query = torch.arange(1, 25, device=history.device, dtype=times.dtype)[None].expand(len(history), -1) / 4
            weather_15min = interpolate(weather, times, query)[0]
            auxiliary = history.new_zeros(())
        else:
            assert self.generator is not None
            weather_15min = self.refine(weather, times, noise)
            if self.training:
                targets = torch.stack([weather[:, j + 1 : j + 4] for j in range(3)], 1).reshape(-1, 3, 14)
                left = weather[:, :3].reshape(-1, 14)
                right = weather[:, 4:7].reshape(-1, 14)
                context = weather.mean(1)[:, None].expand(-1, 3, -1).reshape(-1, 14)
                auxiliary = self.generator(
                    left,
                    right,
                    context,
                    weather.new_full((len(left),), 4.0),
                    targets,
                    auxiliary_noise.reshape(-1, 3, 14),
                    auxiliary_time.reshape(-1),
                    auxiliary_extra.reshape(-1, 3, 14),
                ).mean()
            else:
                auxiliary = weather.new_zeros(())
        return self.predictor(history, weather_15min, phase, baseline), auxiliary
