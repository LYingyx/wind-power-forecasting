from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from wind_forcing.models.baselines import DLinearHead
from wind_forcing.models.components import CausalResidualBlock, inverse_softplus


def _heads(d_model: int, requested: int) -> int:
    value = min(int(requested), d_model)
    while value > 1 and d_model % value:
        value -= 1
    return value


class LeadEmbedding(nn.Module):
    def __init__(self, d_model: int, forecast_steps: int) -> None:
        super().__init__()
        leads = torch.arange(1, forecast_steps + 1, dtype=torch.float32) / 4.0
        self.register_buffer("lead_hours", leads)
        self.network = nn.Sequential(
            nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model)
        )

    def forward(self, batch: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        lead = self.lead_hours.to(device=device, dtype=dtype)
        angle = lead * (2.0 * math.pi / 24.0)
        feature = torch.stack(
            (lead / 6.0, torch.sin(angle), torch.cos(angle), torch.sin(4 * angle), torch.cos(4 * angle)),
            dim=-1,
        )
        return self.network(feature).unsqueeze(0).expand(batch, -1, -1)


def history_features(history: torch.Tensor, visible_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    if history.ndim != 2 or not 1 <= visible_steps <= history.shape[1]:
        raise ValueError("Invalid aggregate history or visible window")
    batch, steps = history.shape
    mask = torch.arange(steps, device=history.device) >= steps - visible_steps
    mask = mask.reshape(1, steps).expand(batch, -1)
    values = history * mask.to(history)
    delta = torch.zeros_like(history)
    if visible_steps > 1:
        delta[:, -visible_steps + 1 :] = (
            history[:, -visible_steps + 1 :] - history[:, -visible_steps:-1]
        )
    relative = torch.arange(-(steps - 1), 1, device=history.device, dtype=history.dtype) / 4.0
    relative = relative.reshape(1, steps).expand(batch, -1)
    features = torch.stack((values, delta, relative / 6.0, mask.to(history)), dim=-1)
    return features, mask


class AggregateHistoryForecast(nn.Module):
    """Station-free history forecast and reusable B(H) implementation."""

    def __init__(
        self,
        mode: str,
        *,
        total_capacity_mw: float,
        history_steps: int,
        forecast_steps: int,
        visible_steps: int,
        d_model: int,
        layers: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if mode not in {"persistence", "dlinear", "tcn", "patchtst"}:
            raise ValueError("Unsupported aggregate history mode")
        self.mode = mode
        self.history_steps = int(history_steps)
        self.forecast_steps = int(forecast_steps)
        self.visible_steps = int(visible_steps)
        self.d_model = int(d_model)
        self.register_buffer("total_capacity_mw", torch.tensor(float(total_capacity_mw)))
        self.zero_gradient_anchor = nn.Parameter(torch.zeros(()))
        if mode == "dlinear":
            self.network: nn.Module = DLinearHead(history_steps, forecast_steps)
        elif mode == "tcn":
            self.input_projection = nn.Conv1d(4, d_model, 1)
            self.network = nn.Sequential(
                *[CausalResidualBlock(d_model, 2**layer, dropout) for layer in range(layers)]
            )
            self.head = nn.Linear(d_model, forecast_steps)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        elif mode == "patchtst":
            self.patch_length = 4
            self.patch_stride = 2
            count = 1 + (history_steps - self.patch_length) // self.patch_stride
            self.input_projection = nn.Linear(4 * self.patch_length, d_model)
            self.position = nn.Parameter(torch.zeros(1, count, d_model))
            nn.init.normal_(self.position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model,
                _heads(d_model, attention_heads),
                4 * d_model,
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.network = nn.TransformerEncoder(layer, max(1, layers), nn.LayerNorm(d_model))
            self.head = nn.Linear(d_model, forecast_steps)

    def forward(self, *, cluster_history: torch.Tensor, **_: Any) -> dict[str, Any]:
        if cluster_history.shape[1:] != (self.history_steps,):
            raise ValueError("cluster_history has the wrong shape")
        history_latent: torch.Tensor | None = None
        if self.mode == "persistence":
            prediction = cluster_history[:, -1:].expand(-1, self.forecast_steps)
            prediction = prediction + self.zero_gradient_anchor * 0.0
        else:
            features, _ = history_features(cluster_history, self.visible_steps)
            if self.mode == "dlinear":
                delta = self.network(features[..., 0])
            elif self.mode == "tcn":
                encoded = self.network(self.input_projection(features.transpose(1, 2)))
                history_latent = encoded[:, :, -1]
                delta = self.head(history_latent)
            else:
                patches = features.unfold(1, self.patch_length, self.patch_stride)
                patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
                encoded = self.network(self.input_projection(patches) + self.position)
                delta = self.head(encoded.mean(dim=1))
            prediction = F.softplus(
                inverse_softplus(cluster_history[:, -1:].clamp_min(1e-4)) + delta
            )
        return {
            "cluster_factor": prediction,
            "baseline_cluster_factor": prediction,
            "anchor_baseline_cluster_factor": None,
            "weather_features": None,
            "state_attention": None,
            "weather_attention": None,
            "history_latent": history_latent,
        }

    def encode_latent(self, cluster_history: torch.Tensor) -> torch.Tensor:
        """Return the representation used by a TCN baseline forecast head."""
        if self.mode != "tcn":
            raise ValueError("A reusable history latent is only defined for the TCN baseline")
        features, _ = history_features(cluster_history, self.visible_steps)
        encoded = self.network(self.input_projection(features.transpose(1, 2)))
        return encoded[:, :, -1]


class RegionalCNN(nn.Module):
    def __init__(self, channels: int, d_model: int) -> None:
        super().__init__()
        hidden = max(32, d_model // 2)
        self.input = nn.Conv2d(channels, hidden, 3, padding=1)
        self.block1 = nn.Sequential(
            nn.GroupNorm(4, hidden), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(4, hidden), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(4, hidden), nn.GELU(), nn.Conv2d(hidden, d_model, 1)
        )

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        leading = weather.shape[:-3]
        frames = weather.reshape(-1, *weather.shape[-3:])
        values = self.input(frames)
        values = values + self.block1(values)
        values = values + self.block2(values)
        return self.output(values).reshape(*leading, -1, *weather.shape[-2:])


class FullPathQuery(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = _heads(d_model, heads)
        self.head_dim = d_model // self.heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.time_bias = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, self.heads))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        token_times: torch.Tensor,
        *,
        value_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, queries, _ = query.shape
        count = tokens.shape[1]
        q = self.q(query).reshape(batch, queries, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(tokens).reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)
        if value_override is None:
            v = self.v(tokens).reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)
            v = v[:, :, None].expand(-1, -1, queries, -1, -1)
        else:
            v = self.v(value_override).reshape(
                batch, queries, count, self.heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
        lead = torch.arange(1, queries + 1, device=query.device, dtype=query.dtype) / 4.0
        delta = lead.reshape(1, queries, 1) - token_times[:, None]
        bias = self.time_bias(torch.stack((delta / 6.0, delta.abs() / 6.0), dim=-1))
        score = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.head_dim)
        score = score + bias.permute(0, 3, 1, 2)
        attention = torch.softmax(score, dim=-1)
        if value_override is None:
            output = torch.einsum("bhqk,bhqkd->bhqd", self.dropout(attention), v)
        else:
            output = torch.einsum("bhqk,bhqkd->bhqd", self.dropout(attention), v)
        output = output.transpose(1, 2).reshape(batch, queries, self.d_model)
        return self.out(output), attention.mean(dim=1)


def linear_interpolate(
    values: torch.Tensor, node_times: torch.Tensor, target_times: torch.Tensor
) -> torch.Tensor:
    """Sample-specific interpolation with boundary replication, never extrapolation."""
    if values.shape[:2] != node_times.shape:
        raise ValueError("values and node_times disagree")
    batch, nodes = node_times.shape
    targets = target_times.shape[1]
    right = torch.searchsorted(node_times.contiguous(), target_times.contiguous(), right=False)
    right = right.clamp(1, nodes - 1)
    left = right - 1
    left_time = node_times.gather(1, left)
    right_time = node_times.gather(1, right)
    fraction = ((target_times - left_time) / (right_time - left_time).clamp_min(1e-6)).clamp(0, 1)
    tail = values.shape[2:]
    gather_shape = (batch, targets) + (1,) * len(tail)
    expand_shape = (batch, targets) + tail
    left_value = values.gather(1, left.reshape(gather_shape).expand(expand_shape))
    right_value = values.gather(1, right.reshape(gather_shape).expand(expand_shape))
    weight = fraction.reshape(gather_shape)
    before = target_times <= node_times[:, :1]
    after = target_times >= node_times[:, -1:]
    output = left_value * (1.0 - weight) + right_value * weight
    output = torch.where(
        before.reshape(gather_shape), values[:, :1].expand(expand_shape), output
    )
    output = torch.where(
        after.reshape(gather_shape), values[:, -1:].expand(expand_shape), output
    )
    return output


class AggregateWeatherEncoder(nn.Module):
    MODES = {
        "raw_interpolation",
        "latent_interpolation",
        "full_path",
        "target_value",
        "target_value_no_time",
        "dynamic_pool",
        "regional_tokens",
        "patch_regional",
        "joint_spacetime",
        "factorized_spacetime",
    }

    def __init__(
        self,
        mode: str,
        *,
        channels: int,
        d_model: int,
        heads: int,
        dropout: float,
        forecast_steps: int,
        capacity_density: Sequence[Sequence[float]],
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported aggregate weather mode: {mode}")
        self.mode = mode
        self.d_model = int(d_model)
        density = torch.as_tensor(capacity_density, dtype=torch.float32)
        if density.ndim != 2 or not torch.isclose(density.sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError("capacity_density must be a normalized 2-D map")
        self.register_buffer("capacity_density", density)
        self.lead = LeadEmbedding(d_model, forecast_steps)
        self.path_query = FullPathQuery(d_model, heads, dropout)
        if mode in {"patch_regional", "joint_spacetime", "factorized_spacetime"}:
            self.patch = nn.Conv2d(channels, d_model, kernel_size=(4, 5), stride=(4, 5))
            layer = nn.TransformerEncoderLayer(
                d_model, _heads(d_model, heads), 2 * d_model, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.patch_encoder = nn.TransformerEncoder(layer, 2, nn.LayerNorm(d_model))
            patch_density = F.avg_pool2d(density[None, None], (4, 5), (4, 5)).flatten()
            row = torch.linspace(-1, 1, 6).reshape(6, 1).expand(6, 5).flatten()
            col = torch.linspace(-1, 1, 5).reshape(1, 5).expand(6, 5).flatten()
            self.register_buffer("patch_metadata", torch.stack((row, col, patch_density), dim=-1))
            self.patch_position = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        else:
            self.cnn = RegionalCNN(channels, d_model)
        self.region_queries = nn.Parameter(torch.randn(4, d_model) * 0.02)
        self.region_attention = nn.MultiheadAttention(
            d_model, _heads(d_model, heads), dropout=dropout, batch_first=True
        )
        self.value_adapter = nn.Sequential(
            nn.Linear(d_model + 2, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )
        self.dynamic_score = nn.Sequential(
            nn.Linear(d_model + 3, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.factor_time = nn.MultiheadAttention(
            d_model, _heads(d_model, heads), dropout=dropout, batch_first=True
        )
        self.factor_space = nn.MultiheadAttention(
            d_model, _heads(d_model, heads), dropout=dropout, batch_first=True
        )
        # All weather routes pass through the same residual adapter family.  A
        # wider bottleneck on the lighter CNN routes keeps total trainable
        # capacity aligned with the patch routes without changing B or C.
        adapter_hidden = 16 if mode in {
            "patch_regional", "joint_spacetime", "factorized_spacetime"
        } else 512
        self.budget_adapter = nn.Sequential(
            nn.Linear(d_model, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, d_model),
        )

    def _capacity_pool(self, grid: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...dhw,hw->...d", grid, self.capacity_density.to(grid))

    def encode_hourly(self, weather: torch.Tensor) -> torch.Tensor:
        """Encode native NWP nodes before target-time interpolation.

        This path is intentionally exposed for the frozen, seed-specific latent
        cache used by response-state identification.
        """
        if self.mode != "latent_interpolation":
            raise ValueError("Native hourly latent caching is only defined for R2")
        return self._capacity_pool(self.cnn(weather))

    def forward_from_hourly(
        self,
        hourly_unique: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if self.mode != "latent_interpolation":
            raise ValueError("Cached hourly latents are only supported by R2")
        batch = len(weather_batch_index)
        hourly = hourly_unique.index_select(0, weather_batch_index)
        target_times = self.lead.lead_hours.to(hourly).reshape(1, -1).expand(batch, -1)
        target = linear_interpolate(hourly, weather_time_offset_hours, target_times)
        target = target + self.budget_adapter(target)
        return {
            "weather_features": target,
            "hourly_weather_tokens": hourly,
            "weather_attention": None,
            "spatial_attention": None,
        }

    def _patch_tokens(self, weather: torch.Tensor) -> torch.Tensor:
        cycles, nodes = weather.shape[:2]
        tokens = self.patch(weather.flatten(0, 1)).flatten(2).transpose(1, 2)
        tokens = tokens + self.patch_position(self.patch_metadata.to(tokens))[None]
        return self.patch_encoder(tokens).reshape(cycles, nodes, 30, self.d_model)

    def _regional_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        cycles, nodes, dim, height, width = grid.shape
        cells = grid.permute(0, 1, 3, 4, 2).reshape(cycles * nodes, height * width, dim)
        query = self.region_queries[None].expand(cycles * nodes, -1, -1)
        pooled, _ = self.region_attention(query, cells, cells, need_weights=False)
        return pooled.reshape(cycles, nodes, 4, dim)

    def forward(
        self,
        weather: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        batch = len(weather_batch_index)
        lead = self.lead(batch, dtype=weather.dtype, device=weather.device)
        target_times = self.lead.lead_hours.to(weather).reshape(1, -1).expand(batch, -1)
        spatial_attention: torch.Tensor | None = None
        if self.mode == "raw_interpolation":
            samples = weather.index_select(0, weather_batch_index)
            interpolated = linear_interpolate(samples, weather_time_offset_hours, target_times)
            grid = self.cnn(interpolated)
            target = self._capacity_pool(grid)
            hourly = None
            attention = None
        elif self.mode in {"joint_spacetime", "factorized_spacetime"}:
            patch = self._patch_tokens(weather).index_select(0, weather_batch_index)
            tokens = patch.reshape(batch, -1, self.d_model)
            token_times = weather_time_offset_hours[:, :, None].expand(-1, -1, 30).reshape(batch, -1)
            if self.mode == "joint_spacetime":
                first, a1 = self.factor_time(lead, tokens, tokens, need_weights=True, average_attn_weights=False)
                target, a2 = self.factor_space(first, tokens, tokens, need_weights=True, average_attn_weights=False)
                attention = (a1.mean(dim=1) + a2.mean(dim=1)) / 2.0
            else:
                time_tokens = patch.permute(0, 2, 1, 3).reshape(batch * 30, 8, self.d_model)
                query = lead[:, :, None].expand(-1, -1, 30, -1).permute(0, 2, 1, 3)
                query = query.reshape(batch * 30, lead.shape[1], self.d_model)
                time_read, _ = self.factor_time(query, time_tokens, time_tokens, need_weights=False)
                time_read = time_read.reshape(batch, 30, lead.shape[1], self.d_model).permute(0, 2, 1, 3)
                spatial_query = lead.reshape(batch * lead.shape[1], 1, self.d_model)
                spatial_value = time_read.reshape(batch * lead.shape[1], 30, self.d_model)
                target, a2 = self.factor_space(
                    spatial_query, spatial_value, spatial_value,
                    need_weights=True, average_attn_weights=False,
                )
                target = target.reshape(batch, lead.shape[1], self.d_model)
                attention = a2.mean(dim=1).reshape(batch, lead.shape[1], 30)
            hourly = patch.mean(dim=2)
        elif self.mode == "patch_regional":
            patch = self._patch_tokens(weather)
            cycles, nodes = patch.shape[:2]
            query = self.region_queries[None].expand(cycles * nodes, -1, -1)
            values = patch.reshape(cycles * nodes, 30, self.d_model)
            regions, weights = self.region_attention(
                query, values, values, need_weights=True, average_attn_weights=False
            )
            regions = regions.reshape(cycles, nodes * 4, self.d_model).index_select(0, weather_batch_index)
            times = weather_time_offset_hours[:, :, None].expand(-1, -1, 4).reshape(batch, -1)
            target, attention = self.path_query(lead, regions, times)
            hourly = regions.reshape(batch, 8, 4, self.d_model).mean(dim=2)
            spatial_attention = weights.mean(dim=1).reshape(cycles, nodes, 4, 30)
        else:
            grid = self.cnn(weather)
            if self.mode == "dynamic_pool":
                cycles, nodes, dim, height, width = grid.shape
                row = torch.linspace(-1, 1, height, device=grid.device, dtype=grid.dtype)
                col = torch.linspace(-1, 1, width, device=grid.device, dtype=grid.dtype)
                yy, xx = torch.meshgrid(row, col, indexing="ij")
                meta = torch.stack((yy, xx, self.capacity_density.to(grid)), dim=-1)
                cells = grid.permute(0, 1, 3, 4, 2)
                logits = self.dynamic_score(
                    torch.cat((cells, meta.reshape(1, 1, height, width, 3).expand(cycles, nodes, -1, -1, -1)), dim=-1)
                ).squeeze(-1)
                weights = torch.softmax(logits.flatten(-2), dim=-1).reshape(cycles, nodes, height, width)
                hourly_unique = torch.einsum("ctdhw,cthw->ctd", grid, weights)
                spatial_attention = weights
            elif self.mode == "regional_tokens":
                regions = self._regional_tokens(grid)
                samples = regions.index_select(0, weather_batch_index).reshape(batch, 32, self.d_model)
                times = weather_time_offset_hours[:, :, None].expand(-1, -1, 4).reshape(batch, 32)
                target, attention = self.path_query(lead, samples, times)
                target = target + self.budget_adapter(target)
                hourly = regions.mean(dim=2).index_select(0, weather_batch_index)
                return {
                    "weather_features": target,
                    "hourly_weather_tokens": hourly,
                    "weather_attention": attention,
                    "spatial_attention": None,
                }
            else:
                hourly_unique = self._capacity_pool(grid)
            hourly = hourly_unique.index_select(0, weather_batch_index)
            if self.mode == "latent_interpolation":
                target = linear_interpolate(hourly, weather_time_offset_hours, target_times)
                attention = None
            else:
                value_override = None
                if self.mode in {"target_value", "target_value_no_time"}:
                    delta = target_times[:, :, None] - weather_time_offset_hours[:, None]
                    time = torch.zeros_like(delta) if self.mode == "target_value_no_time" else delta
                    expanded = hourly[:, None].expand(-1, target_times.shape[1], -1, -1)
                    value_override = self.value_adapter(
                        torch.cat((expanded, time[..., None] / 6.0, time.abs()[..., None] / 6.0), dim=-1)
                    )
                target, attention = self.path_query(
                    lead, hourly, weather_time_offset_hours, value_override=value_override
                )
        target = target + self.budget_adapter(target)
        return {
            "weather_features": target,
            "hourly_weather_tokens": hourly,
            "weather_attention": attention,
            "spatial_attention": spatial_attention,
        }


class AggregateStateEncoder(nn.Module):
    MODES = {"current", "recent_1h", "recent_3h", "full_6h", "multiscale", "horizon"}

    def __init__(self, mode: str, d_model: int, heads: int, dropout: float, forecast_steps: int) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported aggregate state mode: {mode}")
        self.mode = mode
        self.d_model = d_model
        self.visible_steps_override: int | None = None
        self.token = nn.Sequential(nn.Linear(5, d_model), nn.GELU(), nn.LayerNorm(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, _heads(d_model, heads), 2 * d_model, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2, nn.LayerNorm(d_model))
        self.pool = nn.MultiheadAttention(d_model, _heads(d_model, heads), dropout=dropout, batch_first=True)
        self.shared_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.lead = LeadEmbedding(d_model, forecast_steps)

    def _inputs(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps = history.shape
        if steps != 24:
            raise ValueError("State encoder expects 24 historical quarter hours")
        if self.mode == "multiscale":
            older = history[:, :20].reshape(batch, 5, 4).mean(dim=-1)
            values = torch.cat((older, history[:, -4:]), dim=1)
            times = torch.tensor(
                [-5.5, -4.5, -3.5, -2.5, -1.5, -0.75, -0.5, -0.25, 0.0],
                device=history.device, dtype=history.dtype,
            ).reshape(1, 9).expand(batch, -1)
            duration = torch.cat(
                (torch.ones(batch, 5, device=history.device, dtype=history.dtype),
                 torch.full((batch, 4), 0.25, device=history.device, dtype=history.dtype)),
                dim=1,
            )
            delta = torch.zeros_like(values)
            delta[:, 1:] = values[:, 1:] - values[:, :-1]
            features = torch.stack((values, delta, times / 6.0, torch.ones_like(values), duration), dim=-1)
            pad = torch.zeros(batch, 15, 5, device=history.device, dtype=history.dtype)
            features = torch.cat((pad, features), dim=1)
            mask = torch.zeros(batch, 24, device=history.device, dtype=torch.bool)
            mask[:, -9:] = True
            if self.visible_steps_override is not None:
                mask[:, :-min(int(self.visible_steps_override), 4)] = False
            return features, mask
        visible = {"current": 1, "recent_1h": 4, "recent_3h": 12}.get(self.mode, 24)
        if self.visible_steps_override is not None:
            visible = min(visible, int(self.visible_steps_override))
        base, mask = history_features(history, visible)
        duration = torch.full_like(history, 0.25)
        return torch.cat((base, duration[..., None]), dim=-1), mask

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features, mask = self._inputs(history)
        encoded = self.encoder(self.token(features), src_key_padding_mask=~mask)
        batch = len(history)
        if self.mode == "horizon":
            query = self.lead(batch, dtype=history.dtype, device=history.device)
        else:
            query = self.shared_query.to(history).expand(batch, -1, -1)
        pooled, attention = self.pool(
            query, encoded, encoded, key_padding_mask=~mask,
            need_weights=True, average_attn_weights=False,
        )
        if self.mode != "horizon":
            pooled = pooled.expand(-1, 24, -1)
            attention = attention.expand(-1, -1, 24, -1)
        return pooled, attention.mean(dim=1)


class AggregateCorrectionModel(nn.Module):
    def __init__(
        self,
        baseline: AggregateHistoryForecast,
        weather_encoder: AggregateWeatherEncoder,
        state_encoder: AggregateStateEncoder,
        *,
        d_model: int,
        forecast_steps: int,
        trajectory_layers: int,
        dropout: float,
        conditioning: str,
        fusion: str,
    ) -> None:
        super().__init__()
        if conditioning not in {"none", "state", "weather", "state_weather"}:
            raise ValueError("Invalid correction conditioning")
        if fusion not in {"blackbox", "additive"}:
            raise ValueError("Invalid correction fusion")
        self.baseline = baseline
        self.anchor_baseline = copy.deepcopy(baseline).eval().requires_grad_(False)
        self.weather_encoder = weather_encoder
        self.state_encoder = state_encoder
        self.conditioning = conditioning
        self.fusion = fusion
        self.forecast_steps = forecast_steps
        self.register_buffer("total_capacity_mw", baseline.total_capacity_mw.detach().clone())
        self.lead = LeadEmbedding(d_model, forecast_steps)
        self.fusion_network = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model),
        )
        self.trajectory = nn.Sequential(
            *[CausalResidualBlock(d_model, 2**layer, dropout) for layer in range(trajectory_layers)]
        )
        self.head = nn.Conv1d(d_model, 1, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def load_history_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.baseline.load_state_dict(state, strict=True)
        self.anchor_baseline.load_state_dict(state, strict=True)
        self.anchor_baseline.eval().requires_grad_(False)

    def _decode(self, representation: torch.Tensor) -> torch.Tensor:
        return self.head(self.trajectory(representation.transpose(1, 2))).squeeze(1)

    def forward(
        self,
        *,
        cluster_history: torch.Tensor,
        correction_state_history: torch.Tensor | None = None,
        weather: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
        **_: Any,
    ) -> dict[str, Any]:
        baseline = self.baseline(cluster_history=cluster_history)["cluster_factor"]
        with torch.no_grad():
            anchor = self.anchor_baseline(cluster_history=cluster_history)["cluster_factor"]
        state_active = self.conditioning in {"state", "state_weather"}
        lead = self.lead(len(cluster_history), dtype=cluster_history.dtype, device=cluster_history.device)
        if state_active:
            state_history = (
                cluster_history if correction_state_history is None else correction_state_history
            )
            if state_history.shape != cluster_history.shape:
                raise ValueError(
                    "correction_state_history must have the same shape as cluster_history"
                )
            state, state_attention = self.state_encoder(state_history)
        else:
            # Keep the module and checkpoint keys for parameter-matched controls,
            # but do not spend compute on a conditioner whose output is inactive.
            state = torch.zeros_like(lead)
            state_attention = None
        weather_output = self.weather_encoder(weather, weather_batch_index, weather_time_offset_hours)
        weather_features = weather_output["weather_features"]
        assert weather_features is not None
        weather_active = self.conditioning in {"weather", "state_weather"}
        state_input = state if state_active else torch.zeros_like(state)
        weather_input = weather_features if weather_active else torch.zeros_like(weather_features)
        if self.fusion == "additive":
            zero_state = torch.zeros_like(state_input)
            zero_weather = torch.zeros_like(weather_input)
            context = self.fusion_network(torch.cat((zero_state, zero_weather, lead), dim=-1))
            state_part = self.fusion_network(torch.cat((state_input, zero_weather, lead), dim=-1))
            weather_part = self.fusion_network(torch.cat((zero_state, weather_input, lead), dim=-1))
            correction = self._decode(state_part) + self._decode(weather_part) - self._decode(context)
        else:
            fused = self.fusion_network(torch.cat((state_input, weather_input, lead), dim=-1))
            correction = self._decode(fused)
        prediction = (baseline + correction).clamp_min(0.0)
        return {
            "cluster_factor": prediction,
            "baseline_cluster_factor": baseline,
            "anchor_baseline_cluster_factor": anchor,
            "weather_correction_factor": correction,
            "weather_features": weather_features,
            "state_attention": state_attention,
            "weather_attention": weather_output["weather_attention"],
            "spatial_attention": weather_output["spatial_attention"],
        }


class AggregateStrictModulationModel(AggregateCorrectionModel):
    """R2 weather correction whose state can only alter a weather-driven effect.

    The fixed R2-W decoder is decomposed as

        D_W(z, tau) = D_W(0, tau) + E_W(z, tau).

    Every state-conditioned path acts on ``E_W`` or on ``z`` through a
    zero-preserving map.  Consequently ``C(0, s_1, tau) == C(0, s_2, tau)``
    by construction, even when the decoder has biases or lead-only effects.
    """

    MODES = {"none", "effect_scale", "channel_scale", "low_rank", "experts", "ramp"}
    STATE_SOURCES = {"current", "baseline_latent"}
    TRAINING_STAGES = {"adapter", "joint"}

    def __init__(
        self,
        baseline: AggregateHistoryForecast,
        weather_encoder: AggregateWeatherEncoder,
        state_encoder: AggregateStateEncoder,
        *,
        d_model: int,
        forecast_steps: int,
        trajectory_layers: int,
        dropout: float,
        modulation_mode: str,
        state_source: str,
        training_stage: str,
        use_reference_state: bool = False,
        amplitude: float = 0.25,
        low_rank: int = 4,
        expert_count: int = 3,
        expert_coefficient: float = 0.1,
    ) -> None:
        if modulation_mode not in self.MODES:
            raise ValueError(f"Unsupported strict modulation mode: {modulation_mode}")
        if state_source not in self.STATE_SOURCES:
            raise ValueError(f"Unsupported modulation state source: {state_source}")
        if training_stage not in self.TRAINING_STAGES:
            raise ValueError(f"Unsupported modulation training stage: {training_stage}")
        if low_rank < 1 or expert_count < 2:
            raise ValueError("low_rank and expert_count must be positive/non-trivial")
        super().__init__(
            baseline,
            weather_encoder,
            state_encoder,
            d_model=d_model,
            forecast_steps=forecast_steps,
            trajectory_layers=trajectory_layers,
            dropout=dropout,
            conditioning="weather",
            fusion="blackbox",
        )
        self.modulation_mode = modulation_mode
        self.modulation_state_source = state_source
        self.modulation_training_stage = training_stage
        self.use_reference_state = bool(use_reference_state)
        self.d_model = int(d_model)
        self.low_rank = int(low_rank)
        self.expert_count = int(expert_count)
        self.register_buffer("modulation_amplitude", torch.tensor(float(amplitude)))
        self.register_buffer("expert_coefficient", torch.tensor(float(expert_coefficient)))
        self.register_buffer("current_state_reference", torch.zeros(1))
        self.register_buffer("latent_state_reference", torch.zeros(d_model))
        self.register_buffer("state_reference_calibrated", torch.tensor(False))

        # Both adapters exist in every candidate. Current-vs-latent comparisons
        # therefore have identical parameter counts.
        self.current_state_adapter = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.latent_state_adapter = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        output_width = {
            "none": 1,
            "effect_scale": 1,
            "channel_scale": d_model,
            "low_rank": low_rank,
            "experts": expert_count,
            "ramp": 1,
        }[modulation_mode]
        self.modulation_controller = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_width),
        )
        # Identity at step zero while retaining non-zero first gradients.
        nn.init.zeros_(self.modulation_controller[-1].weight)
        nn.init.zeros_(self.modulation_controller[-1].bias)
        self.low_rank_down = nn.Linear(d_model, low_rank, bias=False)
        self.low_rank_up = nn.Linear(low_rank, d_model, bias=False)
        self.expert_down = nn.ModuleList(
            nn.Linear(d_model, low_rank, bias=False) for _ in range(expert_count)
        )
        self.expert_up = nn.ModuleList(
            nn.Linear(low_rank, 1, bias=False) for _ in range(expert_count)
        )
        self._base_loaded = False
        self.configure_training_stage(training_stage)

    def load_base_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load a complete same-seed R2-W solution into the nested model."""
        result = self.load_state_dict(state, strict=False)
        allowed = (
            "modulation_amplitude", "expert_coefficient", "current_state_reference",
            "latent_state_reference", "state_reference_calibrated",
            "current_state_adapter.", "latent_state_adapter.", "modulation_controller.",
            "low_rank_down.", "low_rank_up.", "expert_down.", "expert_up.",
        )
        unexpected = list(result.unexpected_keys)
        invalid_missing = [
            key for key in result.missing_keys if not any(key == prefix or key.startswith(prefix) for prefix in allowed)
        ]
        if unexpected or invalid_missing:
            raise ValueError(
                f"R2-W checkpoint is incompatible: missing={invalid_missing}, unexpected={unexpected}"
            )
        self._base_loaded = True
        self.configure_training_stage(self.modulation_training_stage)

    def set_state_reference(
        self, current: torch.Tensor | float, latent: torch.Tensor
    ) -> None:
        current_value = torch.as_tensor(current, dtype=self.current_state_reference.dtype).reshape(1)
        latent_value = torch.as_tensor(latent, dtype=self.latent_state_reference.dtype).reshape(-1)
        if latent_value.shape != self.latent_state_reference.shape:
            raise ValueError("History-latent reference has the wrong shape")
        if not torch.isfinite(current_value).all() or not torch.isfinite(latent_value).all():
            raise ValueError("State references must be finite")
        self.current_state_reference.copy_(current_value.to(self.current_state_reference))
        self.latent_state_reference.copy_(latent_value.to(self.latent_state_reference))
        self.state_reference_calibrated.fill_(True)

    def configure_training_stage(self, stage: str) -> None:
        if stage not in self.TRAINING_STAGES:
            raise ValueError(f"Unsupported modulation training stage: {stage}")
        self.modulation_training_stage = stage
        self.requires_grad_(False)
        for module in (
            self.current_state_adapter,
            self.latent_state_adapter,
            self.modulation_controller,
            self.low_rank_down,
            self.low_rank_up,
            self.expert_down,
            self.expert_up,
        ):
            module.requires_grad_(True)
        if stage == "joint":
            for module in (self.lead, self.fusion_network, self.trajectory, self.head):
                module.requires_grad_(True)
        # The anchor is always a frozen reference, including during evaluation.
        self.anchor_baseline.requires_grad_(False)

    def optimizer_parameter_groups(
        self, learning_rate: float, decoder_lr_multiplier: float
    ) -> list[dict[str, Any]]:
        decoder_ids = {
            id(parameter)
            for module in (self.lead, self.fusion_network, self.trajectory, self.head)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        adapter = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in decoder_ids
        ]
        decoder = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) in decoder_ids
        ]
        groups: list[dict[str, Any]] = []
        if adapter:
            groups.append({"params": adapter, "lr": float(learning_rate), "name": "modulator"})
        if decoder:
            groups.append({
                "params": decoder,
                "lr": float(learning_rate) * float(decoder_lr_multiplier),
                "name": "decoder",
            })
        return groups

    def train(self, mode: bool = True) -> AggregateStrictModulationModel:
        super().train(mode)
        # Frozen R2-W modules must stay in eval mode; otherwise dropout changes
        # the allegedly fixed base mapping during adapter-only training.
        if mode:
            frozen = [self.baseline, self.anchor_baseline, self.weather_encoder]
            if self.modulation_training_stage == "adapter":
                frozen.extend((self.lead, self.fusion_network, self.trajectory, self.head))
            for module in frozen:
                module.eval()
        return self

    def _decode_weather(self, weather_features: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
        state_zero = torch.zeros_like(weather_features)
        fused = self.fusion_network(torch.cat((state_zero, weather_features, lead), dim=-1))
        return self._decode(fused)

    def _state_pair(
        self, cluster_history: torch.Tensor, alternate_history: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_history = cluster_history if alternate_history is None else alternate_history
        if source_history.shape != cluster_history.shape:
            raise ValueError("correction_state_history must match cluster_history")
        baseline_latent = self.baseline.encode_latent(source_history).detach()
        current_raw = source_history[:, -1:].detach()
        latent_raw = baseline_latent
        current_reference = self.current_state_reference.reshape(1, 1).expand_as(current_raw)
        latent_reference = self.latent_state_reference.reshape(1, -1).expand_as(latent_raw)
        if self.modulation_state_source == "current":
            raw, reference = current_raw, current_reference
            adapted = self.current_state_adapter(raw)
            adapted_reference = self.current_state_adapter(reference)
        else:
            raw, reference = latent_raw, latent_reference
            adapted = self.latent_state_adapter(raw)
            adapted_reference = self.latent_state_adapter(reference)
        if self.use_reference_state:
            adapted = adapted_reference
            raw = reference
        return adapted, adapted_reference, raw

    def _centered_control(
        self, state: torch.Tensor, state_reference: torch.Tensor, lead: torch.Tensor
    ) -> torch.Tensor:
        expanded = state[:, None].expand(-1, lead.shape[1], -1)
        expanded_reference = state_reference[:, None].expand_as(expanded)
        actual = self.modulation_controller(torch.cat((expanded, lead), dim=-1))
        reference = self.modulation_controller(torch.cat((expanded_reference, lead), dim=-1))
        return self.modulation_amplitude.to(actual) * (torch.tanh(actual) - torch.tanh(reference))

    def forward(
        self,
        *,
        cluster_history: torch.Tensor,
        correction_state_history: torch.Tensor | None = None,
        weather: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
        weather_latent_override: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        baseline_output = self.baseline(cluster_history=cluster_history)
        baseline = baseline_output["cluster_factor"]
        with torch.no_grad():
            anchor = self.anchor_baseline(cluster_history=cluster_history)["cluster_factor"]
        weather_output = self.weather_encoder(weather, weather_batch_index, weather_time_offset_hours)
        weather_features = weather_output["weather_features"]
        assert weather_features is not None
        if weather_latent_override is not None:
            if weather_latent_override.shape != weather_features.shape:
                raise ValueError("weather_latent_override has the wrong shape")
            weather_features = weather_latent_override
        lead = self.lead(len(cluster_history), dtype=cluster_history.dtype, device=cluster_history.device)
        state, state_reference, raw_state = self._state_pair(
            cluster_history, correction_state_history
        )
        control = self._centered_control(state, state_reference, lead)
        zero_weather = torch.zeros_like(weather_features)
        lead_only = self._decode_weather(zero_weather, lead)
        base_correction = self._decode_weather(weather_features, lead)
        weather_effect = base_correction - lead_only

        mode = self.modulation_mode
        if mode == "none":
            correction = base_correction
        elif mode == "effect_scale":
            correction = lead_only + (1.0 + control.squeeze(-1)) * weather_effect
        elif mode == "channel_scale":
            modulated_weather = weather_features * (1.0 + control)
            correction = self._decode_weather(modulated_weather, lead)
        elif mode == "low_rank":
            projected = self.low_rank_down(weather_features)
            delta = self.low_rank_up(projected * control)
            correction = self._decode_weather(weather_features + delta, lead)
        elif mode == "experts":
            # Centered logits make the reference-state routing exactly uniform.
            routing = torch.softmax(control / self.modulation_amplitude.to(control), dim=-1)
            centered = routing - 1.0 / self.expert_count
            responses = torch.stack(
                [up(torch.tanh(down(weather_features))).squeeze(-1)
                 for down, up in zip(self.expert_down, self.expert_up)],
                dim=-1,
            )
            expert_delta = (centered * responses).sum(dim=-1)
            correction = base_correction + self.expert_coefficient.to(expert_delta) * expert_delta
        else:
            difference = torch.cat(
                (weather_effect[:, :1], weather_effect[:, 1:] - weather_effect[:, :-1]), dim=1
            )
            scale = 1.0 + control.squeeze(-1)
            scale = torch.cat((torch.ones_like(scale[:, :1]), scale[:, 1:]), dim=1)
            modulated_difference = difference * scale
            modulated_effect = torch.cumsum(modulated_difference, dim=1)
            correction = lead_only + modulated_effect

        prediction = (baseline + correction).clamp_min(0.0)
        return {
            "cluster_factor": prediction,
            "baseline_cluster_factor": baseline,
            "anchor_baseline_cluster_factor": anchor,
            "weather_correction_factor": correction,
            "base_weather_correction_factor": base_correction,
            "lead_only_correction_factor": lead_only,
            "weather_effect_factor": weather_effect,
            "modulation_control": control,
            "modulation_state": raw_state,
            "weather_features": weather_features,
            "state_attention": None,
            "weather_attention": weather_output["weather_attention"],
            "spatial_attention": weather_output["spatial_attention"],
        }


class AggregateFusionModel(nn.Module):
    MODES = {
        "weather_only", "early_concat", "target_concat", "lead_gate",
        "prediction_blend", "cross_attention", "bidirectional", "tft",
        "astafn", "timexer",
    }

    def __init__(
        self,
        weather_encoder: AggregateWeatherEncoder,
        *,
        mode: str,
        state_mode: str,
        total_capacity_mw: float,
        d_model: int,
        heads: int,
        dropout: float,
        forecast_steps: int,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported aggregate fusion mode: {mode}")
        self.mode = mode
        self.weather_encoder = weather_encoder
        # A genuine weather-only baseline must not instantiate or execute any
        # history representation.  The other fusion baselines share exactly
        # the same aggregate-history encoder selected on validation.
        self.history_encoder = (
            None
            if mode == "weather_only"
            else AggregateStateEncoder(state_mode, d_model, heads, dropout, forecast_steps)
        )
        self.lead = LeadEmbedding(d_model, forecast_steps)
        self.forecast_steps = forecast_steps
        self.register_buffer("total_capacity_mw", torch.tensor(float(total_capacity_mw)))
        self.response = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model),
        )
        self.history_branch = (
            nn.Linear(2 * d_model, d_model)
            if mode in {"lead_gate", "prediction_blend", "astafn"} else None
        )
        self.weather_branch = (
            nn.Linear(2 * d_model, d_model)
            if mode in {"lead_gate", "prediction_blend"} else None
        )
        self.gate = (
            nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            if mode in {"lead_gate", "prediction_blend", "astafn"} else None
        )
        self.history_reads_weather = (
            nn.MultiheadAttention(d_model, _heads(d_model, heads), dropout=dropout, batch_first=True)
            if mode in {"cross_attention", "bidirectional", "timexer"} else None
        )
        self.weather_reads_history = (
            nn.MultiheadAttention(d_model, _heads(d_model, heads), dropout=dropout, batch_first=True)
            if mode == "bidirectional" else None
        )
        if mode in {"tft", "timexer"}:
            layer = nn.TransformerEncoderLayer(
                d_model, _heads(d_model, heads), 2 * d_model, dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.joint_encoder = nn.TransformerEncoder(layer, 2, nn.LayerNorm(d_model))
            self.source_embedding = nn.Parameter(torch.randn(2, d_model) * 0.02)
        else:
            self.joint_encoder = None
            self.source_embedding = None
        self.trajectory = nn.Sequential(
            *[CausalResidualBlock(d_model, 2**layer_index, dropout) for layer_index in range(3)]
        )
        self.head = nn.Conv1d(d_model, 1, 1)
        self.weather_base = nn.Parameter(torch.tensor(-1.5))
        self.history_base = (
            nn.Parameter(torch.tensor(-1.5)) if mode != "weather_only" else None
        )
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def _decode(self, representation: torch.Tensor) -> torch.Tensor:
        return self.head(self.trajectory(representation.transpose(1, 2))).squeeze(1)

    def forward(
        self,
        *,
        cluster_history: torch.Tensor,
        weather: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
        **_: Any,
    ) -> dict[str, Any]:
        weather_output = self.weather_encoder(weather, weather_batch_index, weather_time_offset_hours)
        weather_features = weather_output["weather_features"]
        assert weather_features is not None
        lead = self.lead(len(cluster_history), dtype=cluster_history.dtype, device=cluster_history.device)
        mode = self.mode
        history = torch.zeros_like(weather_features)
        history_attention = None
        if mode != "weather_only":
            assert self.history_encoder is not None
            history, history_attention = self.history_encoder(cluster_history)
        if mode == "weather_only":
            representation = self.response(torch.cat((history, weather_features, lead), dim=-1))
            prediction = F.softplus(self.weather_base + self._decode(representation))
        elif mode == "early_concat":
            assert self.history_base is not None
            h = history.mean(dim=1, keepdim=True).expand_as(history)
            w = weather_features.mean(dim=1, keepdim=True).expand_as(weather_features)
            representation = self.response(torch.cat((h, w, lead), dim=-1))
            prediction = F.softplus(self.history_base + self._decode(representation))
        elif mode == "target_concat":
            assert self.history_base is not None
            representation = self.response(torch.cat((history, weather_features, lead), dim=-1))
            prediction = F.softplus(self.history_base + self._decode(representation))
        elif mode in {"lead_gate", "prediction_blend"}:
            assert self.history_branch is not None and self.weather_branch is not None
            assert self.gate is not None and self.history_base is not None
            h = self.history_branch(torch.cat((history, lead), dim=-1))
            w = self.weather_branch(torch.cat((weather_features, lead), dim=-1))
            gate = torch.sigmoid(self.gate(torch.cat((h, w, lead), dim=-1)))
            if mode == "lead_gate":
                representation = self.response(torch.cat((gate * h + (1 - gate) * w, lead, lead), dim=-1))
                prediction = F.softplus(self.history_base + self._decode(representation))
            else:
                history_delta = self._decode(h)
                weather_delta = self._decode(w)
                scalar_gate = gate.mean(dim=-1)
                history_prediction = F.softplus(self.history_base + history_delta)
                weather_prediction = F.softplus(self.weather_base + weather_delta)
                prediction = scalar_gate * history_prediction + (1 - scalar_gate) * weather_prediction
                representation = gate * h + (1 - gate) * w
        elif mode in {"cross_attention", "bidirectional"}:
            assert self.history_reads_weather is not None and self.history_base is not None
            h_read, _ = self.history_reads_weather(history, weather_features, weather_features, need_weights=False)
            if mode == "bidirectional":
                assert self.weather_reads_history is not None
                w_read, _ = self.weather_reads_history(weather_features, history, history, need_weights=False)
            else:
                w_read = weather_features
            representation = self.response(torch.cat((h_read, w_read, lead), dim=-1))
            prediction = F.softplus(self.history_base + self._decode(representation))
        elif mode in {"tft", "timexer"}:
            assert self.joint_encoder is not None and self.source_embedding is not None
            assert self.history_base is not None
            combined = torch.cat(
                (history + self.source_embedding[0], weather_features + self.source_embedding[1]), dim=1
            )
            encoded = self.joint_encoder(combined)
            if mode == "timexer":
                assert self.history_reads_weather is not None
                h_read, _ = self.history_reads_weather(
                    encoded[:, :24], encoded[:, 24:], encoded[:, 24:], need_weights=False
                )
                representation = self.response(torch.cat((h_read, encoded[:, 24:], lead), dim=-1))
            else:
                representation = self.response(torch.cat((encoded[:, :24], encoded[:, 24:], lead), dim=-1))
            prediction = F.softplus(self.history_base + self._decode(representation))
        else:
            assert self.history_branch is not None and self.gate is not None
            assert self.history_base is not None
            adapted = torch.tanh(self.history_branch(torch.cat((history, lead), dim=-1)))
            gate = torch.sigmoid(self.gate(torch.cat((history, weather_features, lead), dim=-1)))
            representation = self.response(torch.cat((gate * adapted + (1 - gate) * history, weather_features, lead), dim=-1))
            prediction = F.softplus(self.history_base + self._decode(representation))
        return {
            "cluster_factor": prediction,
            "baseline_cluster_factor": None,
            "anchor_baseline_cluster_factor": None,
            "weather_features": weather_features,
            "state_attention": history_attention,
            "weather_attention": weather_output["weather_attention"],
            "spatial_attention": weather_output["spatial_attention"],
        }


class ResponseStateEncoder(nn.Module):
    """Small parameter-matched encoder for six native forcing-realization nodes."""

    def __init__(self, input_width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_width, hidden, 1)
        self.blocks = nn.Sequential(
            CausalResidualBlock(hidden, 1, dropout),
            CausalResidualBlock(hidden, 2, dropout),
        )
        self.output = nn.LayerNorm(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("Response tokens must have shape [batch,time,features]")
        encoded = self.blocks(self.input(tokens.transpose(1, 2))).transpose(1, 2)
        return self.output(encoded.mean(dim=1))


class AggregateResponseStateModel(AggregateStrictModulationModel):
    """Identify a response state from causal historical forecast-realization pairs."""

    RESPONSE_INPUTS = {
        "power_only", "weather_only", "level_pair", "transition_pair", "transition_pair_id"
    }

    def __init__(
        self,
        baseline: AggregateHistoryForecast,
        weather_encoder: AggregateWeatherEncoder,
        state_encoder: AggregateStateEncoder,
        *,
        d_model: int,
        forecast_steps: int,
        trajectory_layers: int,
        dropout: float,
        response_input: str,
        training_stage: str,
        response_hidden: int = 32,
        low_rank: int = 4,
        amplitude: float = 0.25,
    ) -> None:
        if response_input not in self.RESPONSE_INPUTS:
            raise ValueError(f"Unsupported response-state input: {response_input}")
        super().__init__(
            baseline,
            weather_encoder,
            state_encoder,
            d_model=d_model,
            forecast_steps=forecast_steps,
            trajectory_layers=trajectory_layers,
            dropout=dropout,
            modulation_mode="low_rank",
            state_source="baseline_latent",
            training_stage=training_stage,
            use_reference_state=False,
            amplitude=amplitude,
            low_rank=low_rank,
            expert_count=3,
            expert_coefficient=0.1,
        )
        self.response_input = response_input
        self.response_hidden = int(response_hidden)
        self.response_token_width = 2 * int(d_model) + 4
        self.response_state_encoder = ResponseStateEncoder(
            self.response_token_width, self.response_hidden, dropout
        )
        self.response_controller = nn.Sequential(
            nn.Linear(self.response_hidden, d_model),
            nn.GELU(),
            nn.Linear(d_model, low_rank),
        )
        nn.init.zeros_(self.response_controller[-1].weight)
        nn.init.zeros_(self.response_controller[-1].bias)
        self.identification_query = nn.Linear(2 * d_model, d_model, bias=False)
        self.identification_head = nn.Linear(d_model, 1, bias=False)
        reference_count = 6 if response_input == "level_pair" else 5
        self.register_buffer(
            "response_reference_tokens",
            torch.zeros(reference_count, self.response_token_width),
        )
        self.register_buffer("response_reference_calibrated", torch.tensor(False))
        self.response_use_reference_state = False
        self.configure_training_stage(training_stage)

    def load_base_state(self, state: Mapping[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state, strict=False)
        allowed = (
            "modulation_amplitude", "expert_coefficient", "current_state_reference",
            "latent_state_reference", "state_reference_calibrated",
            "current_state_adapter.", "latent_state_adapter.", "modulation_controller.",
            "low_rank_down.", "low_rank_up.", "expert_down.", "expert_up.",
            "response_state_encoder.", "response_controller.", "identification_query.",
            "identification_head.", "response_reference_tokens",
            "response_reference_calibrated",
        )
        invalid = [
            key for key in result.missing_keys
            if not any(key == prefix or key.startswith(prefix) for prefix in allowed)
        ]
        if result.unexpected_keys or invalid:
            raise ValueError(
                f"R2-W checkpoint is incompatible with response state: "
                f"missing={invalid}, unexpected={result.unexpected_keys}"
            )
        self._base_loaded = True
        self.configure_training_stage(self.modulation_training_stage)

    def configure_training_stage(self, stage: str) -> None:
        if not hasattr(self, "response_state_encoder"):
            AggregateStrictModulationModel.configure_training_stage(self, stage)
            return
        if stage not in self.TRAINING_STAGES:
            raise ValueError(f"Unsupported response-state training stage: {stage}")
        self.modulation_training_stage = stage
        self.requires_grad_(False)
        for module in (
            self.response_state_encoder,
            self.response_controller,
            self.low_rank_down,
            self.low_rank_up,
            self.identification_query,
            self.identification_head,
        ):
            module.requires_grad_(True)
        if stage == "joint":
            for module in (self.lead, self.fusion_network, self.trajectory, self.head):
                module.requires_grad_(True)
        self.anchor_baseline.requires_grad_(False)

    def train(self, mode: bool = True) -> AggregateResponseStateModel:
        nn.Module.train(self, mode)
        if mode:
            frozen = [self.baseline, self.anchor_baseline, self.weather_encoder, self.state_encoder]
            if self.modulation_training_stage == "adapter":
                frozen.extend((self.lead, self.fusion_network, self.trajectory, self.head))
            for module in frozen:
                module.eval()
        return self

    def optimizer_parameter_groups(
        self, learning_rate: float, decoder_lr_multiplier: float
    ) -> list[dict[str, Any]]:
        decoder_ids = {
            id(parameter)
            for module in (self.lead, self.fusion_network, self.trajectory, self.head)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        response = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in decoder_ids
        ]
        decoder = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) in decoder_ids
        ]
        groups: list[dict[str, Any]] = []
        if response:
            groups.append({"params": response, "lr": float(learning_rate), "name": "response"})
        if decoder:
            groups.append(
                {
                    "params": decoder,
                    "lr": float(learning_rate) * float(decoder_lr_multiplier),
                    "name": "decoder",
                }
            )
        return groups

    def build_response_tokens(
        self,
        past_weather: torch.Tensor,
        past_power: torch.Tensor,
        past_time: torch.Tensor,
    ) -> torch.Tensor:
        if past_weather.shape[1:] != (6, self.d_model):
            raise ValueError("past_hourly_weather_latent must have shape [batch,6,d_model]")
        if past_power.shape[1:] != (6,) or past_time.shape[1:] != (6,):
            raise ValueError("Historical response power/time must have six nodes")
        weather_delta = past_weather[:, 1:] - past_weather[:, :-1]
        power_delta = past_power[:, 1:] - past_power[:, :-1]
        lead_age = torch.arange(1, 7, device=past_weather.device, dtype=past_weather.dtype)
        lead_age = (lead_age / 6.0).reshape(1, 6, 1).expand(len(past_weather), -1, -1)
        if self.response_input == "level_pair":
            weather_level = past_weather
            delta = torch.zeros_like(past_weather)
            power_level = past_power[..., None]
            power_change = torch.zeros_like(power_level)
            time = past_time[..., None] / 6.0
        else:
            weather_level = past_weather[:, :-1]
            delta = weather_delta
            power_level = past_power[:, :-1, None]
            power_change = power_delta[..., None]
            time = past_time[:, 1:, None] / 6.0
            lead_age = lead_age[:, 1:]
            if self.response_input == "power_only":
                weather_level = torch.zeros_like(weather_level)
                delta = torch.zeros_like(delta)
            elif self.response_input == "weather_only":
                power_level = torch.zeros_like(power_level)
                power_change = torch.zeros_like(power_change)
        return torch.cat(
            (weather_level, delta, power_level, power_change, time, lead_age), dim=-1
        )

    @torch.no_grad()
    def set_response_reference_tokens(self, values: torch.Tensor) -> None:
        if values.shape != self.response_reference_tokens.shape:
            raise ValueError("Response reference tokens have the wrong shape")
        if not torch.isfinite(values).all():
            raise ValueError("Response reference tokens must be finite")
        self.response_reference_tokens.copy_(values.to(self.response_reference_tokens))
        self.response_reference_calibrated.fill_(True)

    def _response_state_pair(
        self, tokens: torch.Tensor, *, support_only: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actual_tokens = tokens[:, :3] if support_only else tokens
        reference_tokens = self.response_reference_tokens[: actual_tokens.shape[1]]
        reference_tokens = reference_tokens.to(tokens).unsqueeze(0).expand(len(tokens), -1, -1)
        actual = self.response_state_encoder(actual_tokens)
        reference = self.response_state_encoder(reference_tokens)
        if self.response_use_reference_state:
            actual = reference
        return actual, reference

    def _response_control(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        actual = self.response_controller(state)
        center = self.response_controller(reference)
        return self.modulation_amplitude.to(actual) * (torch.tanh(actual) - torch.tanh(center))

    def _low_rank_modulate(self, weather: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        projected = self.low_rank_down(weather)
        if control.ndim == 2:
            control = control[:, None]
        return weather + self.low_rank_up(projected * control)

    def forward(
        self,
        *,
        cluster_history: torch.Tensor,
        future_hourly_weather_latent: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
        past_hourly_weather_latent: torch.Tensor,
        past_cluster_power: torch.Tensor,
        past_weather_time_offset_hours: torch.Tensor,
        response_past_weather_override: torch.Tensor | None = None,
        response_past_power_override: torch.Tensor | None = None,
        response_pair_shift: bool = False,
        response_support_only: bool = False,
        future_weather_latent_override: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        baseline_output = self.baseline(cluster_history=cluster_history)
        baseline = baseline_output["cluster_factor"]
        with torch.no_grad():
            anchor = self.anchor_baseline(cluster_history=cluster_history)["cluster_factor"]
        weather_output = self.weather_encoder.forward_from_hourly(
            future_hourly_weather_latent,
            weather_batch_index,
            weather_time_offset_hours,
        )
        weather_features = weather_output["weather_features"]
        assert weather_features is not None
        if future_weather_latent_override is not None:
            if future_weather_latent_override.shape != weather_features.shape:
                raise ValueError("future_weather_latent_override has the wrong shape")
            weather_features = future_weather_latent_override

        past_weather = (
            past_hourly_weather_latent
            if response_past_weather_override is None
            else response_past_weather_override
        )
        past_power = past_cluster_power if response_past_power_override is None else response_past_power_override
        if past_weather.shape != past_hourly_weather_latent.shape or past_power.shape != past_cluster_power.shape:
            raise ValueError("Response-state intervention shape mismatch")
        if response_pair_shift:
            past_power = torch.cat((past_power[:, :1], past_power[:, :-1]), dim=1)
        tokens = self.build_response_tokens(
            past_weather, past_power, past_weather_time_offset_hours
        )
        response_state, response_reference = self._response_state_pair(
            tokens, support_only=response_support_only
        )
        control = self._response_control(response_state, response_reference)
        modulated_weather = self._low_rank_modulate(weather_features, control)
        correction = self._decode_weather(modulated_weather, self.lead(
            len(cluster_history), dtype=cluster_history.dtype, device=cluster_history.device
        ))

        identification_prediction: torch.Tensor | None = None
        identification_target: torch.Tensor | None = None
        if self.response_input == "transition_pair_id":
            support_state, support_reference = self._response_state_pair(tokens, support_only=True)
            support_control = self._response_control(support_state, support_reference)
            query_weather = torch.cat(
                (
                    past_weather[:, 3:5],
                    past_weather[:, 4:6] - past_weather[:, 3:5],
                ),
                dim=-1,
            )
            query = self.identification_query(query_weather)
            query = self._low_rank_modulate(query, support_control)
            identification_prediction = self.identification_head(query).squeeze(-1)
            identification_target = past_power[:, 4:6] - past_power[:, 3:5]

        prediction = (baseline + correction).clamp_min(0.0)
        return {
            "cluster_factor": prediction,
            "baseline_cluster_factor": baseline,
            "anchor_baseline_cluster_factor": anchor,
            "weather_correction_factor": correction,
            "weather_features": weather_features,
            "state_attention": None,
            "weather_attention": None,
            "spatial_attention": None,
            "response_state": response_state,
            "response_control": control,
            "response_tokens": tokens,
            "identification_prediction": identification_prediction,
            "identification_target": identification_target,
        }


class AggregateWeatherDynamicsModel(AggregateCorrectionModel):
    """Structured weather-driven aggregate-power mechanisms on frozen R2 latents."""

    MODES = {
        "weather_mean", "weather_envelope",
        "m1_baseline_headroom", "m2_anomaly_headroom", "m3_recursive_headroom",
        "m4_fixed_equilibrium", "m4_adaptive_equilibrium",
        "m5_envelope_realization", "m6_fixed_error_decay", "m6_adaptive_error_decay",
        "m4_no_recurrence",
    }
    TRAINING_STAGES = {"head", "joint"}

    def __init__(
        self,
        baseline: AggregateHistoryForecast,
        weather_encoder: AggregateWeatherEncoder,
        state_encoder: AggregateStateEncoder,
        *,
        d_model: int,
        forecast_steps: int,
        trajectory_layers: int,
        dropout: float,
        dynamics_mode: str,
        training_stage: str,
        rate_min_per_hour: float = 1.0 / 12.0,
        rate_max_per_hour: float = 4.0,
        initial_rate_per_hour: float = 0.5,
        realization_max: float = 1.2,
    ) -> None:
        if dynamics_mode not in self.MODES:
            raise ValueError(f"Unsupported weather dynamics mode: {dynamics_mode}")
        if training_stage not in self.TRAINING_STAGES:
            raise ValueError(f"Unsupported weather dynamics training stage: {training_stage}")
        if not 0 < rate_min_per_hour < initial_rate_per_hour < rate_max_per_hour:
            raise ValueError("Dynamics rates must satisfy min < initial < max")
        super().__init__(
            baseline, weather_encoder, state_encoder, d_model=d_model,
            forecast_steps=forecast_steps, trajectory_layers=trajectory_layers,
            dropout=dropout, conditioning="weather", fusion="blackbox",
        )
        self.dynamics_mode = dynamics_mode
        self.dynamics_training_stage = training_stage
        self.d_model = int(d_model)
        self.register_buffer("rate_min_per_hour", torch.tensor(float(rate_min_per_hour)))
        self.register_buffer("rate_max_per_hour", torch.tensor(float(rate_max_per_hour)))
        self.register_buffer("realization_max", torch.tensor(float(realization_max)))
        rate_fraction = (
            (float(initial_rate_per_hour) - float(rate_min_per_hour))
            / (float(rate_max_per_hour) - float(rate_min_per_hour))
        )
        initial_logit = math.log(rate_fraction / (1.0 - rate_fraction))
        self.fixed_rate_logit = nn.Parameter(torch.tensor(initial_logit))
        self.signal_head = nn.Conv1d(d_model, 1, 1)
        self.equilibrium_head = nn.Conv1d(d_model, 1, 1)
        self.rate_head = nn.Conv1d(d_model, 1, 1)
        self.absolute_head = nn.Conv1d(d_model, 1, 1)
        self.absolute_mapper = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.realization_input = nn.Conv1d(2, d_model, 1)
        self.realization_trajectory = nn.Sequential(
            CausalResidualBlock(d_model, 1, dropout),
            CausalResidualBlock(d_model, 2, dropout),
        )
        self.realization_summary = nn.Linear(d_model, d_model)
        self.realization_head = nn.Conv1d(d_model, 1, 1)
        for module in (self.signal_head, self.equilibrium_head, self.rate_head, self.absolute_head):
            nn.init.normal_(module.weight, std=1e-3)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.rate_head.weight)
        nn.init.constant_(self.rate_head.bias, initial_logit)
        nn.init.zeros_(self.realization_head.weight)
        nn.init.zeros_(self.realization_head.bias)
        self._base_loaded = False
        self.register_buffer("auxiliary_loaded", torch.tensor(False))
        self.configure_training_stage(training_stage)

    def load_base_state(self, state: Mapping[str, torch.Tensor]) -> None:
        result = self.load_state_dict(state, strict=False)
        prefixes = (
            "rate_min_per_hour", "rate_max_per_hour", "realization_max", "fixed_rate_logit",
            "signal_head.", "equilibrium_head.", "rate_head.", "absolute_head.",
            "absolute_mapper.",
            "realization_input.", "realization_trajectory.", "realization_summary.",
            "realization_head.", "auxiliary_loaded",
        )
        invalid = [
            key for key in result.missing_keys
            if not any(key == prefix or key.startswith(prefix) for prefix in prefixes)
        ]
        if result.unexpected_keys or invalid:
            raise ValueError(
                f"R2-W checkpoint is incompatible with dynamics model: "
                f"missing={invalid}, unexpected={result.unexpected_keys}"
            )
        self._base_loaded = True
        self.configure_training_stage(self.dynamics_training_stage)

    def load_auxiliary_state(self, state: Mapping[str, torch.Tensor]) -> None:
        required = ("absolute_mapper.", "absolute_head.")
        own = self.state_dict()
        copied = 0
        for key, value in state.items():
            if key.startswith(required):
                if key not in own or own[key].shape != value.shape:
                    raise ValueError(f"Auxiliary weather checkpoint tensor mismatch: {key}")
                own[key].copy_(value)
                copied += 1
        if copied == 0:
            raise ValueError("Auxiliary checkpoint contains no weather absolute predictor")
        self.auxiliary_loaded.fill_(True)
        self.configure_training_stage(self.dynamics_training_stage)

    def configure_training_stage(self, stage: str) -> None:
        if stage not in self.TRAINING_STAGES:
            raise ValueError(f"Unsupported dynamics training stage: {stage}")
        self.dynamics_training_stage = stage
        self.requires_grad_(False)
        mode = self.dynamics_mode
        if mode in {"weather_mean", "weather_envelope"}:
            heads: tuple[nn.Module, ...] = (self.absolute_mapper, self.absolute_head)
        elif mode.startswith("m1_") or mode.startswith("m2_") or mode.startswith("m3_"):
            heads = (self.signal_head,)
        elif mode in {"m4_fixed_equilibrium", "m4_no_recurrence"}:
            heads = (self.equilibrium_head,)
            if mode == "m4_fixed_equilibrium":
                self.fixed_rate_logit.requires_grad_(True)
        elif mode == "m4_adaptive_equilibrium":
            heads = (self.equilibrium_head, self.rate_head)
        elif mode == "m5_envelope_realization":
            heads = (
                self.realization_input, self.realization_trajectory,
                self.realization_summary, self.realization_head,
            )
        elif mode == "m6_fixed_error_decay":
            heads = ()
            self.fixed_rate_logit.requires_grad_(True)
        else:
            heads = (self.rate_head,)
        for module in heads:
            module.requires_grad_(True)
        if stage == "joint" and mode not in {
            "weather_mean", "weather_envelope", "m5_envelope_realization",
            "m6_fixed_error_decay", "m6_adaptive_error_decay"
        }:
            for module in (self.lead, self.fusion_network, self.trajectory):
                module.requires_grad_(True)
        # M5/M6 preserve the separately trained definition of A(W).
        self.baseline.requires_grad_(False)
        self.anchor_baseline.requires_grad_(False)
        self.weather_encoder.requires_grad_(False)

    def optimizer_parameter_groups(
        self, learning_rate: float, decoder_lr_multiplier: float
    ) -> list[dict[str, Any]]:
        decoder_ids = {
            id(parameter)
            for module in (self.lead, self.fusion_network, self.trajectory)
            for parameter in module.parameters() if parameter.requires_grad
        }
        head = [p for p in self.parameters() if p.requires_grad and id(p) not in decoder_ids]
        decoder = [p for p in self.parameters() if p.requires_grad and id(p) in decoder_ids]
        groups: list[dict[str, Any]] = []
        if head:
            groups.append({"params": head, "lr": float(learning_rate), "name": "structured_head"})
        if decoder:
            groups.append({
                "params": decoder,
                "lr": float(learning_rate) * float(decoder_lr_multiplier),
                "name": "trajectory_decoder",
            })
        return groups

    def train(self, mode: bool = True) -> AggregateWeatherDynamicsModel:
        super().train(mode)
        if mode:
            self.baseline.eval()
            self.anchor_baseline.eval()
            self.weather_encoder.eval()
            if self.dynamics_training_stage == "head":
                self.lead.eval()
                self.fusion_network.eval()
                self.trajectory.eval()
        return self

    def _lead_at(self, hours: torch.Tensor) -> torch.Tensor:
        angle = hours * (2.0 * math.pi / 24.0)
        values = torch.stack(
            (hours / 6.0, torch.sin(angle), torch.cos(angle), torch.sin(4 * angle), torch.cos(4 * angle)),
            dim=-1,
        )
        return self.lead.network(values)

    def _context(self, weather: torch.Tensor, lead: torch.Tensor, *, temporal: bool = True) -> torch.Tensor:
        fused = self.fusion_network(torch.cat((torch.zeros_like(weather), weather, lead), dim=-1))
        return (
            self.trajectory(fused.transpose(1, 2)).transpose(1, 2)
            if temporal else fused
        )

    @staticmethod
    def _bounded_displacement(anchor: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        anchor = anchor.clamp(0.0, 1.0)
        return anchor + signal.clamp_min(0.0) * (1.0 - anchor) - (-signal).clamp_min(0.0) * anchor

    def _rate(self, logits: torch.Tensor) -> torch.Tensor:
        return self.rate_min_per_hour.to(logits) + (
            self.rate_max_per_hour.to(logits) - self.rate_min_per_hour.to(logits)
        ) * torch.sigmoid(logits)

    def forward(
        self,
        *,
        cluster_history: torch.Tensor,
        future_hourly_weather_latent: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
        current_weather_latent: torch.Tensor,
        past_realization_factor: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        batch = len(cluster_history)
        weather_output = self.weather_encoder.forward_from_hourly(
            future_hourly_weather_latent, weather_batch_index, weather_time_offset_hours
        )
        weather = weather_output["weather_features"]
        assert weather is not None
        current_weather = current_weather_latent + self.weather_encoder.budget_adapter(current_weather_latent)
        lead = self.lead(batch, dtype=weather.dtype, device=weather.device)
        context = self._context(weather, lead)
        zero_hours = torch.zeros(batch, 1, dtype=weather.dtype, device=weather.device)
        current_lead = self._lead_at(zero_hours)
        current_context = self._context(current_weather[:, None], current_lead, temporal=False)
        absolute_context = self.absolute_mapper(weather)
        current_absolute_context = self.absolute_mapper(current_weather[:, None])
        raw_baseline = self.baseline(cluster_history=cluster_history)["cluster_factor"]
        baseline = raw_baseline.clamp(0.0, 1.0)
        p0 = cluster_history[:, -1].clamp(0.0, 1.0)
        mode = self.dynamics_mode
        diagnostics: dict[str, Any] = {}

        if mode in {"weather_mean", "weather_envelope"}:
            prediction = torch.sigmoid(self.absolute_head(absolute_context.transpose(1, 2)).squeeze(1))
            current_prediction = torch.sigmoid(
                self.absolute_head(current_absolute_context.transpose(1, 2)).squeeze(1)
            )
            diagnostics["current_weather_prediction_factor"] = current_prediction
        elif mode == "m1_baseline_headroom":
            signal = torch.tanh(self.signal_head(context.transpose(1, 2)).squeeze(1))
            prediction = self._bounded_displacement(baseline, signal)
            diagnostics["dynamics_signal"] = signal
        elif mode == "m2_anomaly_headroom":
            anomaly = weather - current_weather[:, None]
            anomaly_context = self._context(anomaly, lead)
            signal = torch.tanh(self.signal_head(anomaly_context.transpose(1, 2)).squeeze(1))
            prediction = self._bounded_displacement(p0[:, None], signal)
            diagnostics["dynamics_signal"] = signal
        elif mode == "m3_recursive_headroom":
            signal = torch.tanh(self.signal_head(context.transpose(1, 2)).squeeze(1))
            values = []
            state = p0
            for index in range(self.forecast_steps):
                state = self._bounded_displacement(state, signal[:, index])
                values.append(state)
            prediction = torch.stack(values, dim=1)
            diagnostics["dynamics_signal"] = signal
        elif mode in {"m4_fixed_equilibrium", "m4_adaptive_equilibrium", "m4_no_recurrence"}:
            equilibrium = torch.sigmoid(
                self.equilibrium_head(context.transpose(1, 2)).squeeze(1)
            )
            if mode == "m4_adaptive_equilibrium":
                rate = self._rate(self.rate_head(context.transpose(1, 2)).squeeze(1))
            else:
                fixed = self._rate(self.fixed_rate_logit).to(equilibrium)
                rate = fixed.reshape(1, 1).expand_as(equilibrium)
            if mode == "m4_no_recurrence":
                prediction = equilibrium
            else:
                values = []
                state = p0
                for index in range(self.forecast_steps):
                    target = equilibrium[:, index]
                    state = target + (state - target) * torch.exp(-rate[:, index] * 0.25)
                    values.append(state)
                prediction = torch.stack(values, dim=1)
            diagnostics.update({"equilibrium_factor": equilibrium, "response_rate_per_hour": rate})
        elif mode == "m5_envelope_realization":
            if not bool(self.auxiliary_loaded):
                raise RuntimeError("M5 requires a frozen envelope checkpoint")
            if past_realization_factor is None or past_realization_factor.shape[1:] != (12,):
                raise ValueError("M5 requires 12 past realization factors")
            envelope = torch.sigmoid(self.absolute_head(absolute_context.transpose(1, 2)).squeeze(1))
            delta = torch.zeros_like(past_realization_factor)
            delta[:, 1:] = past_realization_factor[:, 1:] - past_realization_factor[:, :-1]
            eta_features = torch.stack((past_realization_factor, delta), dim=1)
            encoded = self.realization_trajectory(self.realization_input(eta_features))
            summary = self.realization_summary(encoded[:, :, -1])[:, :, None].expand(-1, -1, self.forecast_steps)
            eta = self.realization_max.to(envelope) * torch.sigmoid(
                self.realization_head(summary).squeeze(1)
            )
            prediction = (envelope * eta).clamp(0.0, 1.0)
            diagnostics.update({"weather_envelope_factor": envelope, "realization_factor": eta})
        else:
            if not bool(self.auxiliary_loaded):
                raise RuntimeError("M6 requires a frozen weather-mean checkpoint")
            absolute = torch.sigmoid(self.absolute_head(absolute_context.transpose(1, 2)).squeeze(1))
            current_absolute = torch.sigmoid(
                self.absolute_head(current_absolute_context.transpose(1, 2)).squeeze(1)
            ).squeeze(1)
            if mode == "m6_adaptive_error_decay":
                rate = self._rate(self.rate_head(context.transpose(1, 2)).squeeze(1))
            else:
                fixed = self._rate(self.fixed_rate_logit).to(absolute)
                rate = fixed.reshape(1, 1).expand_as(absolute)
            hours = self.lead.lead_hours.to(absolute)[None]
            persistence = torch.exp(-rate * hours)
            discrepancy = p0 - current_absolute
            prediction = (absolute + persistence * discrepancy[:, None]).clamp(0.0, 1.0)
            diagnostics.update({
                "weather_absolute_factor": absolute,
                "current_weather_prediction_factor": current_absolute[:, None],
                "current_discrepancy_factor": discrepancy,
                "response_rate_per_hour": rate,
                "discrepancy_persistence": persistence,
            })
        return {
            "cluster_factor": prediction,
            "baseline_cluster_factor": baseline,
            "anchor_baseline_cluster_factor": None,
            "weather_features": weather,
            "weather_attention": None,
            "spatial_attention": None,
            "state_attention": None,
            **diagnostics,
        }


WEATHER_MODE_MAP = {
    "r1_raw_interpolation": "raw_interpolation",
    "r2_latent_interpolation": "latent_interpolation",
    "r3_full_path": "full_path",
    "r4_target_value": "target_value",
    "r4_no_time": "target_value_no_time",
    "r5_dynamic_pool": "dynamic_pool",
    "r6_regional_tokens": "regional_tokens",
    "r7_patch_regional": "patch_regional",
    "r8_joint_spacetime": "joint_spacetime",
    "r8_factorized": "factorized_spacetime",
}


FUSION_MODE_MAP = {
    "w_only": "weather_only",
    "f_early_concat": "early_concat",
    "f_target_concat": "target_concat",
    "f_lead_gate": "lead_gate",
    "f_prediction_blend": "prediction_blend",
    "f_cross_attention": "cross_attention",
    "f_bidirectional": "bidirectional",
    "f_tft": "tft",
    "f_astafn": "astafn",
    "f_timexer": "timexer",
}


def build_aggregate_model(
    config: Mapping[str, Any],
    static: Mapping[str, Any],
    *,
    total_capacity_mw: float,
    history_steps: int = 24,
    forecast_steps: int = 24,
) -> nn.Module:
    architecture = str(config["architecture"])
    d_model = int(config.get("d_model", 64))
    layers = int(config.get("history_layers", 3))
    heads = int(config.get("attention_heads", 4))
    dropout = float(config.get("dropout", 0.05))
    visible = int(config.get("history_visible_steps", history_steps))
    if architecture.startswith("hb_"):
        mode = architecture.removeprefix("hb_")
        return AggregateHistoryForecast(
            mode, total_capacity_mw=total_capacity_mw, history_steps=history_steps,
            forecast_steps=forecast_steps, visible_steps=visible, d_model=d_model,
            layers=layers, attention_heads=heads, dropout=dropout,
        )
    weather_name = str(config.get("weather_route", "r3_full_path"))
    weather = AggregateWeatherEncoder(
        WEATHER_MODE_MAP[weather_name], channels=int(config.get("weather_channels", 14)),
        d_model=d_model, heads=heads, dropout=dropout, forecast_steps=forecast_steps,
        capacity_density=static["capacity_density"],
    )
    state_mode = str(config.get("state_mode", "recent_1h"))
    if architecture in FUSION_MODE_MAP:
        fusion_state_mode = str(config.get("fusion_history_state_mode", ""))
        if not fusion_state_mode:
            fusion_state_mode = {
                1: "current", 4: "recent_1h", 12: "recent_3h", 24: "full_6h"
            }.get(visible, "full_6h")
        return AggregateFusionModel(
            weather, mode=FUSION_MODE_MAP[architecture], state_mode=fusion_state_mode,
            total_capacity_mw=total_capacity_mw, d_model=d_model, heads=heads,
            dropout=dropout, forecast_steps=forecast_steps,
        )
    if architecture not in {
        "aggregate_correction", "aggregate_strict_modulation", "aggregate_response_state",
        "aggregate_weather_dynamics",
    }:
        raise ValueError(f"Unsupported aggregate architecture: {architecture}")
    baseline = AggregateHistoryForecast(
        "tcn", total_capacity_mw=total_capacity_mw, history_steps=history_steps,
        forecast_steps=forecast_steps, visible_steps=visible, d_model=d_model,
        layers=layers, attention_heads=heads, dropout=float(config.get("baseline_dropout", 0.1)),
    )
    state = AggregateStateEncoder(state_mode, d_model, heads, dropout, forecast_steps)
    if architecture == "aggregate_weather_dynamics":
        if WEATHER_MODE_MAP[weather_name] != "latent_interpolation":
            raise ValueError("Weather dynamics is frozen to the R2 latent-interpolation route")
        return AggregateWeatherDynamicsModel(
            baseline, weather, state, d_model=d_model, forecast_steps=forecast_steps,
            trajectory_layers=int(config.get("trajectory_layers", 3)), dropout=dropout,
            dynamics_mode=str(config.get("weather_dynamics_mode", "m4_adaptive_equilibrium")),
            training_stage=str(config.get("dynamics_training_stage", "head")),
            rate_min_per_hour=float(config.get("dynamics_rate_min_per_hour", 1.0 / 12.0)),
            rate_max_per_hour=float(config.get("dynamics_rate_max_per_hour", 4.0)),
            initial_rate_per_hour=float(config.get("dynamics_initial_rate_per_hour", 0.5)),
            realization_max=float(config.get("realization_factor_max", 1.2)),
        )
    if architecture == "aggregate_strict_modulation":
        if WEATHER_MODE_MAP[weather_name] != "latent_interpolation":
            raise ValueError("Strict modulation is frozen to the R2 latent-interpolation weather route")
        return AggregateStrictModulationModel(
            baseline, weather, state, d_model=d_model, forecast_steps=forecast_steps,
            trajectory_layers=int(config.get("trajectory_layers", 3)), dropout=dropout,
            modulation_mode=str(config.get("strict_modulation_mode", "channel_scale")),
            state_source=str(config.get("modulation_state_source", "baseline_latent")),
            training_stage=str(config.get("modulation_training_stage", "adapter")),
            use_reference_state=bool(config.get("modulation_use_reference_state", False)),
            amplitude=float(config.get("modulation_amplitude", 0.25)),
            low_rank=int(config.get("modulation_low_rank", 4)),
            expert_count=int(config.get("modulation_expert_count", 3)),
            expert_coefficient=float(config.get("modulation_expert_coefficient", 0.1)),
        )
    if architecture == "aggregate_response_state":
        if WEATHER_MODE_MAP[weather_name] != "latent_interpolation":
            raise ValueError("Response-state identification is frozen to the R2 weather route")
        return AggregateResponseStateModel(
            baseline,
            weather,
            state,
            d_model=d_model,
            forecast_steps=forecast_steps,
            trajectory_layers=int(config.get("trajectory_layers", 3)),
            dropout=dropout,
            response_input=str(config.get("response_state_input", "transition_pair")),
            training_stage=str(config.get("modulation_training_stage", "adapter")),
            response_hidden=int(config.get("response_state_hidden", 32)),
            low_rank=int(config.get("modulation_low_rank", 4)),
            amplitude=float(config.get("modulation_amplitude", 0.25)),
        )
    return AggregateCorrectionModel(
        baseline, weather, state, d_model=d_model, forecast_steps=forecast_steps,
        trajectory_layers=int(config.get("trajectory_layers", 3)), dropout=dropout,
        conditioning=str(config.get("correction_conditioning", "state_weather")),
        fusion=str(config.get("correction_fusion", "blackbox")),
    )
