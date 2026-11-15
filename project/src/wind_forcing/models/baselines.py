from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from wind_forcing.models.components import (
    CausalResidualBlock,
    inverse_softplus,
)
from wind_forcing.models.forced_evolution import StationForcingEncoder


class ForecastModelBase(nn.Module):
    def __init__(
        self,
        station_capacity_mw: Sequence[float],
        forecast_steps: int,
    ) -> None:
        super().__init__()
        capacity = torch.as_tensor(station_capacity_mw, dtype=torch.float32)
        self.register_buffer("capacity_mw", capacity)
        self.register_buffer("total_capacity_mw", capacity.sum())
        self.forecast_steps = int(forecast_steps)

    def hierarchical_output(self, station_factor: torch.Tensor) -> dict[str, Any]:
        cluster_factor = (
            station_factor * self.capacity_mw.reshape(1, -1, 1)
        ).sum(dim=1) / self.total_capacity_mw
        return {
            "station_factor": station_factor,
            "cluster_factor": cluster_factor,
            "hourly_station_factor_unique": None,
            "spatial_attention_unique": None,
        }


class PersistenceForecastModel(ForecastModelBase):
    def __init__(self, station_capacity_mw: Sequence[float], forecast_steps: int) -> None:
        super().__init__(station_capacity_mw, forecast_steps)
        self.zero_gradient_anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *, station_history: torch.Tensor, **_: Any) -> dict[str, Any]:
        prediction = station_history[:, :, -1:].expand(
            -1, -1, self.forecast_steps
        )
        prediction = prediction + self.zero_gradient_anchor * 0.0
        return self.hierarchical_output(prediction)


class DLinearHead(nn.Module):
    def __init__(self, history_steps: int, forecast_steps: int) -> None:
        super().__init__()
        self.trend = nn.Linear(history_steps, forecast_steps)
        self.seasonal = nn.Linear(history_steps, forecast_steps)
        nn.init.zeros_(self.trend.weight)
        nn.init.zeros_(self.trend.bias)
        nn.init.zeros_(self.seasonal.weight)
        nn.init.zeros_(self.seasonal.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        padded = F.pad(history.unsqueeze(1), (2, 2), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=5, stride=1).squeeze(1)
        seasonal = history - trend
        return self.trend(trend) + self.seasonal(seasonal)


class HistoryForecastModel(ForecastModelBase):
    MODES = {
        "history_linear",
        "history_dlinear",
        "history_lstm",
        "history_tcn",
        "history_transformer",
        "history_patchtst",
        "history_tft",
        "history_itransformer",
    }

    def __init__(
        self,
        mode: str,
        station_capacity_mw: Sequence[float],
        history_steps: int,
        forecast_steps: int,
        d_model: int,
        layers: int,
        attention_heads: int,
        dropout: float,
        output_level: str = "hierarchical",
    ) -> None:
        super().__init__(station_capacity_mw, forecast_steps)
        if mode not in self.MODES:
            raise ValueError(f"Unsupported history baseline: {mode}")
        self.mode = mode
        if output_level not in {"hierarchical", "direct_cluster"}:
            raise ValueError("output_level must be hierarchical or direct_cluster")
        self.output_level = output_level
        self.history_steps = int(history_steps)
        self.d_model = int(d_model)
        self.position = nn.Parameter(torch.zeros(1, history_steps, d_model))
        nn.init.normal_(self.position, std=0.02)

        if mode == "history_linear":
            self.encoder: nn.Module = nn.Linear(history_steps, forecast_steps)
            nn.init.zeros_(self.encoder.weight)
            nn.init.zeros_(self.encoder.bias)
        elif mode == "history_dlinear":
            self.encoder = DLinearHead(history_steps, forecast_steps)
        elif mode == "history_lstm":
            self.input_projection = nn.Linear(1, d_model)
            self.encoder = nn.LSTM(
                d_model,
                d_model,
                num_layers=max(1, layers),
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(d_model, forecast_steps)
        elif mode == "history_tcn":
            self.input_projection = nn.Conv1d(1, d_model, kernel_size=1)
            self.encoder = nn.Sequential(
                *[
                    CausalResidualBlock(d_model, 2**layer, dropout)
                    for layer in range(layers)
                ]
            )
            self.head = nn.Linear(d_model, forecast_steps)
        elif mode in {"history_transformer", "history_patchtst"}:
            patch = 4 if mode == "history_patchtst" else 1
            self.patch_length = patch
            self.patch_stride = 2 if patch > 1 else 1
            self.input_projection = nn.Linear(patch, d_model)
            token_count = (
                1 + (history_steps - patch) // self.patch_stride
                if patch > 1
                else history_steps
            )
            self.patch_position = nn.Parameter(torch.zeros(1, token_count, d_model))
            nn.init.normal_(self.patch_position, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=max(1, layers), norm=nn.LayerNorm(d_model)
            )
            self.head = nn.Linear(d_model, forecast_steps)
        elif mode == "history_tft":
            self.input_projection = nn.Linear(3, d_model)
            self.encoder = nn.LSTM(
                d_model,
                d_model,
                num_layers=max(1, layers),
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
            self.tft_attention = nn.MultiheadAttention(
                d_model, attention_heads, dropout=dropout, batch_first=True
            )
            self.tft_gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            self.tft_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, forecast_steps)
        else:
            self.station_embedding = nn.Parameter(
                torch.zeros(1, len(self.capacity_mw), d_model)
            )
            nn.init.normal_(self.station_embedding, std=0.02)
            self.input_projection = nn.Linear(history_steps, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=max(1, layers), norm=nn.LayerNorm(d_model)
            )
            self.head = nn.Linear(d_model, forecast_steps)

    def _shared_delta(self, history: torch.Tensor) -> torch.Tensor:
        batch, stations, steps = history.shape
        flat = history.reshape(batch * stations, steps)
        if self.mode in {"history_linear", "history_dlinear"}:
            return self.encoder(flat).reshape(batch, stations, self.forecast_steps)
        if self.mode == "history_lstm":
            tokens = self.input_projection(flat.unsqueeze(-1)) + self.position
            encoded, _ = self.encoder(tokens)
            delta = self.head(encoded[:, -1])
        elif self.mode == "history_tcn":
            encoded = self.encoder(self.input_projection(flat.unsqueeze(1)))
            delta = self.head(encoded[:, :, -1])
        elif self.mode in {"history_transformer", "history_patchtst"}:
            if self.patch_length > 1:
                patches = flat.unfold(
                    -1, self.patch_length, self.patch_stride
                )
            else:
                patches = flat.unsqueeze(-1)
            tokens = self.input_projection(patches) + self.patch_position
            delta = self.head(self.encoder(tokens).mean(dim=1))
        else:
            time = torch.linspace(-1.0, 0.0, steps, device=flat.device)
            time_features = torch.stack((time.sin(), time.cos()), dim=-1)
            time_features = time_features.unsqueeze(0).expand(len(flat), -1, -1)
            tokens = self.input_projection(
                torch.cat((flat.unsqueeze(-1), time_features.to(flat)), dim=-1)
            )
            encoded, _ = self.encoder(tokens)
            attended, _ = self.tft_attention(encoded, encoded, encoded, need_weights=False)
            gate = self.tft_gate(torch.cat((encoded, attended), dim=-1))
            fused = self.tft_norm(encoded + gate * attended)
            delta = self.head(fused[:, -1])
        return delta.reshape(batch, stations, self.forecast_steps)

    def forward(
        self,
        *,
        station_history: torch.Tensor,
        cluster_history: torch.Tensor,
        **_: Any,
    ) -> dict[str, Any]:
        if self.output_level == "direct_cluster":
            if self.mode == "history_itransformer":
                raise ValueError("Direct-cluster iTransformer is not defined")
            delta = self._shared_delta(cluster_history[:, None])[:, 0]
            cluster_factor = F.softplus(
                inverse_softplus(cluster_history[:, -1:].clamp_min(1e-4)) + delta
            )
            return {
                "station_factor": None,
                "cluster_factor": cluster_factor,
                "hourly_station_factor_unique": None,
                "spatial_attention_unique": None,
            }
        if self.mode == "history_itransformer":
            tokens = self.input_projection(station_history) + self.station_embedding
            delta = self.head(self.encoder(tokens))
        else:
            delta = self._shared_delta(station_history)
        base = inverse_softplus(station_history[:, :, -1:].clamp_min(1e-4))
        return self.hierarchical_output(F.softplus(base + delta))


class ResidualFrameBackbone(nn.Module):
    def __init__(self, channels: int, d_model: int) -> None:
        super().__init__()
        hidden = max(16, d_model // 2)
        self.input = nn.Conv2d(channels, hidden, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.GroupNorm(4, hidden), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(4, hidden), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(4, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        batch, steps = weather.shape[:2]
        values = self.input(weather.flatten(0, 1))
        values = values + self.block1(values)
        values = values + self.block2(values)
        pooled = values.mean(dim=(-2, -1))
        return self.output(pooled).reshape(batch, steps, -1)


class Conv3DBackbone(nn.Module):
    def __init__(self, channels: int, d_model: int) -> None:
        super().__init__()
        hidden = max(16, d_model // 2)
        self.network = nn.Sequential(
            nn.Conv3d(channels, hidden, 3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
            nn.Conv3d(hidden, d_model, 3, padding=1),
            nn.GroupNorm(4, d_model),
            nn.GELU(),
        )

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        values = self.network(weather.permute(0, 2, 1, 3, 4))
        return values.mean(dim=(-2, -1)).transpose(1, 2)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1
        )

    def forward(
        self, values: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        input_gate, forget_gate, output_gate, proposal = self.gates(
            torch.cat((values, hidden), dim=1)
        ).chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(proposal)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTMBackbone(nn.Module):
    def __init__(self, channels: int, d_model: int) -> None:
        super().__init__()
        hidden = max(16, d_model // 2)
        self.input = nn.Conv2d(channels, hidden, 3, stride=2, padding=1)
        self.cell = ConvLSTMCell(hidden, hidden)
        self.output = nn.Linear(hidden, d_model)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        batch = weather.shape[0]
        first = self.input(weather[:, 0])
        hidden = torch.zeros_like(first)
        cell = torch.zeros_like(first)
        outputs = []
        for step in range(weather.shape[1]):
            encoded = first if step == 0 else self.input(weather[:, step])
            hidden, cell = self.cell(encoded, (hidden, cell))
            outputs.append(self.output(hidden.mean(dim=(-2, -1))))
        return torch.stack(outputs, dim=1)


class TorchvisionFrameBackbone(nn.Module):
    def __init__(self, mode: str, channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.mode = mode
        if mode == "weather_vit":
            from torchvision.models.vision_transformer import VisionTransformer

            self.network = VisionTransformer(
                image_size=32,
                patch_size=4,
                num_layers=4,
                num_heads=4,
                hidden_dim=d_model,
                mlp_dim=4 * d_model,
                dropout=dropout,
                num_classes=d_model,
            )
            self.network.conv_proj = nn.Conv2d(
                channels, d_model, kernel_size=4, stride=4
            )
        elif mode == "weather_swin":
            from torchvision.models.swin_transformer import SwinTransformer

            self.network = SwinTransformer(
                patch_size=[2, 2],
                embed_dim=32,
                depths=[1, 1, 2],
                num_heads=[2, 4, 8],
                window_size=[3, 3],
                dropout=dropout,
                stochastic_depth_prob=0.1,
                num_classes=d_model,
            )
            self.network.features[0][0] = nn.Conv2d(
                channels, 32, kernel_size=2, stride=2
            )
        else:
            raise ValueError(mode)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = weather.shape
        frames = weather.reshape(batch * steps, channels, height, width)
        if self.mode == "weather_vit":
            frames = F.pad(frames, (0, 32 - width, 0, 32 - height))
        return self.network(frames).reshape(batch, steps, -1)


class TorchvisionVideoSwinBackbone(nn.Module):
    def __init__(self, channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        from torchvision.models.video.swin_transformer import SwinTransformer3d

        self.network = SwinTransformer3d(
            patch_size=[2, 2, 2],
            embed_dim=32,
            depths=[1, 1, 2],
            num_heads=[2, 4, 8],
            window_size=[2, 3, 3],
            dropout=dropout,
            stochastic_depth_prob=0.1,
            num_classes=d_model,
        )
        self.network.patch_embed.proj = nn.Conv3d(
            channels, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        encoded = self.network(weather.permute(0, 2, 1, 3, 4))
        return encoded[:, None].expand(-1, weather.shape[1], -1)


class WeatherForecastModel(ForecastModelBase):
    MODES = {
        "weather_nearest",
        "weather_bilinear",
        "weather_resnet2d",
        "weather_3dcnn",
        "weather_convlstm",
        "weather_vit",
        "weather_swin",
        "weather_video_swin",
    }

    def __init__(
        self,
        mode: str,
        model_config: Mapping[str, Any],
        static: Mapping[str, Sequence[float]],
        forecast_steps: int,
    ) -> None:
        super().__init__(static["station_capacity_mw"], forecast_steps)
        if mode not in self.MODES:
            raise ValueError(f"Unsupported weather baseline: {mode}")
        self.mode = mode
        self.output_level = str(model_config.get("output_level", "direct_cluster"))
        if self.output_level not in {"hierarchical", "direct_cluster"}:
            raise ValueError("output_level must be hierarchical or direct_cluster")
        channels = int(model_config["weather_channels"])
        d_model = int(model_config["d_model"])
        dropout = float(model_config["dropout"])
        self.point_forcing: StationForcingEncoder | None = None
        if mode in {"weather_nearest", "weather_bilinear"}:
            query_mode = "nearest" if mode == "weather_nearest" else "bilinear"
            self.point_forcing = StationForcingEncoder(
                nwp_channels=channels,
                weather_hidden_channels=int(model_config["weather_hidden_channels"]),
                d_model=d_model,
                weather_latitude=static["weather_latitude"],
                weather_longitude=static["weather_longitude"],
                station_latitude=static["station_latitude"],
                station_longitude=static["station_longitude"],
                station_capacity_mw=static["station_capacity_mw"],
                nwp_mean=static["nwp_mean"],
                nwp_std=static["nwp_std"],
                station_query_mode=query_mode,
                station_neighborhood_cells=1,
                include_wind_geometry=False,
                include_region_context=False,
                forcing_path_mode="independent",
                attention_heads=int(model_config["attention_heads"]),
                dropout=dropout,
            )
            self.backbone: nn.Module | None = None
        elif mode == "weather_resnet2d":
            self.backbone = ResidualFrameBackbone(channels, d_model)
        elif mode == "weather_3dcnn":
            self.backbone = Conv3DBackbone(channels, d_model)
        elif mode == "weather_convlstm":
            self.backbone = ConvLSTMBackbone(channels, d_model)
        elif mode in {"weather_vit", "weather_swin"}:
            self.backbone = TorchvisionFrameBackbone(mode, channels, d_model, dropout)
        else:
            self.backbone = TorchvisionVideoSwinBackbone(channels, d_model, dropout)

        static_values = torch.stack(
            (
                torch.as_tensor(static["station_latitude"], dtype=torch.float32),
                torch.as_tensor(static["station_longitude"], dtype=torch.float32),
                self.capacity_mw / self.total_capacity_mw,
            ),
            dim=-1,
        )
        static_values = (static_values - static_values.mean(dim=0)) / static_values.std(
            dim=0
        ).clamp_min(1e-6)
        self.register_buffer("station_static", static_values)
        self.station_encoder = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.phase_embedding = nn.Embedding(4, d_model)
        self.station_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, forecast_steps),
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, forecast_steps),
        )
        for head in (self.station_head[-1], self.cluster_head[-1]):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, -1.5)

    def forward(
        self,
        *,
        weather: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
        origin_phase: torch.Tensor,
        **_: Any,
    ) -> dict[str, Any]:
        auxiliary = None
        if self.point_forcing is not None:
            point = self.point_forcing(weather)
            cycle_tokens = point["forcing"]
            auxiliary = point["hourly_station_factor"]
            tokens = cycle_tokens.index_select(0, weather_batch_index)
        else:
            assert self.backbone is not None
            global_cycle = self.backbone(weather)
            global_tokens = global_cycle.index_select(0, weather_batch_index)
            tokens = global_tokens[:, None].expand(-1, len(self.capacity_mw), -1, -1)
        time_features = torch.stack(
            (
                weather_time_offset_hours / 6.0,
                weather_time_offset_hours.abs() / 6.0,
            ),
            dim=-1,
        )
        tokens = tokens + self.time_encoder(time_features)[:, None]
        pooled = tokens.mean(dim=2)
        pooled = pooled + self.station_encoder(self.station_static)[None]
        pooled = pooled + self.phase_embedding(origin_phase)[:, None]
        if self.output_level == "hierarchical":
            station_factor = F.softplus(self.station_head(pooled))
            output = self.hierarchical_output(station_factor)
        else:
            station_weight = self.capacity_mw / self.total_capacity_mw
            cluster_token = (pooled * station_weight.reshape(1, -1, 1)).sum(dim=1)
            output = {
                "station_factor": None,
                "cluster_factor": F.softplus(self.cluster_head(cluster_token)),
                "hourly_station_factor_unique": None,
                "spatial_attention_unique": None,
            }
        output["hourly_station_factor_unique"] = auxiliary
        return output


def build_history_baseline(
    model_config: Mapping[str, Any],
    static: Mapping[str, Sequence[float]],
    history_steps: int,
    forecast_steps: int,
) -> ForecastModelBase:
    architecture = str(model_config["architecture"])
    if architecture == "history_persistence":
        return PersistenceForecastModel(static["station_capacity_mw"], forecast_steps)
    return HistoryForecastModel(
        architecture,
        static["station_capacity_mw"],
        history_steps,
        forecast_steps,
        int(model_config["d_model"]),
        int(model_config["history_tcn_layers"]),
        int(model_config["attention_heads"]),
        float(model_config["dropout"]),
        str(model_config.get("output_level", "hierarchical")),
    )


def build_weather_baseline(
    model_config: Mapping[str, Any],
    static: Mapping[str, Sequence[float]],
    forecast_steps: int,
) -> WeatherForecastModel:
    return WeatherForecastModel(
        str(model_config["architecture"]), model_config, static, forecast_steps
    )
