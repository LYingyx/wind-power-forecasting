from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


WIND_VECTOR_PAIRS = ((1, 2), (10, 12), (11, 13))


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormActivation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class WeatherFrameEncoder(nn.Module):
    """Encode every native grid point independently without spatial leakage."""

    def __init__(
        self,
        nwp_channels: int,
        hidden_channels: int,
        d_model: int,
        nwp_mean: Sequence[float],
        nwp_std: Sequence[float],
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(nwp_mean, dtype=torch.float32)
        std = torch.as_tensor(nwp_std, dtype=torch.float32)
        if mean.shape != (nwp_channels,) or std.shape != mean.shape or torch.any(std <= 0):
            raise ValueError("NWP normalization metadata is inconsistent")
        self.nwp_channels = nwp_channels
        self.register_buffer("nwp_mean", mean.reshape(1, 1, -1, 1, 1))
        self.register_buffer("nwp_std", std.reshape(1, 1, -1, 1, 1))
        self.network = nn.Sequential(
            nn.Linear(nwp_channels + len(WIND_VECTOR_PAIRS), hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def physical(self, weather: torch.Tensor) -> torch.Tensor:
        return weather * self.nwp_std.to(weather) + self.nwp_mean.to(weather)

    def encode_points(self, weather: torch.Tensor) -> torch.Tensor:
        """Encode standardized weather vectors stored along the final axis."""
        if weather.ndim < 2 or weather.shape[-1] != self.nwp_channels:
            raise ValueError(
                f"Expected point weather [...,{self.nwp_channels}], got {tuple(weather.shape)}"
            )
        broadcast_shape = (1,) * (weather.ndim - 1) + (self.nwp_channels,)
        mean = self.nwp_mean.reshape(broadcast_shape).to(weather)
        std = self.nwp_std.reshape(broadcast_shape).to(weather)
        physical = weather * std + mean
        speeds = torch.stack(
            [
                torch.log1p(
                    torch.sqrt(
                        physical[..., u].square()
                        + physical[..., v].square()
                        + 1e-8
                    )
                )
                for u, v in WIND_VECTOR_PAIRS
            ],
            dim=-1,
        )
        return self.network(torch.cat((weather, speeds), dim=-1))

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        if weather.ndim != 5 or weather.shape[2] != self.nwp_channels:
            raise ValueError(
                f"Expected weather [B,T,{self.nwp_channels},H,W], got {tuple(weather.shape)}"
            )
        encoded = self.encode_points(weather.permute(0, 1, 3, 4, 2))
        return encoded.permute(0, 1, 4, 2, 3)


class RegionalContextEncoder(nn.Module):
    """Optional CNN branch for weather-system context over the complete crop."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            ConvNormActivation(d_model, d_model),
            ConvNormActivation(d_model, d_model),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output = nn.LayerNorm(d_model)

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        if point_features.ndim != 5:
            raise ValueError("point_features must have shape [B,T,D,H,W]")
        batch, steps = point_features.shape[:2]
        encoded = self.network(point_features.flatten(0, 1))
        context = encoded.mean(dim=(-2, -1)).reshape(batch, steps, -1)
        return self.output(context)


def build_station_patch_metadata(
    weather_latitude: Sequence[float],
    weather_longitude: Sequence[float],
    station_latitude: Sequence[float],
    station_longitude: Sequence[float],
    query_mode: str,
    neighborhood_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if query_mode not in {"nearest", "bilinear", "fixed_gaussian", "learned_cells"}:
        raise ValueError(
            "station_query_mode must be nearest, bilinear, fixed_gaussian, or learned_cells"
        )
    if neighborhood_cells < 1 or neighborhood_cells % 2 != 1:
        raise ValueError("station_neighborhood_cells must be a positive odd integer")
    latitude = np.asarray(weather_latitude, dtype=np.float64)
    longitude = np.asarray(weather_longitude, dtype=np.float64)
    station_lat = np.asarray(station_latitude, dtype=np.float64)
    station_lon = np.asarray(station_longitude, dtype=np.float64)
    patch_indices: list[list[int]] = []
    geometry: list[list[list[float]]] = []
    fixed_weights: list[list[float]] = []
    grid_scale_km = 111.32 * float(abs(latitude[1] - latitude[0]))
    for farm_lat, farm_lon in zip(station_lat, station_lon):
        if query_mode == "nearest":
            rows = [int(np.argmin(np.abs(latitude - farm_lat)))]
            columns = [int(np.argmin(np.abs(longitude - farm_lon)))]
            weights = [1.0]
        else:
            north = int(np.searchsorted(-latitude, -farm_lat, side="right") - 1)
            west = int(np.searchsorted(longitude, farm_lon, side="right") - 1)
            if not (0 <= north < len(latitude) - 1 and 0 <= west < len(longitude) - 1):
                raise ValueError("A station lies outside the weather grid cells")
            cell_span = 1 if query_mode == "bilinear" else neighborhood_cells
            radius = cell_span // 2
            rows = list(range(north - radius, north + radius + 2))
            columns = list(range(west - radius, west + radius + 2))
            if query_mode == "bilinear":
                west_fraction = (farm_lon - longitude[west]) / (
                    longitude[west + 1] - longitude[west]
                )
                north_fraction = (farm_lat - latitude[north + 1]) / (
                    latitude[north] - latitude[north + 1]
                )
                weights = [
                    (1.0 - west_fraction) * north_fraction,
                    west_fraction * north_fraction,
                    (1.0 - west_fraction) * (1.0 - north_fraction),
                    west_fraction * (1.0 - north_fraction),
                ]
            elif query_mode == "fixed_gaussian":
                delta_y_km = 111.32 * float(abs(latitude[1] - latitude[0]))
                delta_x_km = (
                    111.32
                    * float(abs(longitude[1] - longitude[0]))
                    * math.cos(math.radians(float(farm_lat)))
                )
                sigma_km = math.sqrt(max(delta_x_km * delta_y_km, 1e-6))
                distances = []
                for row in rows:
                    for column in columns:
                        dx_km = (
                            (longitude[column] - farm_lon)
                            * 111.32
                            * math.cos(math.radians(float(farm_lat)))
                        )
                        dy_km = (latitude[row] - farm_lat) * 111.32
                        distances.append(math.sqrt(dx_km**2 + dy_km**2))
                kernel = np.exp(
                    -np.square(np.asarray(distances, dtype=np.float64))
                    / (2.0 * sigma_km**2)
                )
                weights = (kernel / kernel.sum()).tolist()
            else:
                weights = [1.0 / (len(rows) * len(columns))] * (
                    len(rows) * len(columns)
                )
        if min(rows) < 0 or max(rows) >= len(latitude) or min(columns) < 0 or max(
            columns
        ) >= len(longitude):
            raise ValueError(
                "The weather crop does not contain a complete station neighborhood"
            )
        indices: list[int] = []
        station_geometry: list[list[float]] = []
        for row in rows:
            for column in columns:
                dx_km = (
                    (longitude[column] - farm_lon)
                    * 111.32
                    * math.cos(math.radians(float(farm_lat)))
                )
                dy_km = (latitude[row] - farm_lat) * 111.32
                distance_km = math.sqrt(dx_km**2 + dy_km**2)
                indices.append(row * len(longitude) + column)
                station_geometry.append(
                    [
                        dx_km / grid_scale_km,
                        dy_km / grid_scale_km,
                        distance_km / grid_scale_km,
                        dx_km / max(distance_km, 1e-6),
                        dy_km / max(distance_km, 1e-6),
                    ]
                )
        patch_indices.append(indices)
        geometry.append(station_geometry)
        fixed_weights.append(weights)
    return (
        np.asarray(patch_indices, dtype=np.int64),
        np.asarray(geometry, dtype=np.float32),
        np.asarray(fixed_weights, dtype=np.float32),
    )


class WindConditionedStationQuery(nn.Module):
    """Learn a station-specific aggregation over native local grid tokens."""

    def __init__(
        self,
        d_model: int,
        weather_latitude: Sequence[float],
        weather_longitude: Sequence[float],
        station_latitude: Sequence[float],
        station_longitude: Sequence[float],
        station_capacity_mw: Sequence[float],
        nwp_mean: Sequence[float],
        nwp_std: Sequence[float],
        query_mode: str = "learned_cells",
        neighborhood_cells: int = 3,
        include_wind_geometry: bool = True,
        include_region_context: bool = False,
    ) -> None:
        super().__init__()
        patch_index, geometry, fixed_weights = build_station_patch_metadata(
            weather_latitude,
            weather_longitude,
            station_latitude,
            station_longitude,
            query_mode,
            neighborhood_cells,
        )
        if query_mode != "learned_cells" and include_wind_geometry:
            raise ValueError("Wind geometry applies only to learned_cells queries")
        station_lat = torch.as_tensor(station_latitude, dtype=torch.float32)
        station_lon = torch.as_tensor(station_longitude, dtype=torch.float32)
        capacity = torch.as_tensor(station_capacity_mw, dtype=torch.float32)
        lat_axis = torch.as_tensor(weather_latitude, dtype=torch.float32)
        lon_axis = torch.as_tensor(weather_longitude, dtype=torch.float32)
        lat_center = 0.5 * (lat_axis.min() + lat_axis.max())
        lon_center = 0.5 * (lon_axis.min() + lon_axis.max())
        static = torch.stack(
            (
                (station_lat - lat_center) / (lat_axis.max() - lat_axis.min()),
                (station_lon - lon_center) / (lon_axis.max() - lon_axis.min()),
                capacity / capacity.sum(),
                torch.log(capacity) / torch.log(capacity).mean(),
            ),
            dim=-1,
        )
        self.d_model = d_model
        self.query_mode = query_mode
        self.include_wind_geometry = include_wind_geometry
        self.include_region_context = include_region_context
        self.register_buffer("patch_index", torch.from_numpy(patch_index))
        self.register_buffer("relative_geometry", torch.from_numpy(geometry))
        self.register_buffer("fixed_spatial_weights", torch.from_numpy(fixed_weights))
        self.register_buffer("station_static", static)
        mean = torch.as_tensor(nwp_mean, dtype=torch.float32)
        std = torch.as_tensor(nwp_std, dtype=torch.float32)
        self.register_buffer("u10_mean", mean[1])
        self.register_buffer("u10_std", std[1])
        self.register_buffer("v10_mean", mean[2])
        self.register_buffer("v10_std", std[2])

        self.static_encoder = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        if query_mode == "learned_cells":
            geometry_dim = 7 if include_wind_geometry else 3
            self.geometry_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(geometry_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.key_projection: nn.Module | None = nn.Linear(d_model, d_model)
            self.attention_score: nn.Module | None = nn.Sequential(
                nn.GELU(), nn.Linear(d_model, 1)
            )
        else:
            self.geometry_encoder = None
            self.key_projection = None
            self.attention_score = None
        fusion_inputs = 3 if include_region_context else 2
        self.output = nn.Sequential(
            nn.Linear(fusion_inputs * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    @property
    def station_count(self) -> int:
        return int(self.patch_index.shape[0])

    @property
    def patch_token_count(self) -> int:
        return int(self.patch_index.shape[1])

    def station_embedding(self) -> torch.Tensor:
        return self.static_encoder(self.station_static)

    def fuse_station_features(
        self,
        local: torch.Tensor,
        region_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse pre-aggregated station features with static station context."""
        if local.ndim != 4 or local.shape[2:] != (self.station_count, self.d_model):
            raise ValueError("local station features have an invalid shape")
        batch, steps = local.shape[:2]
        station_context = self.station_embedding().reshape(
            1, 1, self.station_count, self.d_model
        ).expand(batch, steps, -1, -1)
        fusion = [local, station_context]
        if self.include_region_context:
            if region_context is None or region_context.shape != (
                batch,
                steps,
                self.d_model,
            ):
                raise ValueError("Regional context is missing or has an invalid shape")
            fusion.append(
                region_context.unsqueeze(2).expand(-1, -1, self.station_count, -1)
            )
        return self.output(torch.cat(fusion, dim=-1))

    def forward(
        self,
        encoded_map: torch.Tensor,
        standardized_weather: torch.Tensor,
        region_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoded_map.ndim != 5 or encoded_map.shape[2] != self.d_model:
            raise ValueError("encoded_map has an invalid shape")
        batch, steps, _, height, width = encoded_map.shape
        if standardized_weather.shape[:2] != (batch, steps) or tuple(
            standardized_weather.shape[-2:]
        ) != (height, width):
            raise ValueError("Encoded and raw weather grids are inconsistent")
        flat_indices = self.patch_index.flatten()
        feature_flat = encoded_map.flatten(-2)
        tokens = feature_flat[:, :, :, flat_indices]
        tokens = tokens.reshape(
            batch,
            steps,
            self.d_model,
            self.station_count,
            self.patch_token_count,
        ).permute(0, 1, 3, 4, 2)

        station_token = self.station_embedding().reshape(
            1, 1, self.station_count, 1, self.d_model
        )
        if self.query_mode == "learned_cells":
            if self.geometry_encoder is None or self.key_projection is None or self.attention_score is None:
                raise RuntimeError("Learned spatial-query modules are missing")
            static_geometry = self.relative_geometry[..., :3]
            if self.include_wind_geometry:
                weather_flat = standardized_weather.flatten(-2)
                u = weather_flat[:, :, 1, flat_indices]
                v = weather_flat[:, :, 2, flat_indices]
                u = u * self.u10_std.to(u) + self.u10_mean.to(u)
                v = v * self.v10_std.to(v) + self.v10_mean.to(v)
                u = u.reshape(batch, steps, self.station_count, self.patch_token_count)
                v = v.reshape(batch, steps, self.station_count, self.patch_token_count)
                unit_x = self.relative_geometry[..., 3].to(u)
                unit_y = self.relative_geometry[..., 4].to(u)
                along = (u * unit_x + v * unit_y) / 25.0
                cross = (u * unit_y - v * unit_x) / 25.0
                geometry = torch.cat(
                    (
                        static_geometry.to(u).reshape(
                            1, 1, self.station_count, self.patch_token_count, 3
                        ).expand(batch, steps, -1, -1, -1),
                        along.unsqueeze(-1),
                        cross.unsqueeze(-1),
                        (u / 25.0).unsqueeze(-1),
                        (v / 25.0).unsqueeze(-1),
                    ),
                    dim=-1,
                )
            else:
                geometry = static_geometry.to(tokens).reshape(
                    1, 1, self.station_count, self.patch_token_count, 3
                ).expand(batch, steps, -1, -1, -1)
            geometry_token = self.geometry_encoder(geometry)
            logits = self.attention_score(
                self.key_projection(tokens) + geometry_token + station_token
            ).squeeze(-1)
            weights = torch.softmax(logits, dim=-1)
        else:
            weights = self.fixed_spatial_weights.to(tokens).reshape(
                1, 1, self.station_count, self.patch_token_count
            ).expand(batch, steps, -1, -1)
        local = (tokens * weights.unsqueeze(-1)).sum(dim=-2)
        return self.fuse_station_features(local, region_context), weights


def _coarsen_axis(values: Sequence[float], pool_size: int) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float32)
    if pool_size <= 1:
        raise ValueError("pool_size must be greater than one")
    return np.asarray(
        [axis[start : start + pool_size].mean() for start in range(0, len(axis), pool_size)],
        dtype=np.float32,
    )


class MultiScaleWindConditionedStationQuery(nn.Module):
    """Fuse station-conditioned native-grid and coarse-grid weather queries."""

    def __init__(
        self,
        d_model: int,
        weather_latitude: Sequence[float],
        weather_longitude: Sequence[float],
        station_latitude: Sequence[float],
        station_longitude: Sequence[float],
        station_capacity_mw: Sequence[float],
        nwp_mean: Sequence[float],
        nwp_std: Sequence[float],
        query_mode: str,
        neighborhood_cells: int,
        include_wind_geometry: bool,
        include_region_context: bool,
        regional_pool_size: int,
        regional_neighborhood_cells: int,
    ) -> None:
        super().__init__()
        if regional_pool_size <= 1:
            raise ValueError("regional_pool_size must be greater than one")
        self.d_model = int(d_model)
        self.regional_pool_size = int(regional_pool_size)
        common = {
            "d_model": d_model,
            "station_latitude": station_latitude,
            "station_longitude": station_longitude,
            "station_capacity_mw": station_capacity_mw,
            "nwp_mean": nwp_mean,
            "nwp_std": nwp_std,
            "query_mode": query_mode,
            "include_wind_geometry": include_wind_geometry,
            "include_region_context": include_region_context,
        }
        self.local_query = WindConditionedStationQuery(
            weather_latitude=weather_latitude,
            weather_longitude=weather_longitude,
            neighborhood_cells=neighborhood_cells,
            **common,
        )
        self.regional_query = WindConditionedStationQuery(
            weather_latitude=_coarsen_axis(weather_latitude, regional_pool_size),
            weather_longitude=_coarsen_axis(weather_longitude, regional_pool_size),
            neighborhood_cells=regional_neighborhood_cells,
            **common,
        )
        self.scale_score = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        final = self.scale_score[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("Unexpected scale-score output layer")
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor((1.0, 0.0)))
        self.output_norm = nn.LayerNorm(d_model)

    @property
    def station_count(self) -> int:
        return self.local_query.station_count

    def station_embedding(self) -> torch.Tensor:
        return self.local_query.station_embedding()

    def _pool(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = values.shape
        pooled = F.avg_pool2d(
            values.reshape(batch * steps, channels, height, width),
            kernel_size=self.regional_pool_size,
            stride=self.regional_pool_size,
            ceil_mode=True,
            count_include_pad=False,
        )
        return pooled.reshape(batch, steps, channels, *pooled.shape[-2:])

    def forward(
        self,
        encoded_map: torch.Tensor,
        standardized_weather: torch.Tensor,
        region_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local, local_attention = self.local_query(
            encoded_map, standardized_weather, region_context
        )
        regional, regional_attention = self.regional_query(
            self._pool(encoded_map),
            self._pool(standardized_weather),
            region_context,
        )
        station = self.station_embedding().reshape(
            1, 1, self.station_count, self.d_model
        ).expand_as(local)
        scale_attention = torch.softmax(
            self.scale_score(torch.cat((local, regional, station), dim=-1)), dim=-1
        )
        fused = (
            scale_attention[..., :1] * local
            + scale_attention[..., 1:] * regional
        )
        return (
            self.output_norm(fused),
            local_attention,
            regional_attention,
            scale_attention,
        )


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values + self.network(values)
        return self.norm(residual.transpose(1, 2)).transpose(1, 2)


class ForcingPathEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        mode: str,
        nhead: int,
        dropout: float,
        max_nodes: int = 24,
        difference_input: str = "actual",
    ) -> None:
        super().__init__()
        if mode not in {"independent", "difference", "tcn", "transformer"}:
            raise ValueError("forcing_path_mode must be independent, difference, tcn, or transformer")
        if difference_input not in {"actual", "zero"}:
            raise ValueError("forcing_difference_input must be actual or zero")
        self.mode = mode
        self.difference_input = difference_input
        self.position = nn.Parameter(torch.zeros(1, max_nodes, d_model))
        nn.init.normal_(self.position, std=0.02)
        if mode == "difference":
            self.encoder: nn.Module = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
            )
        elif mode == "tcn":
            self.encoder = nn.Sequential(
                TemporalResidualBlock(d_model, 1, dropout),
                TemporalResidualBlock(d_model, 2, dropout),
            )
        elif mode == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=2, norm=nn.LayerNorm(d_model)
            )
        else:
            self.encoder = nn.LayerNorm(d_model)

    def forward(self, forcing: torch.Tensor) -> torch.Tensor:
        if forcing.ndim != 4:
            raise ValueError("forcing must have shape [B,N,T,D]")
        batch, stations, steps, channels = forcing.shape
        values = forcing.reshape(batch * stations, steps, channels)
        values = values + self.position[:, :steps]
        if self.mode == "difference":
            difference = torch.cat(
                (torch.zeros_like(values[:, :1]), values[:, 1:] - values[:, :-1]),
                dim=1,
            )
            if self.difference_input == "zero":
                difference = torch.zeros_like(difference)
            values = self.encoder(torch.cat((values, difference), dim=-1))
        elif self.mode == "tcn":
            values = self.encoder(values.transpose(1, 2)).transpose(1, 2)
        else:
            values = self.encoder(values)
        return values.reshape(batch, stations, steps, channels)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=self.padding, dilation=dilation
        )
        self.projection = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.conv(values)
        if self.padding:
            update = update[..., :-self.padding]
        update = self.projection(self.dropout(F.gelu(update)))
        return self.norm((values + update).transpose(1, 2)).transpose(1, 2)


class SharedHistoryEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        layers: int,
        dropout: float,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        self.input_channels = int(input_channels)
        self.input = nn.Conv1d(self.input_channels, d_model, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                CausalResidualBlock(d_model, 2**layer, dropout)
                for layer in range(layers)
            ]
        )

    def forward(
        self, history: torch.Tensor, station_embedding: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim == 2:
            if self.input_channels != 1:
                raise ValueError(
                    "two-dimensional history requires input_channels=1"
                )
            values = history.unsqueeze(1)
            batch, steps = history.shape
            stations = None
        elif history.ndim == 3:
            if self.input_channels != 1:
                raise ValueError(
                    "three-dimensional history requires input_channels=1"
                )
            batch, stations, steps = history.shape
            values = history.reshape(batch * stations, 1, steps)
        elif history.ndim == 4:
            batch, stations, channels, steps = history.shape
            if channels != self.input_channels:
                raise ValueError("history channel count is inconsistent")
            values = history.reshape(batch * stations, channels, steps)
        else:
            raise ValueError(
                "history must have shape [B,T], [B,N,T], or [B,N,C,T]"
            )
        tokens = self.input(values)
        if station_embedding is not None:
            if stations is None or station_embedding.shape != (stations, tokens.shape[1]):
                raise ValueError("station_embedding shape is inconsistent with history")
            tokens = tokens + station_embedding.repeat(batch, 1).unsqueeze(-1)
        tokens = self.blocks(tokens).transpose(1, 2)
        if stations is None:
            return tokens, tokens[:, -1]
        tokens = tokens.reshape(batch, stations, steps, -1)
        return tokens, tokens[:, :, -1]


class TargetTimeEmbedding(nn.Module):
    def __init__(self, d_model: int, forecast_hours: float = 6.0) -> None:
        super().__init__()
        self.forecast_hours = float(forecast_hours)
        self.network = nn.Sequential(
            nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.phase_embedding = nn.Embedding(4, d_model)

    def forward(self, lead_hours: torch.Tensor, origin_phase: torch.Tensor) -> torch.Tensor:
        normalized = lead_hours / self.forecast_hours
        features = torch.stack(
            (
                normalized,
                torch.sin(math.pi * normalized),
                torch.cos(math.pi * normalized),
                torch.sin(2.0 * math.pi * normalized),
                torch.cos(2.0 * math.pi * normalized),
            ),
            dim=-1,
        )
        return self.network(features) + self.phase_embedding(origin_phase).unsqueeze(1)


class RelativeTimeCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, time_scale_hours: float) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by attention_heads")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.time_scale_hours = float(time_scale_hours)
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.time_bias = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, nhead)
        )
        self.output = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_time_hours: torch.Tensor,
        context_time_hours: torch.Tensor,
        *,
        attention_query: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or context.ndim != 4:
            raise ValueError("query and context must have shape [B,N,T,D]")
        batch, stations, query_steps, channels = query.shape
        if context.shape[:2] != (batch, stations) or context.shape[-1] != channels:
            raise ValueError("query and context batch/station dimensions differ")
        context_steps = context.shape[2]
        if query_time_hours.shape != (batch, query_steps):
            raise ValueError("query_time_hours has an invalid shape")
        if context_time_hours.shape != (batch, context_steps):
            raise ValueError("context_time_hours has an invalid shape")
        flat = batch * stations
        score_query = query if attention_query is None else attention_query
        if score_query.shape != query.shape:
            raise ValueError("attention_query must have the same shape as query")
        q = self.q_projection(score_query).reshape(
            flat, query_steps, self.nhead, self.head_dim
        ).transpose(1, 2)
        k = self.k_projection(context).reshape(
            flat, context_steps, self.nhead, self.head_dim
        ).transpose(1, 2)
        v = self.v_projection(context).reshape(
            flat, context_steps, self.nhead, self.head_dim
        ).transpose(1, 2)
        logits = torch.einsum("bhqd,bhsd->bhqs", q, k) / math.sqrt(self.head_dim)
        delta = (
            query_time_hours[:, :, None] - context_time_hours[:, None, :]
        ) / self.time_scale_hours
        time_features = torch.stack((delta, delta.abs()), dim=-1)
        bias = self.time_bias(time_features).permute(0, 3, 1, 2)
        bias = bias.repeat_interleave(stations, dim=0)
        logits = logits + bias
        if attention_bias is not None:
            if attention_bias.shape == (batch, stations, query_steps, context_steps):
                extra_bias = attention_bias.reshape(
                    flat, 1, query_steps, context_steps
                )
            elif attention_bias.shape == (
                batch,
                stations,
                self.nhead,
                query_steps,
                context_steps,
            ):
                extra_bias = attention_bias.reshape(
                    flat, self.nhead, query_steps, context_steps
                )
            else:
                raise ValueError("attention_bias has an invalid shape")
            logits = logits + extra_bias.to(logits)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhqs,bhsd->bhqd", weights, v)
        attended = attended.transpose(1, 2).reshape(flat, query_steps, channels)
        attended = self.output(attended).reshape(batch, stations, query_steps, channels)
        output = self.norm(query + attended)
        if return_attention:
            return output, weights.reshape(
                batch, stations, self.nhead, query_steps, context_steps
            )
        return output


class WeatherTargetQuery(nn.Module):
    """Map native hourly forcing tokens to quarter-hour target timestamps."""

    MODES = {
        "nearest",
        "raw_linear",
        "latent_linear",
        "bracketing_mlp",
        "bracketing_attention",
        "full_attention",
    }

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        mode: str,
        time_scale_hours: float = 2.0,
        bracket_time_feature_mode: str = "relative_distances",
        bracket_mapper_activation: str = "gelu",
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"time_query_mode must be one of {sorted(self.MODES)}")
        if bracket_time_feature_mode not in {"relative_distances", "zero"}:
            raise ValueError(
                "bracket_time_feature_mode must be relative_distances or zero"
            )
        if bracket_mapper_activation not in {"gelu", "identity"}:
            raise ValueError("bracket_mapper_activation must be gelu or identity")
        self.mode = mode
        self.d_model = int(d_model)
        self.time_scale_hours = float(time_scale_hours)
        self.bracket_time_feature_mode = bracket_time_feature_mode
        self.bracket_mapper_activation = bracket_mapper_activation
        self.full_attention = RelativeTimeCrossAttention(
            d_model, nhead, dropout, time_scale_hours
        )
        self.nearest_projection = nn.Linear(d_model, d_model)
        local_activation: nn.Module = (
            nn.GELU() if bracket_mapper_activation == "gelu" else nn.Identity()
        )
        self.local_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 2, 2 * d_model),
            local_activation,
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.local_q = nn.Linear(d_model, d_model)
        self.local_k = nn.Linear(d_model, d_model)
        self.local_v = nn.Linear(d_model, d_model)
        self.local_time_bias = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1)
        )
        self.local_output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _gather_nodes(context: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, stations, _, channels = context.shape
        query_steps = indices.shape[1]
        gather_index = indices[:, None, :, None].expand(
            batch, stations, query_steps, channels
        )
        return torch.gather(context, dim=2, index=gather_index)

    @staticmethod
    def _bracketing_indices(
        query_times: torch.Tensor, context_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        comparisons = context_times[:, None, :] <= query_times[:, :, None]
        lower = comparisons.long().sum(dim=-1).sub(1).clamp_min(0)
        upper = lower + 1
        upper = upper.clamp_max(context_times.shape[1] - 1)
        before_first = query_times < context_times[:, :1]
        after_last = query_times >= context_times[:, -1:]
        lower = torch.where(before_first, torch.zeros_like(lower), lower)
        upper = torch.where(before_first, torch.zeros_like(upper), upper)
        last = torch.full_like(lower, context_times.shape[1] - 1)
        lower = torch.where(after_last, last, lower)
        upper = torch.where(after_last, last, upper)
        return lower, upper

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_time_hours: torch.Tensor,
        context_time_hours: torch.Tensor,
        *,
        mode_override: str | None = None,
        attention_query: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mode = self.mode if mode_override is None else str(mode_override)
        if mode not in self.MODES:
            raise ValueError(f"mode_override must be one of {sorted(self.MODES)}")
        if mode == "full_attention":
            return self.full_attention(
                query,
                context,
                query_time_hours,
                context_time_hours,
                attention_query=attention_query,
                attention_bias=attention_bias,
                return_attention=return_attention,
            )
        if attention_query is not None or attention_bias is not None or return_attention:
            raise ValueError(
                "Conditioned weather attention requires time_query_mode=full_attention"
            )
        if mode == "raw_linear":
            if context.shape[2] != query.shape[2]:
                raise ValueError(
                    "raw_linear expects raw-interpolated context at every target time"
                )
            return self.norm(query + self.nearest_projection(context))
        if mode == "nearest":
            indices = (
                query_time_hours[:, :, None] - context_time_hours[:, None, :]
            ).abs().argmin(dim=-1)
            selected = self._gather_nodes(context, indices)
            return self.norm(query + self.nearest_projection(selected))

        lower_index, upper_index = self._bracketing_indices(
            query_time_hours, context_time_hours
        )
        lower = self._gather_nodes(context, lower_index)
        upper = self._gather_nodes(context, upper_index)
        lower_time = torch.gather(context_time_hours, 1, lower_index)
        upper_time = torch.gather(context_time_hours, 1, upper_index)
        lower_delta = (query_time_hours - lower_time) / self.time_scale_hours
        upper_delta = (upper_time - query_time_hours) / self.time_scale_hours

        if mode == "latent_linear":
            duration = upper_time - lower_time
            phase = torch.where(
                duration.abs() > 1e-8,
                (query_time_hours - lower_time) / duration.clamp_min(1e-8),
                torch.zeros_like(duration),
            ).clamp(0.0, 1.0)
            interpolated = (
                (1.0 - phase[:, None, :, None]) * lower
                + phase[:, None, :, None] * upper
            )
            return self.norm(query + self.nearest_projection(interpolated))

        if mode == "bracketing_mlp":
            time_features = torch.stack((lower_delta, upper_delta), dim=-1)
            if self.bracket_time_feature_mode == "zero":
                time_features = torch.zeros_like(time_features)
            time_features = time_features[:, None].expand(
                -1, context.shape[1], -1, -1
            )
            update = self.local_mlp(
                torch.cat((lower, upper, time_features.to(lower)), dim=-1)
            )
            return self.norm(query + update)

        nodes = torch.stack((lower, upper), dim=-2)
        delta = torch.stack((lower_delta, -upper_delta), dim=-1)
        time_features = torch.stack((delta, delta.abs()), dim=-1)
        time_bias = self.local_time_bias(time_features).squeeze(-1)
        time_bias = time_bias[:, None].expand(-1, context.shape[1], -1, -1)
        scores = (
            self.local_q(query).unsqueeze(-2) * self.local_k(nodes)
        ).sum(dim=-1) / math.sqrt(self.d_model)
        weights = torch.softmax(scores + time_bias.to(scores), dim=-1)
        update = (
            weights.unsqueeze(-1) * self.local_v(nodes)
        ).sum(dim=-2)
        return self.norm(query + self.local_output(update))


def inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(1e-4)
    return values + torch.log(-torch.expm1(-values))
