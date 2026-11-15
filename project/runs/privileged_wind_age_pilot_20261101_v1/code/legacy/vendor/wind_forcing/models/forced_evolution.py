from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from wind_forcing.models.components import (
    ForcingPathEncoder,
    MultiScaleWindConditionedStationQuery,
    RegionalContextEncoder,
    RelativeTimeCrossAttention,
    SharedHistoryEncoder,
    TargetTimeEmbedding,
    TemporalResidualBlock,
    WeatherFrameEncoder,
    WeatherTargetQuery,
    WindConditionedStationQuery,
    inverse_softplus,
)


class StationForcingEncoder(nn.Module):
    def __init__(
        self,
        *,
        nwp_channels: int,
        weather_hidden_channels: int,
        d_model: int,
        weather_latitude: Sequence[float],
        weather_longitude: Sequence[float],
        station_latitude: Sequence[float],
        station_longitude: Sequence[float],
        station_capacity_mw: Sequence[float],
        nwp_mean: Sequence[float],
        nwp_std: Sequence[float],
        station_query_mode: str,
        station_neighborhood_cells: int,
        include_wind_geometry: bool,
        include_region_context: bool,
        forcing_path_mode: str,
        attention_heads: int,
        dropout: float,
        forcing_difference_input: str = "actual",
        regional_pool_size: int = 1,
        regional_neighborhood_cells: int = 1,
    ) -> None:
        super().__init__()
        self.frame_encoder = WeatherFrameEncoder(
            nwp_channels, weather_hidden_channels, d_model, nwp_mean, nwp_std
        )
        self.region_encoder = (
            RegionalContextEncoder(d_model) if include_region_context else None
        )
        self.raw_bilinear = station_query_mode == "raw_bilinear"
        if self.raw_bilinear and include_wind_geometry:
            raise ValueError("raw_bilinear does not support wind geometry")
        effective_query_mode = "bilinear" if self.raw_bilinear else station_query_mode
        query_class = (
            MultiScaleWindConditionedStationQuery
            if regional_pool_size > 1
            else WindConditionedStationQuery
        )
        query_settings: dict[str, Any] = {}
        if regional_pool_size > 1:
            query_settings = {
                "regional_pool_size": regional_pool_size,
                "regional_neighborhood_cells": regional_neighborhood_cells,
            }
        self.station_query = query_class(
            d_model=d_model,
            weather_latitude=weather_latitude,
            weather_longitude=weather_longitude,
            station_latitude=station_latitude,
            station_longitude=station_longitude,
            station_capacity_mw=station_capacity_mw,
            nwp_mean=nwp_mean,
            nwp_std=nwp_std,
            query_mode=effective_query_mode,
            neighborhood_cells=station_neighborhood_cells,
            include_wind_geometry=include_wind_geometry,
            include_region_context=include_region_context,
            **query_settings,
        )
        self.path_encoder = ForcingPathEncoder(
            d_model,
            forcing_path_mode,
            attention_heads,
            dropout,
            difference_input=forcing_difference_input,
        )
        self.hourly_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.hourly_head.weight, std=0.01)
        nn.init.constant_(self.hourly_head.bias, -1.5)

    @property
    def station_count(self) -> int:
        return self.station_query.station_count

    def station_embedding(self) -> torch.Tensor:
        return self.station_query.station_embedding()

    def encode_station_frames(
        self, weather: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        """Encode weather frames spatially without imposing a temporal path."""
        feature_map = None
        if not self.raw_bilinear or self.region_encoder is not None:
            feature_map = self.frame_encoder(weather)
        region_context = (
            self.region_encoder(feature_map)
            if self.region_encoder is not None and feature_map is not None
            else None
        )
        if self.raw_bilinear:
            if not isinstance(self.station_query, WindConditionedStationQuery):
                raise RuntimeError("raw_bilinear requires a single-scale station query")
            batch, steps, channels, _, _ = weather.shape
            flat_indices = self.station_query.patch_index.flatten()
            raw_tokens = weather.flatten(-2)[:, :, :, flat_indices]
            raw_tokens = raw_tokens.reshape(
                batch,
                steps,
                channels,
                self.station_query.station_count,
                self.station_query.patch_token_count,
            ).permute(0, 1, 3, 4, 2)
            spatial_attention = self.station_query.fixed_spatial_weights.to(weather)
            spatial_attention = spatial_attention.reshape(
                1,
                1,
                self.station_query.station_count,
                self.station_query.patch_token_count,
            ).expand(batch, steps, -1, -1)
            interpolated = (raw_tokens * spatial_attention.unsqueeze(-1)).sum(dim=-2)
            local = self.frame_encoder.encode_points(interpolated)
            forcing = self.station_query.fuse_station_features(local, region_context)
            regional_attention = None
            scale_attention = None
        else:
            assert feature_map is not None
            query_output = self.station_query(feature_map, weather, region_context)
            forcing, spatial_attention = query_output[:2]
            regional_attention = query_output[2] if len(query_output) == 4 else None
            scale_attention = query_output[3] if len(query_output) == 4 else None
        return {
            "station_frames": forcing.permute(0, 2, 1, 3),
            "spatial_attention": spatial_attention,
            "regional_spatial_attention": regional_attention,
            "spatial_scale_attention": scale_attention,
        }

    def forward(self, weather: torch.Tensor) -> dict[str, torch.Tensor | None]:
        frames = self.encode_station_frames(weather)
        station_frames = frames["station_frames"]
        assert station_frames is not None
        forcing = self.path_encoder(station_frames)
        hourly_factor = F.softplus(self.hourly_head(forcing).squeeze(-1))
        return {
            "forcing": forcing,
            "hourly_station_factor": hourly_factor,
            "spatial_attention": frames["spatial_attention"],
            "regional_spatial_attention": frames["regional_spatial_attention"],
            "spatial_scale_attention": frames["spatial_scale_attention"],
        }


class SpatialPretrainModel(nn.Module):
    def __init__(
        self, forcing_encoder: StationForcingEncoder, station_capacity_mw: Sequence[float]
    ) -> None:
        super().__init__()
        self.forcing_encoder = forcing_encoder
        capacity = torch.as_tensor(station_capacity_mw, dtype=torch.float32)
        self.register_buffer("capacity_mw", capacity)
        self.register_buffer("total_capacity_mw", capacity.sum())

    def forward(self, weather: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.forcing_encoder(weather)
        station_factor = output["hourly_station_factor"]
        cluster_factor = (
            station_factor * self.capacity_mw.reshape(1, -1, 1)
        ).sum(dim=1) / self.total_capacity_mw
        return {**output, "cluster_factor": cluster_factor}


class ForcedEvolutionModel(nn.Module):
    def __init__(
        self,
        forcing_encoder: StationForcingEncoder,
        station_capacity_mw: Sequence[float],
        *,
        d_model: int,
        history_steps: int,
        forecast_steps: int,
        history_tcn_layers: int,
        trajectory_tcn_layers: int,
        attention_heads: int,
        dropout: float,
        time_query_mode: str = "full_attention",
        bracket_time_feature_mode: str = "relative_distances",
        bracket_mapper_activation: str = "gelu",
        fusion_mode: str = "state_conditioned",
        output_level: str = "hierarchical",
    ) -> None:
        super().__init__()
        self.forcing_encoder = forcing_encoder
        self.history_steps = int(history_steps)
        self.forecast_steps = int(forecast_steps)
        self.d_model = int(d_model)
        allowed_fusion = {
            "history_only",
            "weather_only",
            "direct_concat",
            "target_concat",
            "lead_gate",
            "state_conditioned",
            "prediction_weight",
            "cross_attention",
            "bidirectional_cross_attention",
            "tft",
            "astafn",
        }
        if fusion_mode not in allowed_fusion:
            raise ValueError(f"fusion_mode must be one of {sorted(allowed_fusion)}")
        if output_level not in {"hierarchical", "direct_cluster"}:
            raise ValueError("output_level must be hierarchical or direct_cluster")
        self.fusion_mode = fusion_mode
        self.output_level = output_level
        capacity = torch.as_tensor(station_capacity_mw, dtype=torch.float32)
        self.register_buffer("capacity_mw", capacity)
        self.register_buffer("total_capacity_mw", capacity.sum())
        self.register_buffer(
            "history_time_hours",
            torch.arange(-(history_steps - 1), 1, dtype=torch.float32) / 4.0,
        )
        self.register_buffer(
            "target_lead_hours",
            torch.arange(1, forecast_steps + 1, dtype=torch.float32) / 4.0,
        )
        self.station_history_encoder = SharedHistoryEncoder(
            d_model, history_tcn_layers, dropout
        )
        self.cluster_history_encoder = SharedHistoryEncoder(
            d_model, history_tcn_layers, dropout
        )
        self.target_embedding = TargetTimeEmbedding(
            d_model, forecast_hours=forecast_steps / 4.0
        )
        self.history_query = RelativeTimeCrossAttention(
            d_model, attention_heads, dropout, time_scale_hours=6.0
        )
        self.weather_query = WeatherTargetQuery(
            d_model,
            attention_heads,
            dropout,
            mode=time_query_mode,
            time_scale_hours=2.0,
            bracket_time_feature_mode=bracket_time_feature_mode,
            bracket_mapper_activation=bracket_mapper_activation,
        )
        self.state_modulation = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )
        def response_layer(inputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(inputs * d_model, 2 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
            )

        self.response_layers = nn.ModuleDict(
            {
                "history_only": response_layer(3),
                "weather_only": response_layer(2),
                "direct_concat": response_layer(4),
                "target_concat": response_layer(4),
                "lead_gate": response_layer(2),
                "state_conditioned": response_layer(5),
                "cross_attention": response_layer(4),
                "bidirectional_cross_attention": response_layer(5),
                "tft": response_layer(2),
                "astafn": response_layer(4),
                "prediction_history": response_layer(3),
                "prediction_weather": response_layer(2),
            }
        )
        self.lead_history = nn.Linear(2 * d_model, d_model)
        self.lead_weather = nn.Linear(d_model, d_model)
        self.lead_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.prediction_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.history_to_weather = RelativeTimeCrossAttention(
            d_model, attention_heads, dropout, time_scale_hours=2.0
        )
        self.weather_to_history = RelativeTimeCrossAttention(
            d_model, attention_heads, dropout, time_scale_hours=6.0
        )
        tft_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tft_encoder = nn.TransformerEncoder(
            tft_layer, num_layers=2, norm=nn.LayerNorm(d_model)
        )
        self.tft_query = RelativeTimeCrossAttention(
            d_model, attention_heads, dropout, time_scale_hours=6.0
        )
        self.source_embedding = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.source_embedding, std=0.02)
        self.astafn_adaptation = response_layer(3)
        self.astafn_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.trajectory_decoder = nn.Sequential(
            *[
                TemporalResidualBlock(d_model, 2**layer, dropout)
                for layer in range(trajectory_tcn_layers)
            ]
        )
        self.power_delta = nn.Conv1d(d_model, 1, kernel_size=1)
        self.cluster_delta = nn.Conv1d(d_model, 1, kernel_size=1)
        nn.init.normal_(self.power_delta.weight, std=1e-3)
        nn.init.zeros_(self.power_delta.bias)
        nn.init.normal_(self.cluster_delta.weight, std=1e-3)
        nn.init.zeros_(self.cluster_delta.bias)
        self.weather_station_base = nn.Parameter(
            torch.full((len(capacity), 1), -1.5, dtype=torch.float32)
        )
        self.weather_cluster_base = nn.Parameter(torch.tensor(-1.5))

    def forward(
        self,
        *,
        weather: torch.Tensor | None = None,
        weather_batch_index: torch.Tensor | None = None,
        weather_time_offset_hours: torch.Tensor | None = None,
        station_history: torch.Tensor,
        cluster_history: torch.Tensor,
        origin_phase: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if station_history.ndim != 3 or station_history.shape[2] != self.history_steps:
            raise ValueError("station_history has an invalid shape")
        batch, station_count = station_history.shape[:2]
        if station_count != len(self.capacity_mw):
            raise ValueError("station_history station count is inconsistent")
        if cluster_history.shape != (batch, self.history_steps):
            raise ValueError("cluster_history has an invalid shape")
        if (
            self.fusion_mode != "history_only"
            and (weather_batch_index is None or weather_batch_index.shape != (batch,))
        ):
            raise ValueError("weather_batch_index has an invalid shape")

        station_embedding = self.forcing_encoder.station_embedding()
        history_tokens, history_state = self.station_history_encoder(
            station_history, station_embedding
        )
        _, global_state = self.cluster_history_encoder(cluster_history)

        target_times = self.target_lead_hours.to(station_history).reshape(1, -1).expand(
            batch, -1
        )
        target_token = self.target_embedding(target_times, origin_phase)
        target_token = target_token[:, None] + station_embedding.reshape(
            1, station_count, 1, self.d_model
        )
        target_token = target_token.expand(-1, -1, self.forecast_steps, -1)
        history_times = self.history_time_hours.to(station_history).reshape(
            1, -1
        ).expand(batch, -1)
        history_response = self.history_query(
            target_token,
            history_tokens,
            target_times,
            history_times,
        )
        global_response = global_state[:, None, None].expand(
            -1, station_count, self.forecast_steps, -1
        )
        unique_weather: dict[str, torch.Tensor] | None = None
        weather_forcing: torch.Tensor | None = None
        weather_response: torch.Tensor | None = None
        if self.fusion_mode != "history_only":
            if weather is None or weather_batch_index is None or weather_time_offset_hours is None:
                raise ValueError(f"Weather inputs are required for fusion_mode={self.fusion_mode}")
            unique_weather = self.forcing_encoder(weather)
            weather_forcing = unique_weather["forcing"].index_select(
                0, weather_batch_index
            )
            weather_response = self.weather_query(
                target_token,
                weather_forcing,
                target_times,
                weather_time_offset_hours,
            )

        mode = self.fusion_mode
        prediction_branches: tuple[torch.Tensor, torch.Tensor] | None = None
        if mode == "history_only":
            response = self.response_layers[mode](
                torch.cat((history_response, global_response, target_token), dim=-1)
            )
        elif mode == "weather_only":
            assert weather_response is not None
            response = self.response_layers[mode](
                torch.cat((weather_response, target_token), dim=-1)
            )
        elif mode == "direct_concat":
            assert weather_forcing is not None
            history_summary = history_state[:, :, None].expand(
                -1, -1, self.forecast_steps, -1
            )
            weather_summary = weather_forcing.mean(dim=2, keepdim=True).expand(
                -1, -1, self.forecast_steps, -1
            )
            response = self.response_layers[mode](
                torch.cat(
                    (history_summary, global_response, weather_summary, target_token),
                    dim=-1,
                )
            )
        elif mode == "target_concat":
            assert weather_response is not None
            response = self.response_layers[mode](
                torch.cat(
                    (history_response, global_response, weather_response, target_token),
                    dim=-1,
                )
            )
        elif mode == "lead_gate":
            assert weather_response is not None
            history_branch = self.lead_history(
                torch.cat((history_response, global_response), dim=-1)
            )
            weather_branch = self.lead_weather(weather_response)
            gate = torch.sigmoid(
                self.lead_gate(
                    torch.cat((history_branch, weather_branch, target_token), dim=-1)
                )
            )
            fused = gate * history_branch + (1.0 - gate) * weather_branch
            response = self.response_layers[mode](
                torch.cat((fused, target_token), dim=-1)
            )
        elif mode == "state_conditioned":
            assert weather_response is not None
            modulation = self.state_modulation(
                torch.cat((history_response, global_response, target_token), dim=-1)
            )
            gamma, beta = modulation.chunk(2, dim=-1)
            forced = (1.0 + torch.tanh(gamma)) * weather_response + beta
            response = self.response_layers[mode](
                torch.cat(
                    (
                        history_response,
                        global_response,
                        forced,
                        history_response * forced,
                        target_token,
                    ),
                    dim=-1,
                )
            )
        elif mode == "prediction_weight":
            assert weather_response is not None
            history_branch = self.response_layers["prediction_history"](
                torch.cat((history_response, global_response, target_token), dim=-1)
            )
            weather_branch = self.response_layers["prediction_weather"](
                torch.cat((weather_response, target_token), dim=-1)
            )
            prediction_branches = (history_branch, weather_branch)
            response = history_branch
        elif mode in {"cross_attention", "bidirectional_cross_attention"}:
            assert weather_forcing is not None and weather_response is not None
            assert weather_time_offset_hours is not None
            history_reads_weather = self.history_to_weather(
                history_response,
                weather_forcing,
                target_times,
                weather_time_offset_hours,
            )
            if mode == "cross_attention":
                response = self.response_layers[mode](
                    torch.cat(
                        (history_response, history_reads_weather, global_response, target_token),
                        dim=-1,
                    )
                )
            else:
                weather_reads_history = self.weather_to_history(
                    weather_response,
                    history_tokens,
                    target_times,
                    history_times,
                )
                response = self.response_layers[mode](
                    torch.cat(
                        (
                            history_response,
                            history_reads_weather,
                            weather_reads_history,
                            global_response,
                            target_token,
                        ),
                        dim=-1,
                    )
                )
        elif mode == "tft":
            assert weather_forcing is not None and weather_time_offset_hours is not None
            history_source = history_tokens + self.source_embedding[0]
            weather_source = weather_forcing + self.source_embedding[1]
            combined = torch.cat((history_source, weather_source), dim=2)
            combined_steps = combined.shape[2]
            encoded = self.tft_encoder(
                combined.reshape(batch * station_count, combined_steps, self.d_model)
            ).reshape(batch, station_count, combined_steps, self.d_model)
            combined_times = torch.cat((history_times, weather_time_offset_hours), dim=1)
            tft_response = self.tft_query(
                target_token, encoded, target_times, combined_times
            )
            response = self.response_layers[mode](
                torch.cat((tft_response, target_token), dim=-1)
            )
        else:
            assert weather_response is not None
            adapted = self.astafn_adaptation(
                torch.cat((history_response, weather_response, target_token), dim=-1)
            )
            gate = torch.sigmoid(
                self.astafn_gate(
                    torch.cat((history_response, weather_response, target_token), dim=-1)
                )
            )
            proxy_response = gate * adapted + (1.0 - gate) * history_response
            response = self.response_layers[mode](
                torch.cat(
                    (proxy_response, weather_response, global_response, target_token),
                    dim=-1,
                )
            )
        decoded = self.trajectory_decoder(
            response.reshape(batch * station_count, self.forecast_steps, self.d_model).transpose(
                1, 2
            )
        )
        if prediction_branches is not None:
            history_decoded = self.trajectory_decoder(
                prediction_branches[0]
                .reshape(batch * station_count, self.forecast_steps, self.d_model)
                .transpose(1, 2)
            )
            weather_decoded = self.trajectory_decoder(
                prediction_branches[1]
                .reshape(batch * station_count, self.forecast_steps, self.d_model)
                .transpose(1, 2)
            )
            history_delta = self.power_delta(history_decoded).squeeze(1).reshape(
                batch, station_count, self.forecast_steps
            )
            weather_delta = self.power_delta(weather_decoded).squeeze(1).reshape(
                batch, station_count, self.forecast_steps
            )
            history_factor = F.softplus(
                inverse_softplus(station_history[:, :, -1].unsqueeze(-1))
                + history_delta
            )
            weather_factor = F.softplus(
                self.weather_station_base.reshape(1, station_count, 1) + weather_delta
            )
            blend = torch.sigmoid(self.prediction_gate(target_token).squeeze(-1))
            station_factor = blend * history_factor + (1.0 - blend) * weather_factor
        else:
            delta = self.power_delta(decoded).squeeze(1).reshape(
                batch, station_count, self.forecast_steps
            )
            base = (
                self.weather_station_base.reshape(1, station_count, 1)
                if mode == "weather_only"
                else inverse_softplus(station_history[:, :, -1].unsqueeze(-1))
            )
            station_factor = F.softplus(base + delta)

        if self.output_level == "direct_cluster":
            station_weight = self.capacity_mw / self.total_capacity_mw
            cluster_latent = (
                decoded.reshape(batch, station_count, self.d_model, self.forecast_steps)
                * station_weight.reshape(1, station_count, 1, 1)
            ).sum(dim=1)
            cluster_update = self.cluster_delta(cluster_latent).squeeze(1)
            cluster_base = (
                self.weather_cluster_base
                if mode == "weather_only"
                else inverse_softplus(cluster_history[:, -1]).unsqueeze(-1)
            )
            cluster_factor = F.softplus(cluster_base + cluster_update)
            station_output: torch.Tensor | None = None
        else:
            cluster_factor = (
                station_factor * self.capacity_mw.reshape(1, -1, 1)
            ).sum(dim=1) / self.total_capacity_mw
            station_output = station_factor
        return {
            "station_factor": station_output,
            "cluster_factor": cluster_factor,
            "hourly_station_factor_unique": (
                unique_weather["hourly_station_factor"]
                if unique_weather is not None
                else None
            ),
            "spatial_attention_unique": (
                unique_weather["spatial_attention"]
                if unique_weather is not None
                else None
            ),
            "regional_spatial_attention_unique": (
                unique_weather["regional_spatial_attention"]
                if unique_weather is not None
                else None
            ),
            "spatial_scale_attention_unique": (
                unique_weather["spatial_scale_attention"]
                if unique_weather is not None
                else None
            ),
        }


def _forcing_encoder(
    model_config: Mapping[str, Any], static: Mapping[str, Sequence[float]]
) -> StationForcingEncoder:
    return StationForcingEncoder(
        nwp_channels=int(model_config["weather_channels"]),
        weather_hidden_channels=int(model_config["weather_hidden_channels"]),
        d_model=int(model_config["d_model"]),
        weather_latitude=static["weather_latitude"],
        weather_longitude=static["weather_longitude"],
        station_latitude=static["station_latitude"],
        station_longitude=static["station_longitude"],
        station_capacity_mw=static["station_capacity_mw"],
        nwp_mean=static["nwp_mean"],
        nwp_std=static["nwp_std"],
        station_query_mode=str(model_config["station_query_mode"]),
        station_neighborhood_cells=int(model_config["station_neighborhood_cells"]),
        include_wind_geometry=bool(model_config["include_wind_geometry"]),
        include_region_context=bool(model_config["include_region_context"]),
        regional_pool_size=int(model_config.get("regional_pool_size", 1)),
        regional_neighborhood_cells=int(
            model_config.get("regional_neighborhood_cells", 1)
        ),
        forcing_path_mode=str(model_config["forcing_path_mode"]),
        forcing_difference_input=str(
            model_config.get("forcing_difference_input", "actual")
        ),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
    )


def build_spatial_pretrain_model(
    model_config: Mapping[str, Any], static: Mapping[str, Sequence[float]]
) -> SpatialPretrainModel:
    return SpatialPretrainModel(
        _forcing_encoder(model_config, static), static["station_capacity_mw"]
    )


def build_forced_evolution_model(
    model_config: Mapping[str, Any],
    static: Mapping[str, Sequence[float]],
    *,
    history_steps: int = 24,
    forecast_steps: int = 24,
) -> ForcedEvolutionModel:
    return ForcedEvolutionModel(
        _forcing_encoder(model_config, static),
        static["station_capacity_mw"],
        d_model=int(model_config["d_model"]),
        history_steps=history_steps,
        forecast_steps=forecast_steps,
        history_tcn_layers=int(model_config["history_tcn_layers"]),
        trajectory_tcn_layers=int(model_config["trajectory_tcn_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        time_query_mode=str(model_config.get("time_query_mode", "full_attention")),
        bracket_time_feature_mode=str(
            model_config.get("bracket_time_feature_mode", "relative_distances")
        ),
        bracket_mapper_activation=str(
            model_config.get("bracket_mapper_activation", "gelu")
        ),
        fusion_mode=str(model_config.get("fusion_mode", "state_conditioned")),
        output_level=str(model_config.get("output_level", "hierarchical")),
    )
