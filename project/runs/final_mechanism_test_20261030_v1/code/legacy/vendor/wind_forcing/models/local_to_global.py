from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from wind_forcing.models.components import (
    ForcingPathEncoder,
    RelativeTimeCrossAttention,
    SharedHistoryEncoder,
    TargetTimeEmbedding,
    TemporalResidualBlock,
    WeatherFrameEncoder,
    WeatherTargetQuery,
    build_station_patch_metadata,
)


class LocalToGlobalModel(nn.Module):
    """Shared farm response followed by an exact capacity-weighted sum.

    Weather is bilinearly sampled before the shared point encoder.  Optional
    operating state is applied through an explicitly configured interaction
    operator before the shared weather decoder.
    """

    def __init__(
        self,
        static: Mapping[str, Sequence[float]],
        *,
        nwp_channels: int,
        weather_hidden_channels: int,
        d_model: int,
        input_history_steps: int,
        history_visible_steps: int,
        state_visible_steps: int,
        forecast_steps: int,
        history_tcn_layers: int,
        trajectory_tcn_layers: int,
        attention_heads: int,
        dropout: float,
        state_mode: str,
        state_interaction_operator: str,
        station_specific_mode: str,
        station_embedding_dim: int,
        modulation_amplitude: float,
        backbone_lr_multiplier: float,
    ) -> None:
        super().__init__()
        if state_mode not in {"none", "aggregate_recent", "local_recent"}:
            raise ValueError("state_mode must be none, aggregate_recent, or local_recent")
        if state_interaction_operator not in {
            "none", "residual_multiply", "zero_invariant_mlp", "concat_mlp"
        }:
            raise ValueError(
                "state_interaction_operator must be none, residual_multiply, "
                "zero_invariant_mlp, or concat_mlp"
            )
        if state_mode == "none" and state_interaction_operator != "none":
            raise ValueError("state_mode=none requires state_interaction_operator=none")
        if state_mode != "none" and state_interaction_operator == "none":
            raise ValueError("an active state_mode requires a state interaction operator")
        if station_specific_mode not in {"none", "id_embedding", "affine"}:
            raise ValueError("station_specific_mode must be none, id_embedding, or affine")
        if not 1 <= history_visible_steps <= input_history_steps:
            raise ValueError("history_visible_steps is outside the input history")
        if not 1 <= state_visible_steps <= input_history_steps:
            raise ValueError("state_visible_steps is outside the input history")
        capacity = torch.as_tensor(static["station_capacity_mw"], dtype=torch.float32)
        self.register_buffer("capacity_mw", capacity)
        self.register_buffer("total_capacity_mw", capacity.sum())
        patch_index, _, fixed_weights = build_station_patch_metadata(
            static["weather_latitude"],
            static["weather_longitude"],
            static["station_latitude"],
            static["station_longitude"],
            "bilinear",
            1,
        )
        self.register_buffer("bilinear_patch_index", torch.from_numpy(patch_index))
        self.register_buffer("bilinear_weights", torch.from_numpy(fixed_weights))
        self.input_history_steps = int(input_history_steps)
        self.history_visible_steps = int(history_visible_steps)
        self.state_visible_steps = int(state_visible_steps)
        self.forecast_steps = int(forecast_steps)
        self.d_model = int(d_model)
        self.state_mode = state_mode
        self.state_interaction_operator = state_interaction_operator
        self.station_specific_mode = station_specific_mode
        self.modulation_amplitude = float(modulation_amplitude)
        self.backbone_lr_multiplier = float(backbone_lr_multiplier)
        self.state_intervention = "none"

        self.weather_encoder = WeatherFrameEncoder(
            nwp_channels,
            weather_hidden_channels,
            d_model,
            static["nwp_mean"],
            static["nwp_std"],
        )
        self.forcing_path = ForcingPathEncoder(
            d_model, "difference", attention_heads, dropout, difference_input="actual"
        )
        self.history_encoder = SharedHistoryEncoder(
            d_model, history_tcn_layers, dropout
        )
        self.state_encoder = SharedHistoryEncoder(
            d_model, history_tcn_layers, dropout
        )
        self.target_embedding = TargetTimeEmbedding(
            d_model, forecast_hours=forecast_steps / 4.0
        )
        self.history_query = RelativeTimeCrossAttention(
            d_model, attention_heads, dropout, time_scale_hours=3.0
        )
        self.weather_query = WeatherTargetQuery(
            d_model, attention_heads, dropout, mode="latent_linear", time_scale_hours=2.0
        )

        # Always instantiate optional modules so all experiment cells have an
        # identical state_dict and can be nested from the same checkpoint.
        self.state_scale = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        nn.init.zeros_(self.state_scale[-1].weight)
        nn.init.zeros_(self.state_scale[-1].bias)
        self.state_interaction = (
            nn.Sequential(
                nn.Linear(2 * d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
            )
            if state_interaction_operator in {"zero_invariant_mlp", "concat_mlp"}
            else None
        )
        if self.state_interaction is not None:
            nn.init.zeros_(self.state_interaction[-1].weight)
            nn.init.zeros_(self.state_interaction[-1].bias)
        self.station_id = nn.Embedding(len(capacity), station_embedding_dim)
        self.station_id_projection = nn.Linear(station_embedding_dim, d_model, bias=False)
        nn.init.zeros_(self.station_id.weight)
        self.station_affine_gamma = nn.Parameter(torch.zeros(len(capacity), d_model))
        self.station_affine_beta = nn.Parameter(torch.zeros(len(capacity), d_model))

        self.baseline_response = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.weather_response = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.baseline_trajectory = nn.Sequential(
            *[
                TemporalResidualBlock(d_model, 2**layer, dropout)
                for layer in range(trajectory_tcn_layers)
            ]
        )
        self.weather_trajectory = nn.Sequential(
            *[
                TemporalResidualBlock(d_model, 2**layer, dropout)
                for layer in range(trajectory_tcn_layers)
            ]
        )
        self.baseline_head = nn.Conv1d(d_model, 1, 1)
        self.correction_head = nn.Conv1d(d_model, 1, 1)
        nn.init.normal_(self.baseline_head.weight, std=1e-3)
        nn.init.zeros_(self.baseline_head.bias)
        nn.init.normal_(self.correction_head.weight, std=1e-3)
        nn.init.zeros_(self.correction_head.bias)

    @property
    def forcing_encoder(self) -> nn.Module:
        # Compatibility with generic checkpoint metadata.
        return self.weather_encoder

    def set_state_intervention(self, intervention: str) -> None:
        if intervention not in {"none", "station_roll", "batch_roll", "reference"}:
            raise ValueError("unsupported local-state intervention")
        self.state_intervention = intervention

    def _local_weather(self, weather: torch.Tensor) -> torch.Tensor:
        if weather.ndim == 4:
            # Precomputed cache: [B,T,N,C].
            if weather.shape[2] != len(self.capacity_mw):
                raise ValueError("cached local weather has the wrong station count")
            return weather
        if weather.ndim != 5:
            raise ValueError("weather must be [B,T,C,H,W] or [B,T,N,C]")
        batch, steps, channels, _, _ = weather.shape
        flat_index = self.bilinear_patch_index.flatten()
        values = weather.flatten(-2)[:, :, :, flat_index]
        values = values.reshape(
            batch, steps, channels, len(self.capacity_mw), 4
        ).permute(0, 1, 3, 4, 2)
        weights = self.bilinear_weights.to(weather).reshape(1, 1, len(self.capacity_mw), 4, 1)
        return (values * weights).sum(dim=3)

    def _state_history(
        self, station_history: torch.Tensor, cluster_history: torch.Tensor
    ) -> torch.Tensor:
        if self.state_mode == "aggregate_recent":
            values = cluster_history[:, -self.state_visible_steps :]
            values = values[:, None].expand(-1, station_history.shape[1], -1)
        else:
            values = station_history[:, :, -self.state_visible_steps :]
        if self.state_intervention == "station_roll":
            values = values.roll(1, dims=1)
        elif self.state_intervention == "batch_roll":
            values = values.roll(1, dims=0)
        elif self.state_intervention == "reference":
            values = values.mean(dim=(0, 1), keepdim=True).expand_as(values)
        return values

    def optimizer_parameter_groups(
        self, learning_rate: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        active_names = {"state_scale"}
        if self.state_interaction is not None:
            active_names.add("state_interaction")
        if self.station_specific_mode == "id_embedding":
            active_names.update({"station_id", "station_id_projection"})
        elif self.station_specific_mode == "affine":
            active_names.update({"station_affine_gamma", "station_affine_beta"})
        head, backbone = [], []
        for name, parameter in self.named_parameters():
            if any(name == prefix or name.startswith(prefix + ".") for prefix in active_names):
                head.append(parameter)
            else:
                backbone.append(parameter)
        multiplier = self.backbone_lr_multiplier
        if multiplier >= 1.0 or self.state_mode == "none" and self.station_specific_mode == "none":
            return [{"params": [*head, *backbone], "lr": learning_rate, "weight_decay": weight_decay}]
        return [
            {"params": head, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": backbone, "lr": learning_rate * multiplier, "weight_decay": weight_decay},
        ]

    def _apply_state_interaction(
        self, weather_response: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply state at the local weather latent and return its update.

        ``zero_invariant_mlp`` evaluates the same deterministic MLP twice, so
        an exactly zero weather latent always produces an exactly zero update.
        The interaction MLP intentionally contains no dropout.
        """
        expanded_state = state[:, :, None].expand(
            -1, -1, weather_response.shape[2], -1
        )
        if self.state_interaction_operator == "residual_multiply":
            scale = self.modulation_amplitude * torch.tanh(self.state_scale(state))
            update = weather_response * scale[:, :, None]
        elif self.state_interaction_operator == "zero_invariant_mlp":
            if self.state_interaction is None:
                raise RuntimeError("zero-invariant interaction MLP is missing")
            actual = self.state_interaction(
                torch.cat((weather_response, expanded_state), dim=-1)
            )
            zero = self.state_interaction(
                torch.cat((torch.zeros_like(weather_response), expanded_state), dim=-1)
            )
            update = self.modulation_amplitude * (actual - zero)
        elif self.state_interaction_operator == "concat_mlp":
            if self.state_interaction is None:
                raise RuntimeError("concat interaction MLP is missing")
            update = self.modulation_amplitude * self.state_interaction(
                torch.cat((weather_response, expanded_state), dim=-1)
            )
        else:
            raise RuntimeError("state interaction requested with operator=none")
        return weather_response + update, update

    def forward(
        self,
        *,
        weather: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
        station_history: torch.Tensor,
        cluster_history: torch.Tensor,
        origin_phase: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        batch, stations, input_steps = station_history.shape
        if stations != len(self.capacity_mw) or input_steps != self.input_history_steps:
            raise ValueError("station_history has an invalid shape")
        local_weather = self._local_weather(weather)
        weather_nodes = self.weather_encoder.encode_points(local_weather).permute(0, 2, 1, 3)
        weather_nodes = self.forcing_path(weather_nodes)
        weather_nodes = weather_nodes.index_select(0, weather_batch_index)

        history = station_history[:, :, -self.history_visible_steps :]
        history_tokens, _ = self.history_encoder(history)
        target_times = torch.arange(
            1, self.forecast_steps + 1, device=station_history.device, dtype=station_history.dtype
        ).reshape(1, -1).expand(batch, -1) / 4.0
        target = self.target_embedding(target_times, origin_phase)
        target = target[:, None].expand(-1, stations, -1, -1)
        history_times = torch.arange(
            -(self.history_visible_steps - 1), 1,
            device=station_history.device,
            dtype=station_history.dtype,
        ).reshape(1, -1).expand(batch, -1) / 4.0
        history_response = self.history_query(
            target, history_tokens, target_times, history_times
        )
        baseline_latent = self.baseline_response(
            torch.cat((history_response, target), dim=-1)
        )

        weather_response = self.weather_query(
            torch.zeros_like(target),
            weather_nodes,
            target_times,
            weather_time_offset_hours,
        )
        interaction_update = None
        if self.state_mode != "none":
            _, state = self.state_encoder(
                self._state_history(station_history, cluster_history)
            )
            weather_response, interaction_update = self._apply_state_interaction(
                weather_response, state
            )
        weather_latent = self.weather_response(
            torch.cat((weather_response, target), dim=-1)
        )

        if self.station_specific_mode == "id_embedding":
            ids = torch.arange(stations, device=station_history.device)
            station_adjustment = self.station_id_projection(self.station_id(ids))
            weather_latent = weather_latent + station_adjustment[None, :, None]
        elif self.station_specific_mode == "affine":
            weather_latent = weather_latent * (
                1.0 + self.station_affine_gamma[None, :, None]
            ) + self.station_affine_beta[None, :, None]

        baseline_decoded = self.baseline_trajectory(
            baseline_latent.reshape(batch * stations, self.forecast_steps, self.d_model).transpose(1, 2)
        )
        weather_decoded = self.weather_trajectory(
            weather_latent.reshape(batch * stations, self.forecast_steps, self.d_model).transpose(1, 2)
        )
        baseline_delta = self.baseline_head(baseline_decoded).squeeze(1).reshape(
            batch, stations, self.forecast_steps
        )
        correction_logit = self.correction_head(weather_decoded).squeeze(1).reshape(
            batch, stations, self.forecast_steps
        )
        current = station_history[:, :, -1:].clamp(1e-4, 1.0 - 1e-4)
        current_logit = torch.logit(current)
        baseline_logit = current_logit + baseline_delta
        baseline_factor = torch.sigmoid(baseline_logit)
        station_factor = torch.sigmoid(baseline_logit + correction_logit)
        cluster_factor = (
            station_factor * self.capacity_mw.reshape(1, -1, 1)
        ).sum(dim=1) / self.total_capacity_mw
        baseline_cluster = (
            baseline_factor * self.capacity_mw.reshape(1, -1, 1)
        ).sum(dim=1) / self.total_capacity_mw
        return {
            "station_factor": station_factor,
            "cluster_factor": cluster_factor,
            "baseline_station_factor": baseline_factor,
            "baseline_cluster_factor": baseline_cluster,
            "hourly_station_factor_unique": None,
            "local_weather_correction_logit": correction_logit,
            "interaction_update_abs_mean": (
                None
                if interaction_update is None
                else interaction_update.abs().mean(dim=(1, 3))
            ),
        }


def build_local_to_global_model(
    model_config: Mapping[str, Any],
    static: Mapping[str, Sequence[float]],
    *,
    history_steps: int = 24,
    forecast_steps: int = 24,
) -> LocalToGlobalModel:
    state_mode = str(model_config.get("state_mode", "none"))
    default_operator = "none" if state_mode == "none" else "residual_multiply"
    return LocalToGlobalModel(
        static,
        nwp_channels=int(model_config["weather_channels"]),
        weather_hidden_channels=int(model_config["weather_hidden_channels"]),
        d_model=int(model_config["d_model"]),
        input_history_steps=history_steps,
        history_visible_steps=int(model_config.get("history_visible_steps", 12)),
        state_visible_steps=int(model_config.get("state_visible_steps", 4)),
        forecast_steps=forecast_steps,
        history_tcn_layers=int(model_config["history_tcn_layers"]),
        trajectory_tcn_layers=int(model_config["trajectory_tcn_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        state_mode=state_mode,
        state_interaction_operator=str(
            model_config.get("state_interaction_operator", default_operator)
        ),
        station_specific_mode=str(model_config.get("station_specific_mode", "none")),
        station_embedding_dim=int(model_config.get("station_embedding_dim", 16)),
        modulation_amplitude=float(model_config.get("modulation_amplitude", 0.25)),
        backbone_lr_multiplier=float(model_config.get("backbone_lr_multiplier", 1.0)),
    )
