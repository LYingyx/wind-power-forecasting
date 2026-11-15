from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from wind_forcing.models.components import (
    SharedHistoryEncoder,
    TargetTimeEmbedding,
    WeatherTargetQuery,
    inverse_softplus,
)
from wind_forcing.models.forced_evolution import StationForcingEncoder, _forcing_encoder


TRANSITION_TYPES = {
    "gru",
    "selective_ssm",
    "controlled_koopman",
    "mamba",
    "mamba2",
}


def _transition_features(
    control: torch.Tensor,
    target_token: torch.Tensor,
    dt_hours: float,
) -> torch.Tensor:
    difference = torch.cat(
        (torch.zeros_like(control[:, :, :1]), control[:, :, 1:] - control[:, :, :-1]),
        dim=2,
    )
    dt = control.new_full((*control.shape[:-1], 1), float(dt_hours))
    return torch.cat((control, difference, target_token, dt), dim=-1)


class ControlledGRUTransition(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.cell = nn.GRUCell(3 * d_model + 1, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        initial_state: torch.Tensor,
        control: torch.Tensor,
        target_token: torch.Tensor,
        dt_hours: float,
    ) -> torch.Tensor:
        features = _transition_features(control, target_token, dt_hours)
        batch, stations, steps, _ = features.shape
        state = initial_state.reshape(batch * stations, self.d_model)
        outputs = []
        for step in range(steps):
            inputs = features[:, :, step].reshape(batch * stations, -1)
            state = self.norm(self.cell(self.dropout(inputs), state))
            outputs.append(state.reshape(batch, stations, self.d_model))
        return torch.stack(outputs, dim=2)


class SelectiveStateSpaceTransition(nn.Module):
    """Stable input-selective continuous-time diagonal state-space transition."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        feature_dim = 3 * d_model + 1
        initial_rates = torch.logspace(
            math.log10(1.0 / 6.0), math.log10(4.0), d_model
        )
        self.log_rates = nn.Parameter(torch.log(torch.expm1(initial_rates)))
        self.drive = nn.Linear(feature_dim, d_model)
        self.step_selector = nn.Linear(feature_dim, d_model)
        nn.init.zeros_(self.step_selector.weight)
        nn.init.zeros_(self.step_selector.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        initial_state: torch.Tensor,
        control: torch.Tensor,
        target_token: torch.Tensor,
        dt_hours: float,
    ) -> torch.Tensor:
        features = _transition_features(control, target_token, dt_hours)
        state = initial_state
        rates = F.softplus(self.log_rates).to(state)
        outputs = []
        for step in range(features.shape[2]):
            inputs = self.dropout(features[:, :, step])
            drive = torch.tanh(self.drive(inputs))
            # The selector changes the effective input-hold duration but keeps it positive.
            selected_dt = float(dt_hours) * (
                0.5 + torch.sigmoid(self.step_selector(inputs))
            )
            decay = torch.exp(-selected_dt * rates.reshape(1, 1, -1))
            state = decay * state + (1.0 - decay) * drive
            outputs.append(state)
        return torch.stack(outputs, dim=2)


class ControlledKoopmanTransition(nn.Module):
    """Stable linear latent dynamics with a nonlinear exogenous-control lifting."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        feature_dim = 3 * d_model + 1
        self.skew_generator = nn.Parameter(torch.zeros(d_model, d_model))
        initial_rate = torch.full((d_model,), 0.5)
        self.log_rates = nn.Parameter(torch.log(torch.expm1(initial_rate)))
        self.control_lift = nn.Sequential(
            nn.Linear(feature_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.log_input_gain = nn.Parameter(torch.tensor(-1.5))

    def forward(
        self,
        initial_state: torch.Tensor,
        control: torch.Tensor,
        target_token: torch.Tensor,
        dt_hours: float,
    ) -> torch.Tensor:
        features = _transition_features(control, target_token, dt_hours)
        skew = self.skew_generator - self.skew_generator.transpose(0, 1)
        rates = F.softplus(self.log_rates)
        generator = skew - torch.diag(rates)
        # Matrix exponential is evaluated once per rollout in float32 for AMP stability.
        propagator = torch.matrix_exp(float(dt_hours) * generator.float()).to(
            initial_state
        )
        input_gain = F.softplus(self.log_input_gain).to(initial_state)
        state = initial_state
        outputs = []
        for step in range(features.shape[2]):
            lifted_control = torch.tanh(self.control_lift(features[:, :, step]))
            state = torch.matmul(state, propagator.transpose(0, 1))
            state = state + float(dt_hours) * input_gain * lifted_control
            outputs.append(state)
        return torch.stack(outputs, dim=2)


class ControlledMambaTransition(nn.Module):
    """Official causal Mamba scan over an initial state and future controls."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        *,
        version: str,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
    ) -> None:
        super().__init__()
        if version not in {"mamba", "mamba2"}:
            raise ValueError("version must be mamba or mamba2")
        try:
            from mamba_ssm import Mamba, Mamba2
        except (ImportError, OSError) as exc:
            raise ImportError(
                "transition_type=mamba/mamba2 requires the official mamba-ssm "
                "package; see shell/README.md for the pinned wheel installation"
            ) from exc
        self.d_model = int(d_model)
        self.version = version
        feature_dim = 3 * d_model + 1
        self.control_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.initial_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.input_norm = nn.LayerNorm(d_model)
        if version == "mamba":
            self.mixer: nn.Module = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            if (d_model * expand) % headdim:
                raise ValueError("mamba_headdim must divide d_model * mamba_expand")
            self.mixer = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        initial_state: torch.Tensor,
        control: torch.Tensor,
        target_token: torch.Tensor,
        dt_hours: float,
    ) -> torch.Tensor:
        features = _transition_features(control, target_token, dt_hours)
        batch, stations, steps, _ = features.shape
        control_tokens = self.control_projection(features)
        initial_token = self.initial_projection(initial_state).unsqueeze(2)
        sequence = torch.cat((initial_token, control_tokens), dim=2)
        flat = sequence.reshape(batch * stations, steps + 1, self.d_model)
        scanned = self.mixer(self.input_norm(flat))
        states = self.output_norm(scanned[:, 1:] + flat[:, 1:])
        return states.reshape(batch, stations, steps, self.d_model)


def _build_transition(
    kind: str,
    d_model: int,
    dropout: float,
    *,
    mamba_d_state: int,
    mamba_d_conv: int,
    mamba_expand: int,
    mamba_headdim: int,
) -> nn.Module:
    if kind == "gru":
        return ControlledGRUTransition(d_model, dropout)
    if kind == "selective_ssm":
        return SelectiveStateSpaceTransition(d_model, dropout)
    if kind == "controlled_koopman":
        return ControlledKoopmanTransition(d_model, dropout)
    if kind in {"mamba", "mamba2"}:
        return ControlledMambaTransition(
            d_model,
            dropout,
            version=kind,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
        )
    raise ValueError(f"transition_type must be one of {sorted(TRANSITION_TYPES)}")


class ControlledTransitionModel(nn.Module):
    """Observation-initialized, weather-controlled recurrent forecast model."""

    def __init__(
        self,
        forcing_encoder: StationForcingEncoder,
        station_capacity_mw: Sequence[float],
        *,
        d_model: int,
        history_steps: int,
        forecast_steps: int,
        history_tcn_layers: int,
        attention_heads: int,
        dropout: float,
        time_query_mode: str,
        transition_type: str,
        multirate_enabled: bool,
        output_level: str,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_headdim: int,
    ) -> None:
        super().__init__()
        if transition_type not in TRANSITION_TYPES:
            raise ValueError(
                f"transition_type must be one of {sorted(TRANSITION_TYPES)}"
            )
        if output_level != "hierarchical":
            raise ValueError("controlled_transition currently requires hierarchical output")
        if multirate_enabled and transition_type != "selective_ssm":
            raise ValueError("multirate consistency is currently defined for selective_ssm")
        if multirate_enabled and forecast_steps % 4:
            raise ValueError("multirate consistency requires a whole number of hours")
        self.forcing_encoder = forcing_encoder
        self.d_model = int(d_model)
        self.history_steps = int(history_steps)
        self.forecast_steps = int(forecast_steps)
        self.transition_type = transition_type
        self.multirate_enabled = bool(multirate_enabled)
        capacity = torch.as_tensor(station_capacity_mw, dtype=torch.float32)
        self.register_buffer("capacity_mw", capacity)
        self.register_buffer("total_capacity_mw", capacity.sum())
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
        self.initial_state = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.target_embedding = TargetTimeEmbedding(
            d_model, forecast_hours=forecast_steps / 4.0
        )
        self.weather_query = WeatherTargetQuery(
            d_model,
            attention_heads,
            dropout,
            mode=time_query_mode,
            time_scale_hours=2.0,
        )
        self.transition = _build_transition(
            transition_type,
            d_model,
            dropout,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_headdim=mamba_headdim,
        )
        self.power_delta = nn.Linear(d_model, 1)
        nn.init.normal_(self.power_delta.weight, std=1e-3)
        nn.init.zeros_(self.power_delta.bias)

    def _readout(
        self, states: torch.Tensor, station_history: torch.Tensor
    ) -> torch.Tensor:
        delta = self.power_delta(states).squeeze(-1)
        base = inverse_softplus(station_history[:, :, -1]).unsqueeze(-1)
        return F.softplus(base + delta)

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
        if station_history.ndim != 3 or station_history.shape[2] != self.history_steps:
            raise ValueError("station_history has an invalid shape")
        batch, stations, _ = station_history.shape
        if stations != len(self.capacity_mw):
            raise ValueError("station_history station count is inconsistent")
        if cluster_history.shape != (batch, self.history_steps):
            raise ValueError("cluster_history has an invalid shape")
        if weather_batch_index.shape != (batch,):
            raise ValueError("weather_batch_index has an invalid shape")

        station_embedding = self.forcing_encoder.station_embedding()
        _, local_state = self.station_history_encoder(
            station_history, station_embedding
        )
        _, global_state = self.cluster_history_encoder(cluster_history)
        global_station = global_state[:, None].expand(-1, stations, -1)
        embedded_station = station_embedding.unsqueeze(0).expand(batch, -1, -1)
        state = self.initial_state(
            torch.cat((local_state, global_station, embedded_station), dim=-1)
        )

        unique_weather = self.forcing_encoder(weather)
        forcing = unique_weather["forcing"].index_select(0, weather_batch_index)
        target_times = self.target_lead_hours.to(station_history).reshape(1, -1)
        target_times = target_times.expand(batch, -1)
        target_token = self.target_embedding(target_times, origin_phase)
        target_token = target_token[:, None] + embedded_station[:, :, None]
        target_token = target_token.expand(-1, -1, self.forecast_steps, -1)
        control = self.weather_query(
            target_token,
            forcing,
            target_times,
            weather_time_offset_hours,
        )
        states = self.transition(state, control, target_token, 0.25)
        station_factor = self._readout(states, station_history)
        cluster_factor = (
            station_factor * self.capacity_mw.reshape(1, -1, 1)
        ).sum(dim=1) / self.total_capacity_mw

        multirate_station: torch.Tensor | None = None
        multirate_cluster: torch.Tensor | None = None
        if self.multirate_enabled:
            hours = self.forecast_steps // 4
            hourly_control = control.reshape(
                batch, stations, hours, 4, self.d_model
            ).mean(dim=3)
            hourly_target = target_token.reshape(
                batch, stations, hours, 4, self.d_model
            ).mean(dim=3)
            hourly_states = self.transition(
                state, hourly_control, hourly_target, 1.0
            )
            multirate_station = self._readout(hourly_states, station_history)
            multirate_cluster = (
                multirate_station * self.capacity_mw.reshape(1, -1, 1)
            ).sum(dim=1) / self.total_capacity_mw

        return {
            "station_factor": station_factor,
            "cluster_factor": cluster_factor,
            "hourly_station_factor_unique": unique_weather["hourly_station_factor"],
            "spatial_attention_unique": unique_weather["spatial_attention"],
            "regional_spatial_attention_unique": unique_weather[
                "regional_spatial_attention"
            ],
            "spatial_scale_attention_unique": unique_weather[
                "spatial_scale_attention"
            ],
            "multirate_station_factor": multirate_station,
            "multirate_cluster_factor": multirate_cluster,
        }


def build_controlled_transition_model(
    model_config: Mapping[str, Any],
    static: Mapping[str, Sequence[float]],
    *,
    history_steps: int = 24,
    forecast_steps: int = 24,
) -> ControlledTransitionModel:
    return ControlledTransitionModel(
        _forcing_encoder(model_config, static),
        static["station_capacity_mw"],
        d_model=int(model_config["d_model"]),
        history_steps=history_steps,
        forecast_steps=forecast_steps,
        history_tcn_layers=int(model_config["history_tcn_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        time_query_mode=str(model_config.get("time_query_mode", "bracketing_attention")),
        transition_type=str(model_config.get("transition_type", "selective_ssm")),
        multirate_enabled=bool(model_config.get("multirate_enabled", False)),
        output_level=str(model_config.get("output_level", "hierarchical")),
        mamba_d_state=int(model_config.get("mamba_d_state", 64)),
        mamba_d_conv=int(model_config.get("mamba_d_conv", 4)),
        mamba_expand=int(model_config.get("mamba_expand", 2)),
        mamba_headdim=int(model_config.get("mamba_headdim", 64)),
    )
