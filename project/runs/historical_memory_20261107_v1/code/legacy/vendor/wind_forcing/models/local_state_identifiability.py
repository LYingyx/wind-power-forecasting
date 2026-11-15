from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from wind_forcing.models.components import (
    RelativeTimeCrossAttention,
    SharedHistoryEncoder,
    TemporalResidualBlock,
)
from wind_forcing.models.components import WindConditionedStationQuery
from wind_forcing.models.local_to_global import LocalToGlobalModel


STATE_SOURCES = {
    "none",
    "local_power",
    "aggregate_power",
    "physical",
    "hierarchical",
    "io_pair",
    "io_pair_id",
}
CONTEXT_MODES = {
    "none",
    "static",
    "learned_id",
    "static_affine",
    "metadata_residual",
    "metadata_film",
    "id_film",
}
INTERACTION_MODES = {
    "none",
    "strict",
    "direct",
    "faithful_concat",
    "additive_main_effect",
    "output_additive",
    "independent_output_additive",
}
FINE_TUNE_POLICIES = {"all", "freeze_state_response"}
SPATIAL_WEATHER_MODES = {
    "precomputed_bilinear",
    "raw_bilinear",
    "fixed_gaussian_4x4",
    "learned_4x4_wind",
}
SOURCE_ROLE_MODES = {"full", "history_only", "anchor_weather"}
RESPONSE_HISTORY_MODES = {"pooled_state", "lead_summary", "weather_cross_attention"}
FORCING_TRANSITION_MODES = {
    "legacy",
    "origin_delta_linear",
    "origin_delta_tcn",
}


class IndependentAdditiveResponse(nn.Module):
    """Parameter-matched independent weather/history main-effect encoders.

    Each branch has ``d -> 1.5d -> d`` weights.  With first-layer bias and
    bias-free output layers, the two branches together contain exactly as many
    trainable parameters as the faithful ``2d -> 2d -> d`` interaction MLP.
    The two outputs are decoded separately and added at correction-logit level,
    so this module cannot form a weather-by-history interaction.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = 3 * d_model // 2
        if 2 * hidden != 3 * d_model:
            raise ValueError("independent additive response requires an even d_model")
        self.weather = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )
        self.history = nn.Sequential(
            nn.Linear(d_model, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )


class LocalStateIdentifiabilityModel(LocalToGlobalModel):
    """Local-to-global forecast with correction-state isolated from history baseline.

    ``station_history`` always belongs to the baseline/current anchor.  All response
    state is built from the separate ``state_*`` tensors, which permits a matched
    train-time donor control without corrupting the baseline input.
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
        response_state_source: str,
        station_context_mode: str,
        response_interaction_mode: str,
        station_embedding_dim: int,
        modulation_amplitude: float,
        backbone_lr_multiplier: float,
        fine_tune_policy: str,
        spatial_weather_mode: str,
        station_neighborhood_cells: int,
        source_role_mode: str = "full",
        response_history_mode: str = "pooled_state",
        forcing_transition_mode: str = "legacy",
        response_output_mode: str = "legacy_logit",
        response_readout_mode: str = "legacy",
    ) -> None:
        if response_readout_mode not in {"legacy", "capacity_aux", "level_shape"}:
            raise ValueError(f"Unsupported response_readout_mode: {response_readout_mode}")
        if response_output_mode not in {"legacy_logit", "cf_residual", "bounded_cf_mix"}:
            raise ValueError(f"Unsupported response_output_mode: {response_output_mode}")
        if response_output_mode != "legacy_logit" and response_interaction_mode not in {
            "faithful_concat", "independent_output_additive"
        }:
            raise ValueError("CF residual output requires a matched Int/IAdd pair")
        if response_state_source not in STATE_SOURCES:
            raise ValueError(f"Unsupported response_state_source: {response_state_source}")
        if station_context_mode not in CONTEXT_MODES:
            raise ValueError(f"Unsupported station_context_mode: {station_context_mode}")
        if response_interaction_mode not in INTERACTION_MODES:
            raise ValueError(f"Unsupported response_interaction_mode: {response_interaction_mode}")
        if fine_tune_policy not in FINE_TUNE_POLICIES:
            raise ValueError(f"Unsupported fine_tune_policy: {fine_tune_policy}")
        if spatial_weather_mode not in SPATIAL_WEATHER_MODES:
            raise ValueError(f"Unsupported spatial_weather_mode: {spatial_weather_mode}")
        if source_role_mode not in SOURCE_ROLE_MODES:
            raise ValueError(f"Unsupported source_role_mode: {source_role_mode}")
        if response_history_mode not in RESPONSE_HISTORY_MODES:
            raise ValueError(f"Unsupported response_history_mode: {response_history_mode}")
        if forcing_transition_mode not in FORCING_TRANSITION_MODES:
            raise ValueError(
                f"Unsupported forcing_transition_mode: {forcing_transition_mode}"
            )
        if response_history_mode != "pooled_state" and response_state_source != "local_power":
            raise ValueError(
                "lead-specific history response modes require response_state_source=local_power"
            )
        if forcing_transition_mode != "legacy" and spatial_weather_mode not in {
            "precomputed_bilinear",
            "raw_bilinear",
        }:
            raise ValueError(
                "forcing-transition experiments require bilinear station weather"
            )
        if source_role_mode != "full" and (
            response_state_source != "none"
            or station_context_mode != "none"
            or response_interaction_mode != "none"
        ):
            raise ValueError(
                "source-role controls require state/context/interaction to be disabled"
            )
        if spatial_weather_mode in {
            "fixed_gaussian_4x4",
            "learned_4x4_wind",
        } and station_neighborhood_cells != 3:
            raise ValueError(
                "The registered 4x4 support requires station_neighborhood_cells=3"
            )
        has_condition = response_state_source != "none" or station_context_mode != "none"
        if has_condition != (response_interaction_mode != "none"):
            raise ValueError("Interaction must be active exactly when state or station context is active")
        super().__init__(
            static,
            nwp_channels=nwp_channels,
            weather_hidden_channels=weather_hidden_channels,
            d_model=d_model,
            input_history_steps=input_history_steps,
            history_visible_steps=history_visible_steps,
            state_visible_steps=state_visible_steps,
            forecast_steps=forecast_steps,
            history_tcn_layers=history_tcn_layers,
            trajectory_tcn_layers=trajectory_tcn_layers,
            attention_heads=attention_heads,
            dropout=dropout,
            state_mode="none",
            state_interaction_operator="none",
            station_specific_mode="none",
            # Legacy no-state checkpoints always contain this inactive 16-D table.
            station_embedding_dim=16,
            modulation_amplitude=modulation_amplitude,
            backbone_lr_multiplier=backbone_lr_multiplier,
        )
        self.response_state_source = response_state_source
        self.station_context_mode = station_context_mode
        self.response_interaction_mode = response_interaction_mode
        self.fine_tune_policy = fine_tune_policy
        self.spatial_weather_mode = spatial_weather_mode
        self.source_role_mode = source_role_mode
        self.response_history_mode = response_history_mode
        self.forcing_transition_mode = forcing_transition_mode
        self.response_output_mode = response_output_mode
        self.response_readout_mode = response_readout_mode
        self.readout_aux = (
            nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1, bias=False))
            if response_readout_mode != "legacy" else None
        )
        if self.readout_aux is not None:
            nn.init.zeros_(self.readout_aux[-1].weight)
        if response_output_mode == "bounded_cf_mix":
            initial_mix = torch.exp(-torch.arange(1, forecast_steps + 1).float() / 12.)
            self.output_mix_logits = nn.Parameter(torch.logit(initial_mix))
        else:
            self.register_parameter("output_mix_logits", None)
        self.response_state_intervention = "none"

        # The old local-state runs consumed a precomputed bilinear point cache,
        # so they could not test whether surrounding weather support changes the
        # history-weather interaction.  Raw-grid experiments use the same
        # station-query family for all three spatial cells.  This holds the
        # station static projection and downstream backbone fixed while changing
        # only raw bilinear versus fixed/learned 4x4 weather support.
        query_mode = {
            "raw_bilinear": "bilinear",
            "fixed_gaussian_4x4": "fixed_gaussian",
            "learned_4x4_wind": "learned_cells",
        }.get(spatial_weather_mode)
        self.spatial_station_query = (
            None
            if query_mode is None
            else WindConditionedStationQuery(
                d_model=d_model,
                weather_latitude=static["weather_latitude"],
                weather_longitude=static["weather_longitude"],
                station_latitude=static["station_latitude"],
                station_longitude=static["station_longitude"],
                station_capacity_mw=static["station_capacity_mw"],
                nwp_mean=static["nwp_mean"],
                nwp_std=static["nwp_std"],
                query_mode=query_mode,
                neighborhood_cells=(
                    1 if spatial_weather_mode == "raw_bilinear" else station_neighborhood_cells
                ),
                include_wind_geometry=spatial_weather_mode == "learned_4x4_wind",
                include_region_context=False,
            )
        )

        self.global_state_encoder = SharedHistoryEncoder(
            d_model, history_tcn_layers, dropout
        )
        self.hierarchical_state_projection = nn.Linear(2 * d_model, d_model)
        self.physical_state_encoder = nn.Sequential(
            nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.io_token_projection = nn.Sequential(
            nn.Linear(2 * d_model + 2, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.io_state_trajectory = nn.Sequential(
            *[
                TemporalResidualBlock(d_model, 2**layer, dropout)
                for layer in range(max(1, history_tcn_layers - 1))
            ]
        )
        self.static_context_encoder = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.response_station_id = nn.Embedding(len(self.capacity_mw), station_embedding_dim)
        self.response_station_id_projection = nn.Linear(
            station_embedding_dim, d_model, bias=False
        )
        nn.init.zeros_(self.response_station_id.weight)
        self.history_summary_encoder = (
            nn.Sequential(
                nn.Linear(9, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            if response_history_mode == "lead_summary"
            else None
        )
        self.history_lead_projection = (
            nn.Sequential(
                nn.Linear(2 * d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
            )
            if response_history_mode == "lead_summary"
            else None
        )
        self.response_history_attention = (
            RelativeTimeCrossAttention(
                d_model, attention_heads, dropout, time_scale_hours=3.0
            )
            if response_history_mode == "weather_cross_attention"
            else None
        )
        self.forcing_delta_encoder = (
            nn.Linear(nwp_channels, d_model, bias=False)
            if forcing_transition_mode == "origin_delta_linear"
            else
            nn.Sequential(
                nn.Linear(nwp_channels, d_model, bias=False),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            if forcing_transition_mode != "legacy"
            else None
        )
        self.forcing_delta_norm = (
            nn.LayerNorm(d_model)
            if forcing_transition_mode == "origin_delta_linear"
            else None
        )
        self.forcing_transition_fusion = (
            nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            if forcing_transition_mode == "origin_delta_tcn"
            else None
        )
        self.forcing_transition_trajectory = (
            nn.Sequential(
                TemporalResidualBlock(d_model, 1, dropout),
                TemporalResidualBlock(d_model, 2, dropout),
            )
            if forcing_transition_mode == "origin_delta_tcn"
            else None
        )
        response_specific_context = station_context_mode in {
            "metadata_residual", "metadata_film", "id_film"
        }
        self.response_context_weather = (
            nn.Linear(d_model, d_model, bias=False)
            if station_context_mode == "metadata_residual"
            else None
        )
        self.response_context_history = (
            nn.Linear(d_model, d_model, bias=False)
            if station_context_mode == "metadata_residual"
            else None
        )
        self.response_context_weather_norm = (
            nn.LayerNorm(d_model) if response_specific_context else None
        )
        self.response_context_history_norm = (
            nn.LayerNorm(d_model) if response_specific_context else None
        )
        self.response_context_film = (
            nn.Linear(d_model, 4 * d_model)
            if station_context_mode in {"metadata_film", "id_film"}
            else None
        )
        if response_interaction_mode == "independent_output_additive":
            self.response_interaction: nn.Module = IndependentAdditiveResponse(d_model)
        else:
            self.response_interaction = nn.Sequential(
                nn.Linear(2 * d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
            )
        self.direct_state_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.identification_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        # The older strict interaction is a residual update and therefore has
        # an identity initialization.  The transfer experiment deliberately
        # reproduces the original farm-level concat MLP as a *replacement* of
        # the weather response; zero initialization would suppress that path.
        if response_interaction_mode not in {
            "faithful_concat",
            "additive_main_effect",
            "output_additive",
            "independent_output_additive",
        }:
            assert isinstance(self.response_interaction, nn.Sequential)
            nn.init.zeros_(self.response_interaction[-1].weight)
            nn.init.zeros_(self.response_interaction[-1].bias)
        nn.init.zeros_(self.direct_state_head[-1].weight)
        nn.init.zeros_(self.direct_state_head[-1].bias)
        nn.init.normal_(self.identification_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.identification_head[-1].bias)

        latitude = np.asarray(static["station_latitude"], dtype=np.float32)
        longitude = np.asarray(static["station_longitude"], dtype=np.float32)
        capacity = np.asarray(static["station_capacity_mw"], dtype=np.float32)
        features = np.stack((latitude, longitude, np.log(capacity)), axis=1)
        features = (features - features.mean(axis=0)) / np.maximum(
            features.std(axis=0), 1e-6
        )
        self.register_buffer("station_static_features", torch.from_numpy(features))
        coordinates = np.stack((latitude, longitude), axis=1)
        coordinates = (coordinates - coordinates.mean(0)) / np.maximum(
            coordinates.std(0), 1e-6
        )
        distance = ((coordinates[:, None] - coordinates[None]) ** 2).sum(-1)
        self.register_buffer(
            "far_station_index", torch.from_numpy(distance.argmax(axis=1).astype(np.int64))
        )
        if self.fine_tune_policy == "freeze_state_response":
            if self.response_state_source != "local_power" or self.response_interaction_mode != "faithful_concat":
                raise ValueError(
                    "freeze_state_response is defined only for local-power faithful-concat models"
                )
            self.state_encoder.requires_grad_(False)
            self.response_interaction.requires_grad_(False)

    def train(self, mode: bool = True) -> LocalStateIdentifiabilityModel:
        super().train(mode)
        if self.fine_tune_policy == "freeze_state_response":
            # Freezing weights alone would leave TCN dropout active and would
            # not preserve the learned state representation during Stage 2.
            self.state_encoder.eval()
            self.response_interaction.eval()
        return self

    def set_response_state_intervention(self, intervention: str) -> None:
        if intervention not in {"none", "station_permutation", "reference", "zero"}:
            raise ValueError("Unsupported response-state intervention")
        self.response_state_intervention = intervention

    def optimizer_parameter_groups(
        self, learning_rate: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        # State identification modules must never inherit the reduced backbone LR.
        active = (
            "state_encoder.",
            "global_state_encoder.",
            "hierarchical_state_projection.",
            "physical_state_encoder.",
            "io_token_projection.",
            "io_state_trajectory.",
            "static_context_encoder.",
            "response_interaction.",
            "direct_state_head.",
            "identification_head.",
            "station_id.",
            "station_id_projection.",
            "response_station_id.",
            "response_station_id_projection.",
            "station_affine_",
            "history_summary_encoder.",
            "history_lead_projection.",
            "response_history_attention.",
            "forcing_delta_",
            "forcing_transition_",
            "response_context_",
            "readout_aux.",
            "output_mix_logits",
        )
        head, backbone = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (head if name.startswith(active) else backbone).append(parameter)
        groups = []
        if head:
            groups.append({"params": head, "lr": learning_rate, "weight_decay": weight_decay})
        if backbone:
            groups.append(
                {
                    "params": backbone,
                    "lr": learning_rate * self.backbone_lr_multiplier,
                    "weight_decay": weight_decay,
                }
            )
        return groups

    def _power_state(
        self, station_history: torch.Tensor, cluster_history: torch.Tensor
    ) -> torch.Tensor:
        steps = self.state_visible_steps
        local = station_history[:, :, -steps:]
        global_values = cluster_history[:, -steps:]
        if self.response_state_source == "local_power":
            _, state = self.state_encoder(local)
        elif self.response_state_source == "aggregate_power":
            _, pooled = self.global_state_encoder(global_values[:, None])
            state = pooled.expand(-1, local.shape[1], -1)
        elif self.response_state_source == "physical":
            current = local[:, :, -1]
            ramp15 = current - local[:, :, -2]
            ramp60 = current - local[:, :, -min(5, steps)]
            recent = local[:, :, -min(4, steps):]
            global_current = global_values[:, -1, None]
            global_ramp = (global_values[:, -1] - global_values[:, -2])[:, None]
            features = torch.stack(
                (
                    current,
                    1.0 - current,
                    ramp15,
                    ramp60,
                    recent.mean(-1),
                    recent.std(-1, unbiased=False),
                    current - global_current,
                    ramp15 - global_ramp,
                ),
                dim=-1,
            )
            state = self.physical_state_encoder(features)
        elif self.response_state_source == "hierarchical":
            _, global_state = self.global_state_encoder(global_values[:, None])
            _, local_deviation = self.state_encoder(
                local - global_values[:, None, :]
            )
            if self.response_state_intervention == "station_permutation":
                local_deviation = local_deviation.index_select(1, self.far_station_index)
            state = self.hierarchical_state_projection(
                torch.cat((global_state.expand(-1, local.shape[1], -1), local_deviation), -1)
            )
        else:
            raise RuntimeError("Power state requested for a non-power source")
        return state

    def _io_state(
        self, weather_nodes: torch.Tensor, power: torch.Tensor
    ) -> torch.Tensor:
        delta_weather = torch.diff(weather_nodes, dim=2, prepend=weather_nodes[:, :, :1])
        delta_power = torch.diff(power, dim=2, prepend=power[:, :, :1])
        token = self.io_token_projection(
            torch.cat((weather_nodes, delta_weather, power[..., None], delta_power[..., None]), -1)
        )
        batch, stations, steps, width = token.shape
        encoded = self.io_state_trajectory(
            token.reshape(batch * stations, steps, width).transpose(1, 2)
        ).transpose(1, 2)
        return encoded[:, -1].reshape(batch, stations, width)

    def _context(self, batch: int, stations: int, device: torch.device) -> torch.Tensor | None:
        if self.station_context_mode == "none":
            return None
        if self.station_context_mode in {
            "static",
            "static_affine",
            "metadata_residual",
            "metadata_film",
        }:
            context = self.static_context_encoder(self.station_static_features.to(device))
        else:
            ids = torch.arange(stations, device=device)
            context = self.response_station_id_projection(self.response_station_id(ids))
        return context[None].expand(batch, -1, -1)

    @staticmethod
    def _history_summary(history: torch.Tensor) -> torch.Tensor:
        """Nine transparent level/ramp/statistical summaries of recent CF."""
        if history.ndim != 3 or history.shape[-1] < 5:
            raise ValueError("lead-summary history requires at least five steps")
        current = history[..., -1]
        recent_1h = history[..., -4:]
        recent_3h = history[..., -min(12, history.shape[-1]):]

        def slope(values: torch.Tensor) -> torch.Tensor:
            steps = values.shape[-1]
            time = torch.arange(steps, device=values.device, dtype=values.dtype)
            time = time - time.mean()
            denominator = time.square().sum().clamp_min(1e-6)
            return (values * time).sum(-1) / denominator * 4.0

        return torch.stack(
            (
                current,
                current - history[..., -2],
                current - history[..., -5],
                recent_1h.mean(-1),
                recent_1h.std(-1, unbiased=False),
                slope(recent_1h),
                recent_3h.mean(-1),
                recent_3h.std(-1, unbiased=False),
                slope(recent_3h),
            ),
            dim=-1,
        )

    def _lead_history_condition(
        self,
        history: torch.Tensor,
        weather_response: torch.Tensor,
        target: torch.Tensor,
        target_times: torch.Tensor,
        history_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.response_history_mode == "pooled_state":
            _, pooled = self.state_encoder(history)
            return pooled, None
        if self.response_history_mode == "lead_summary":
            if self.history_summary_encoder is None or self.history_lead_projection is None:
                raise RuntimeError("Lead-summary modules are missing")
            summary = self.history_summary_encoder(self._history_summary(history))
            expanded = summary[:, :, None].expand(-1, -1, self.forecast_steps, -1)
            return self.history_lead_projection(torch.cat((expanded, target), -1)), None
        if self.response_history_attention is None:
            raise RuntimeError("History cross-attention module is missing")
        tokens, _ = self.state_encoder(history)
        # The additive control must remain H/W separable.  Only the interaction
        # model is allowed to let future weather choose which history tokens to read.
        query = target
        if self.response_interaction_mode != "independent_output_additive":
            query = target + weather_response
        attended, weights = self.response_history_attention(
            query,
            tokens,
            target_times,
            history_times,
            return_attention=True,
        )
        return attended, weights

    def _response_site_condition(
        self,
        weather: torch.Tensor,
        history: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is None or self.station_context_mode not in {
            "metadata_residual", "metadata_film", "id_film"
        }:
            return weather, history
        if history.ndim == 3:
            history = history[:, :, None].expand(-1, -1, weather.shape[2], -1)
        expanded = context[:, :, None].expand(-1, -1, weather.shape[2], -1)
        if self.response_context_weather_norm is None or self.response_context_history_norm is None:
            raise RuntimeError("Response-specific context normalizers are missing")
        if self.station_context_mode == "metadata_residual":
            if self.response_context_weather is None or self.response_context_history is None:
                raise RuntimeError("Metadata residual projections are missing")
            weather = self.response_context_weather_norm(
                weather + 0.1 * torch.tanh(self.response_context_weather(expanded))
            )
            history = self.response_context_history_norm(
                history + 0.1 * torch.tanh(self.response_context_history(expanded))
            )
            return weather, history
        if self.response_context_film is None:
            raise RuntimeError("Response-specific FiLM projection is missing")
        gamma_w, beta_w, gamma_h, beta_h = self.response_context_film(expanded).chunk(4, -1)
        weather = self.response_context_weather_norm(
            weather * (1.0 + 0.1 * torch.tanh(gamma_w))
            + 0.1 * torch.tanh(beta_w)
        )
        history = self.response_context_history_norm(
            history * (1.0 + 0.1 * torch.tanh(gamma_h))
            + 0.1 * torch.tanh(beta_h)
        )
        return weather, history

    def _intervene(self, state: torch.Tensor, *, local: bool) -> torch.Tensor:
        if self.response_state_intervention == "station_permutation" and local:
            state = state.index_select(1, self.far_station_index)
        elif self.response_state_intervention == "reference":
            state = state.mean(dim=(0, 1), keepdim=True).expand_as(state)
        elif self.response_state_intervention == "zero":
            state = torch.zeros_like(state)
        return state

    def _intervene_history(self, history: torch.Tensor) -> torch.Tensor:
        if self.response_state_intervention == "station_permutation":
            return history.index_select(1, self.far_station_index)
        if self.response_state_intervention == "reference":
            return history.mean(dim=(0, 1), keepdim=True).expand_as(history)
        if self.response_state_intervention == "zero":
            return torch.zeros_like(history)
        return history

    def _conditioned_weather(
        self, weather: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = (
            condition[:, :, None].expand(-1, -1, weather.shape[2], -1)
            if condition.ndim == 3
            else condition
        )
        if expanded.shape != weather.shape:
            raise ValueError("Response condition shape is inconsistent with weather")
        if self.response_interaction_mode == "faithful_concat":
            changed = self.response_interaction(torch.cat((weather, expanded), -1))
            return changed, changed - weather
        if self.response_interaction_mode == "additive_main_effect":
            # Same encoder and MLP parameterization as faithful_concat, but no
            # joint weather-state evaluation.  This is a matched control for a
            # predictive history main effect rather than an interaction.
            zeros_weather = torch.zeros_like(weather)
            zeros_state = torch.zeros_like(expanded)
            weather_main = self.response_interaction(
                torch.cat((weather, zeros_state), -1)
            )
            state_main = self.response_interaction(
                torch.cat((zeros_weather, expanded), -1)
            )
            origin = self.response_interaction(
                torch.cat((zeros_weather, zeros_state), -1)
            )
            changed = weather_main + state_main - origin
            return changed, changed - weather
        if self.response_interaction_mode in {
            "output_additive",
            "independent_output_additive",
        }:
            raise RuntimeError(
                f"{self.response_interaction_mode} is decomposed after the complete response decoder"
            )
        actual = self.response_interaction(torch.cat((weather, expanded), -1))
        zero = self.response_interaction(
            torch.cat((torch.zeros_like(weather), expanded), -1)
        )
        update = self.modulation_amplitude * (actual - zero)
        changed = weather + update
        if self.station_context_mode == "static_affine":
            gamma = torch.tanh(self.station_affine_gamma)[None, :, None]
            beta = torch.tanh(self.station_affine_beta)[None, :, None]
            changed = changed * (1.0 + 0.1 * gamma) + 0.1 * beta * torch.tanh(changed)
        return changed, update

    def _spatial_weather_nodes(
        self, weather: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return [cycle, station, native_time, latent] on matched supports.

        ``raw_bilinear`` reproduces interpolation of the 14 standardized raw
        variables before point encoding.  The two 4x4 modes encode native grid
        points first and then use either a fixed Gaussian kernel or the legacy
        content/geometry/wind-conditioned learned query.
        """
        if self.spatial_weather_mode == "precomputed_bilinear":
            local_weather = self._local_weather(weather)
            encoded = self.weather_encoder.encode_points(local_weather)
            return encoded.permute(0, 2, 1, 3), None
        if weather.ndim != 5:
            raise ValueError(
                f"{self.spatial_weather_mode} requires raw weather [B,T,C,H,W]"
            )
        query = self.spatial_station_query
        if query is None:
            raise RuntimeError("Raw-grid spatial station query is missing")
        if self.spatial_weather_mode == "raw_bilinear":
            batch, steps, channels, _, _ = weather.shape
            flat_indices = query.patch_index.flatten()
            raw_tokens = weather.flatten(-2)[:, :, :, flat_indices]
            raw_tokens = raw_tokens.reshape(
                batch,
                steps,
                channels,
                query.station_count,
                query.patch_token_count,
            ).permute(0, 1, 3, 4, 2)
            weights = query.fixed_spatial_weights.to(weather).reshape(
                1, 1, query.station_count, query.patch_token_count
            )
            weights = weights.expand(batch, steps, -1, -1)
            interpolated = (raw_tokens * weights.unsqueeze(-1)).sum(dim=-2)
            local = query.fuse_station_features(
                self.weather_encoder.encode_points(interpolated)
            )
        else:
            local, weights = query(self.weather_encoder(weather), weather)
        return local.permute(0, 2, 1, 3), weights

    def _bilinear_raw_station_weather(self, weather: torch.Tensor) -> torch.Tensor:
        """Return standardized bilinear station weather as [cycle,N,T,C]."""
        if self.spatial_weather_mode == "precomputed_bilinear":
            if weather.ndim != 4:
                raise ValueError("Precomputed station weather must have shape [B,T,N,C]")
            return weather.permute(0, 2, 1, 3)
        if self.spatial_weather_mode != "raw_bilinear" or weather.ndim != 5:
            raise ValueError("Raw station weather is available only for bilinear modes")
        query = self.spatial_station_query
        if query is None:
            raise RuntimeError("Raw bilinear station query is missing")
        batch, steps, channels, _, _ = weather.shape
        flat_indices = query.patch_index.flatten()
        tokens = weather.flatten(-2)[:, :, :, flat_indices]
        tokens = tokens.reshape(
            batch, steps, channels, query.station_count, query.patch_token_count
        ).permute(0, 1, 3, 4, 2)
        weights = query.fixed_spatial_weights.to(weather).reshape(
            1, 1, query.station_count, query.patch_token_count, 1
        )
        return (tokens * weights).sum(-2).permute(0, 2, 1, 3)

    def _forcing_transition(
        self,
        weather_nodes: torch.Tensor,
        raw_weather: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
    ) -> torch.Tensor:
        base = self.forcing_path(weather_nodes).index_select(0, weather_batch_index)
        if self.forcing_transition_mode == "legacy":
            return base
        if self.forcing_delta_encoder is None:
            raise RuntimeError("Forcing delta encoder is missing")
        raw = raw_weather.index_select(0, weather_batch_index)
        if raw.shape[2] < 2 or weather_time_offset_hours.shape != raw.shape[:1] + raw.shape[2:3]:
            raise ValueError("Origin-delta forcing requires at least two timed weather nodes")
        first_time = weather_time_offset_hours[:, 0]
        duration = (weather_time_offset_hours[:, 1] - first_time).clamp_min(1e-6)
        fraction = (-first_time / duration)[:, None, None]
        reference = raw[:, :, 0] + fraction * (raw[:, :, 1] - raw[:, :, 0])
        origin_delta = raw - reference[:, :, None]
        encoded_origin = self.forcing_delta_encoder(origin_delta)
        if self.forcing_transition_mode == "origin_delta_linear":
            if self.forcing_delta_norm is None:
                raise RuntimeError("Origin-delta normalizer is missing")
            return self.forcing_delta_norm(base + encoded_origin)
        if self.forcing_transition_fusion is None or self.forcing_transition_trajectory is None:
            raise RuntimeError("Origin-delta TCN modules are missing")
        adjacent = torch.diff(raw, dim=2, prepend=raw[:, :, :1])
        encoded_adjacent = self.forcing_delta_encoder(adjacent)
        absolute = weather_nodes.index_select(0, weather_batch_index)
        fused = self.forcing_transition_fusion(
            torch.cat((absolute, encoded_origin, encoded_adjacent), -1)
        )
        batch, stations, steps, channels = fused.shape
        encoded = self.forcing_transition_trajectory(
            fused.reshape(batch * stations, steps, channels).transpose(1, 2)
        ).transpose(1, 2)
        return base + encoded.reshape(batch, stations, steps, channels)

    def _decode_weather_correction(
        self,
        response: torch.Tensor,
        target: torch.Tensor,
        *,
        batch: int,
        stations: int,
    ) -> torch.Tensor:
        latent = self.weather_response(torch.cat((response, target), -1))
        decoded = self.weather_trajectory(
            latent.reshape(
                batch * stations, self.forecast_steps, self.d_model
            ).transpose(1, 2)
        )
        correction = self.correction_head(decoded).squeeze(1)
        if self.readout_aux is not None:
            if self.response_readout_mode == "capacity_aux":
                correction = correction + self.readout_aux(decoded.transpose(1, 2)).squeeze(-1)
            else:
                level = self.readout_aux(decoded.mean(-1))
                correction = correction - correction.mean(-1, keepdim=True) + level
        return correction.reshape(batch, stations, self.forecast_steps)

    def forward(
        self,
        *,
        weather: torch.Tensor,
        weather_batch_index: torch.Tensor,
        weather_time_offset_hours: torch.Tensor,
        station_history: torch.Tensor,
        cluster_history: torch.Tensor,
        origin_phase: torch.Tensor,
        state_station_history: torch.Tensor | None = None,
        state_cluster_history: torch.Tensor | None = None,
        past_weather: torch.Tensor | None = None,
        past_weather_batch_index: torch.Tensor | None = None,
        past_weather_time_offset_hours: torch.Tensor | None = None,
        past_station_power: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        del past_weather_time_offset_hours
        batch, stations, input_steps = station_history.shape
        if stations != len(self.capacity_mw) or input_steps != self.input_history_steps:
            raise ValueError("station_history has an invalid shape")
        state_station_history = (
            station_history if state_station_history is None else state_station_history
        )
        state_cluster_history = (
            cluster_history if state_cluster_history is None else state_cluster_history
        )

        weather_nodes, spatial_attention = self._spatial_weather_nodes(weather)
        raw_station_weather = (
            self._bilinear_raw_station_weather(weather)
            if self.forcing_transition_mode != "legacy"
            else None
        )
        if raw_station_weather is None:
            weather_nodes = self.forcing_path(weather_nodes).index_select(
                0, weather_batch_index
            )
        else:
            weather_nodes = self._forcing_transition(
                weather_nodes,
                raw_station_weather,
                weather_batch_index,
                weather_time_offset_hours,
            )

        history = station_history[:, :, -self.history_visible_steps:]
        if self.source_role_mode == "anchor_weather":
            # The weather-only control retains the operationally necessary
            # forecast-origin anchor but cannot inspect the preceding history.
            history = station_history[:, :, -1:].expand_as(history)
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
        history_response = self.history_query(target, history_tokens, target_times, history_times)
        baseline_latent = self.baseline_response(torch.cat((history_response, target), -1))

        weather_response = self.weather_query(
            torch.zeros_like(target), weather_nodes, target_times, weather_time_offset_hours
        )
        state: torch.Tensor | None = None
        response_history_attention: torch.Tensor | None = None
        identification: torch.Tensor | None = None
        local_state = self.response_state_source not in {"none", "aggregate_power"}
        if (
            self.response_state_source == "local_power"
            and self.response_history_mode != "pooled_state"
        ):
            response_history = state_station_history[:, :, -self.state_visible_steps:]
            response_history = self._intervene_history(response_history)
            response_history_times = torch.arange(
                -(self.state_visible_steps - 1),
                1,
                device=station_history.device,
                dtype=station_history.dtype,
            ).reshape(1, -1).expand(batch, -1) / 4.0
            state, response_history_attention = self._lead_history_condition(
                response_history,
                weather_response,
                target,
                target_times,
                response_history_times,
            )
        elif self.response_state_source in {
            "local_power", "aggregate_power", "physical", "hierarchical"
        }:
            state = self._power_state(state_station_history, state_cluster_history)
            if self.response_state_source != "hierarchical":
                state = self._intervene(state, local=local_state)
        elif self.response_state_source in {"io_pair", "io_pair_id"}:
            if past_weather is None or past_weather_batch_index is None or past_station_power is None:
                raise ValueError("IO-pair state requires causal historical weather-power pairs")
            past_nodes = self.weather_encoder.encode_points(past_weather).permute(0, 2, 1, 3)
            past_nodes = self.forcing_path(past_nodes).index_select(0, past_weather_batch_index)
            state = self._intervene(
                self._io_state(past_nodes, past_station_power), local=True
            )
            if self.response_state_source == "io_pair_id":
                support = self._intervene(
                    self._io_state(past_nodes[:, :, :3], past_station_power[:, :, :3]),
                    local=True,
                )
                support_condition = support
                static_context = self._context(batch, stations, station_history.device)
                if static_context is not None:
                    support_condition = support_condition + static_context
                query, _ = self._conditioned_weather(past_nodes[:, :, 3:], support_condition)
                identification = torch.sigmoid(self.identification_head(query).squeeze(-1))

        context = self._context(batch, stations, station_history.device)
        response_specific_context = self.station_context_mode in {
            "metadata_residual", "metadata_film", "id_film"
        }
        if response_specific_context:
            if state is None:
                raise RuntimeError("Response-specific site context requires history state")
            weather_response, state = self._response_site_condition(
                weather_response, state, context
            )
            condition = state
        else:
            condition = state if context is None else context if state is None else state + context
        update = None
        additive_response_terms: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        if condition is not None and self.response_interaction_mode == "output_additive":
            expanded = (
                condition[:, :, None].expand(-1, -1, weather_response.shape[2], -1)
                if condition.ndim == 3
                else condition
            )
            zeros_weather = torch.zeros_like(weather_response)
            zeros_state = torch.zeros_like(expanded)
            additive_response_terms = (
                self.response_interaction(torch.cat((weather_response, zeros_state), -1)),
                self.response_interaction(torch.cat((zeros_weather, expanded), -1)),
                self.response_interaction(torch.cat((zeros_weather, zeros_state), -1)),
            )
            update = additive_response_terms[1] - additive_response_terms[2]
        elif (
            condition is not None
            and self.response_interaction_mode == "independent_output_additive"
        ):
            expanded = (
                condition[:, :, None].expand(-1, -1, weather_response.shape[2], -1)
                if condition.ndim == 3
                else condition
            )
            independent = self.response_interaction
            if not isinstance(independent, IndependentAdditiveResponse):
                raise RuntimeError("Independent additive response module is missing")
            zeros = torch.zeros_like(weather_response)
            additive_response_terms = (
                independent.weather(weather_response),
                independent.history(expanded),
                zeros,
            )
            update = additive_response_terms[1]
        elif condition is not None:
            weather_response, update = self._conditioned_weather(weather_response, condition)

        baseline_decoded = self.baseline_trajectory(
            baseline_latent.reshape(batch * stations, self.forecast_steps, self.d_model).transpose(1, 2)
        )
        baseline_delta = self.baseline_head(baseline_decoded).squeeze(1).reshape(
            batch, stations, self.forecast_steps
        )
        if self.source_role_mode == "history_only":
            # Use the matched history trajectory and current-CF anchor only.
            # A literal zero makes this path exactly invariant to every weather
            # tensor while retaining an unchanged parameter/state-dict layout.
            correction_logit = torch.zeros_like(baseline_delta)
        elif additive_response_terms is None:
            correction_logit = self._decode_weather_correction(
                weather_response, target, batch=batch, stations=stations
            )
        else:
            weather_main, state_main, origin = additive_response_terms
            weather_logit = self._decode_weather_correction(weather_main, target, batch=batch, stations=stations)
            history_logit = self._decode_weather_correction(state_main, target, batch=batch, stations=stations)
            origin_logit = self._decode_weather_correction(origin, target, batch=batch, stations=stations)
            correction_logit = weather_logit + history_logit - origin_logit
        if self.response_interaction_mode == "direct":
            if condition is None:
                raise RuntimeError("Direct response path requires a condition")
            expanded = (
                condition[:, :, None].expand(-1, -1, self.forecast_steps, -1)
                if condition.ndim == 3
                else condition
            )
            correction_logit = correction_logit + self.direct_state_head(
                torch.cat((expanded, target), -1)
            ).squeeze(-1)
        current = station_history[:, :, -1:].clamp(1e-4, 1.0 - 1e-4)
        baseline_logit = torch.logit(current) + baseline_delta
        baseline_factor = torch.sigmoid(baseline_logit)
        station_factor = torch.sigmoid(baseline_logit + correction_logit)
        if self.response_output_mode == "cf_residual":
            # The shared sigmoid above destroys H/W separability in CF space.
            # Use the same unbounded CF residual link for both matched roles;
            # clipping the sum would reintroduce an interaction. Legacy runs
            # keep their original output and checkpoint layout by default.
            station_factor = baseline_factor + correction_logit
        elif self.response_output_mode == "bounded_cf_mix":
            # Lead-only gate; each Add branch is independently bounded.
            mix = torch.sigmoid(self.output_mix_logits)[None, None, :]
            if self.source_role_mode == "history_only":
                history_factor = weather_factor = baseline_factor
            elif additive_response_terms is not None:
                history_factor = torch.sigmoid(baseline_logit + history_logit - origin_logit)
                weather_factor = torch.sigmoid(weather_logit)
            else:
                history_factor = baseline_factor
                weather_factor = torch.sigmoid(correction_logit)
            station_factor = mix * history_factor + (1. - mix) * weather_factor
        capacity = self.capacity_mw.reshape(1, -1, 1)
        cluster_factor = (station_factor * capacity).sum(1) / self.total_capacity_mw
        baseline_cluster = (baseline_factor * capacity).sum(1) / self.total_capacity_mw
        return {
            "station_factor": station_factor,
            "cluster_factor": cluster_factor,
            "baseline_station_factor": baseline_factor,
            "baseline_cluster_factor": baseline_cluster,
            "hourly_station_factor_unique": None,
            "local_weather_correction_logit": correction_logit,
            "interaction_update_abs_mean": (
                None if update is None else update.abs().mean(dim=(1, 3))
            ),
            "response_identification_station_factor": identification,
            "spatial_attention_unique": spatial_attention,
            "history_response_attention": response_history_attention,
        }


def build_local_state_identifiability_model(
    model_config: Mapping[str, Any],
    static: Mapping[str, Sequence[float]],
    *,
    history_steps: int = 24,
    forecast_steps: int = 24,
) -> LocalStateIdentifiabilityModel:
    return LocalStateIdentifiabilityModel(
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
        response_state_source=str(model_config.get("response_state_source", "none")),
        station_context_mode=str(model_config.get("station_context_mode", "none")),
        response_interaction_mode=str(model_config.get("response_interaction_mode", "none")),
        station_embedding_dim=int(model_config.get("station_embedding_dim", 8)),
        modulation_amplitude=float(model_config.get("modulation_amplitude", 0.25)),
        backbone_lr_multiplier=float(model_config.get("backbone_lr_multiplier", 0.25)),
        fine_tune_policy=str(model_config.get("fine_tune_policy", "all")),
        spatial_weather_mode=str(
            model_config.get("spatial_weather_mode", "precomputed_bilinear")
        ),
        station_neighborhood_cells=int(
            model_config.get("station_neighborhood_cells", 3)
        ),
        source_role_mode=str(model_config.get("source_role_mode", "full")),
        response_history_mode=str(
            model_config.get("response_history_mode", "pooled_state")
        ),
        forcing_transition_mode=str(
            model_config.get("forcing_transition_mode", "legacy")
        ),
        response_output_mode=str(model_config.get("response_output_mode", "legacy_logit")),
        response_readout_mode=str(model_config.get("response_readout_mode", "legacy")),
    )
