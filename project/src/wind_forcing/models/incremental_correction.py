from __future__ import annotations

from contextlib import nullcontext
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call

from wind_forcing.models.components import (
    RelativeTimeCrossAttention,
    SharedHistoryEncoder,
    TargetTimeEmbedding,
    TemporalResidualBlock,
    WeatherTargetQuery,
)
from wind_forcing.models.forced_evolution import (
    ForcedEvolutionModel,
    StationForcingEncoder,
    _forcing_encoder,
    build_forced_evolution_model,
)


def dct_basis(steps: int, components: int) -> torch.Tensor:
    component_count = int(components)
    if not 1 <= component_count <= steps:
        raise ValueError("temporal_components must be between 1 and forecast_steps")
    position = torch.arange(steps, dtype=torch.float32) + 0.5
    frequency = torch.arange(component_count, dtype=torch.float32).reshape(-1, 1)
    basis = torch.cos(torch.pi * frequency * position.reshape(1, -1) / steps)
    basis[0] *= 1.0 / np.sqrt(float(steps))
    if component_count > 1:
        basis[1:] *= np.sqrt(2.0 / steps)
    return basis


class IncrementalCorrectionModel(nn.Module):
    """Frozen history forecast plus one explicitly additive weather correction."""

    def __init__(
        self,
        baseline_model: ForcedEvolutionModel,
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
        time_query_mode: str,
        correction_conditioning: str,
        correction_history_mode: str,
        cross_farm_context_mode: str,
        cross_farm_neighbors: int,
        correction_fusion_mode: str,
        interaction_rank: int,
        matched_conditioning_backbone: bool,
        baseline_anchor_enabled: bool,
        projection_mode: str,
        temporal_components: int,
        spatial_basis: np.ndarray | None,
        freeze_history_baseline: bool,
        station_latitude: Sequence[float],
        station_longitude: Sequence[float],
        weather_view_mode: str = "legacy",
        weather_gate_mode: str = "fixed",
        specialist_mode: str = "shared_head",
        bracket_time_feature_mode: str = "relative_distances",
        bracket_mapper_activation: str = "gelu",
        trajectory_conditioning: str = "none",
        trajectory_interaction: bool = False,
        trajectory_input_mode: str = "matched",
        strict_additive: bool = False,
        output_clamp_min: float | None = 0.0,
    ) -> None:
        super().__init__()
        if trajectory_conditioning not in {"none", "level", "level_delta"}:
            raise ValueError("trajectory_conditioning must be none, level, or level_delta")
        self.trajectory_conditioning = trajectory_conditioning
        self.trajectory_interaction = bool(trajectory_interaction)
        if trajectory_input_mode not in {"matched", "zero", "level", "level_delta"}:
            raise ValueError("trajectory_input_mode must be matched, zero, level, or level_delta")
        self.trajectory_input_mode = trajectory_input_mode
        # In strict mode the weather branch may not receive any baseline-
        # derived trajectory representation.  This is an architectural
        # invariant used by the matched additive control, not a config label.
        self.strict_additive = bool(strict_additive)
        # The mechanism audit compares additive corrections in raw power
        # space.  Keeping the clamp configurable prevents a hidden output
        # nonlinearity from changing the meaning of the additive control.
        self.output_clamp_min = output_clamp_min
        valid_conditioning = {
            "history_weather",
            "history_only",
            "weather_only",
            "context_only",
        }
        if correction_conditioning not in valid_conditioning:
            raise ValueError(
                "correction_conditioning must be history_weather, history_only, "
                "weather_only, or context_only"
            )
        valid_history_modes = {
            "full_history",
            "recent_1h",
            "cluster_recent_1h",
            "cluster_current",
            "station_current",
            "cluster_current_ramp15m",
        }
        if correction_history_mode not in valid_history_modes:
            raise ValueError(
                "correction_history_mode must be full_history, recent_1h, "
                "cluster_recent_1h, cluster_current, station_current, or "
                "cluster_current_ramp15m"
            )
        if correction_history_mode in {"recent_1h", "cluster_recent_1h"} and history_steps < 4:
            raise ValueError(
                "recent_1h and cluster_recent_1h require at least four history steps"
            )
        if correction_history_mode == "cluster_current_ramp15m" and history_steps < 2:
            raise ValueError("cluster_current_ramp15m requires at least two history steps")
        valid_cross_farm_modes = {
            "none",
            "self_duplicate",
            "geographic",
            "global_other",
            "weather_coherent",
            "composition_control",
            "composition_statistics",
            "composition_deepsets",
        }
        if cross_farm_context_mode not in valid_cross_farm_modes:
            raise ValueError(
                "cross_farm_context_mode must be none, self_duplicate, "
                "geographic, global_other, weather_coherent, "
                "composition_control, composition_statistics, or "
                "composition_deepsets"
            )
        if cross_farm_context_mode != "none" and correction_history_mode != "recent_1h":
            raise ValueError(
                "cross-farm context is defined only for correction_history_mode=recent_1h"
            )
        if (
            cross_farm_context_mode != "none"
            and not 1 <= cross_farm_neighbors < len(station_capacity_mw)
        ):
            raise ValueError("cross_farm_neighbors must be between 1 and stations-1")
        valid_fusion_modes = {
            "blackbox",
            "additive",
            "multiplicative",
            "horizon_gated_multiplicative",
            "asymmetric_modulation",
            "horizon_film",
            "horizon_conditioned_query",
            "horizon_attention_bias",
            "horizon_conditional_lora",
        }
        if correction_fusion_mode not in valid_fusion_modes:
            raise ValueError(
                "correction_fusion_mode must be blackbox, additive, "
                "multiplicative, horizon_gated_multiplicative, or "
                "asymmetric_modulation, horizon_film, "
                "horizon_conditioned_query, horizon_attention_bias, or "
                "horizon_conditional_lora"
            )
        if interaction_rank < 1 or interaction_rank > d_model:
            raise ValueError("interaction_rank must be between 1 and d_model")
        attention_fusions = {
            "horizon_conditioned_query",
            "horizon_attention_bias",
        }
        if correction_fusion_mode in attention_fusions and time_query_mode != "full_attention":
            raise ValueError(
                "Conditioned-query and attention-bias fusion require full_attention"
            )
        if (
            correction_fusion_mode != "blackbox"
            and correction_conditioning != "history_weather"
        ):
            raise ValueError(
                "Explicit correction fusion modes require history_weather conditioning"
            )
        if (
            correction_conditioning in {"history_only", "context_only"}
            and not matched_conditioning_backbone
        ):
            raise ValueError(
                "history_only and context_only require matched_conditioning_backbone"
            )
        if projection_mode not in {"full", "temporal", "spatial", "joint"}:
            raise ValueError("projection_mode must be full, temporal, spatial, or joint")
        if projection_mode in {"spatial", "joint"} and spatial_basis is None:
            raise ValueError("A spatial basis is required by the selected projection_mode")
        if weather_view_mode not in {"legacy", "full", "nearest", "horizon_mix"}:
            raise ValueError(
                "weather_view_mode must be legacy, full, nearest, or horizon_mix"
            )
        if weather_gate_mode not in {"fixed", "monotonic_sigmoid"}:
            raise ValueError(
                "weather_gate_mode must be fixed or monotonic_sigmoid"
            )
        if specialist_mode not in {"shared_head", "dual_head"}:
            raise ValueError("specialist_mode must be shared_head or dual_head")
        if specialist_mode == "dual_head" and correction_fusion_mode != "blackbox":
            raise ValueError("dual_head specialists require blackbox correction fusion")
        if specialist_mode == "dual_head" and weather_view_mode == "legacy":
            raise ValueError("dual_head specialists require an explicit weather_view_mode")
        if weather_view_mode != "legacy" and correction_fusion_mode != "blackbox":
            raise ValueError(
                "Explicit weather-view modes require blackbox correction fusion"
            )
        if weather_view_mode != "horizon_mix" and weather_gate_mode != "fixed" and specialist_mode != "dual_head":
            raise ValueError(
                "A learned weather gate requires horizon_mix or dual_head specialists"
            )
        self.baseline_model = baseline_model
        self.freeze_history_baseline = bool(freeze_history_baseline)
        self.baseline_model.requires_grad_(not self.freeze_history_baseline)
        self.forcing_encoder = forcing_encoder
        self.history_steps = int(history_steps)
        self.forecast_steps = int(forecast_steps)
        self.d_model = int(d_model)
        self.correction_conditioning = correction_conditioning
        self.correction_history_mode = correction_history_mode
        self.cross_farm_context_mode = cross_farm_context_mode
        self.cross_farm_neighbors = int(cross_farm_neighbors)
        self.composition_modes = {
            "composition_control",
            "composition_statistics",
            "composition_deepsets",
        }
        self.correction_fusion_mode = correction_fusion_mode
        self.interaction_rank = int(interaction_rank)
        self.matched_conditioning_backbone = bool(matched_conditioning_backbone)
        self.baseline_anchor_enabled = bool(baseline_anchor_enabled)
        self.projection_mode = projection_mode
        self.weather_view_mode = weather_view_mode
        self.weather_gate_mode = weather_gate_mode
        self.specialist_mode = specialist_mode
        self._anchor_state_buffers: dict[str, str] = {}
        if self.baseline_anchor_enabled:
            for index, (name, value) in enumerate(
                self.baseline_model.state_dict().items()
            ):
                buffer_name = f"_baseline_anchor_state_{index:04d}"
                self.register_buffer(
                    buffer_name,
                    value.detach().clone(),
                    persistent=True,
                )
                self._anchor_state_buffers[name] = buffer_name
        capacity = torch.as_tensor(station_capacity_mw, dtype=torch.float32)
        self.register_buffer("capacity_mw", capacity)
        self.register_buffer("total_capacity_mw", capacity.sum())
        station_lat = torch.as_tensor(station_latitude, dtype=torch.float32)
        station_lon = torch.as_tensor(station_longitude, dtype=torch.float32)
        if station_lat.shape != capacity.shape or station_lon.shape != capacity.shape:
            raise ValueError("Station coordinates are inconsistent with capacities")
        mean_latitude = torch.deg2rad(station_lat.mean())
        dx = (
            (station_lon[:, None] - station_lon[None, :])
            * torch.cos(mean_latitude)
            * 111.32
        )
        dy = (station_lat[:, None] - station_lat[None, :]) * 111.32
        distance = torch.sqrt(dx.square() + dy.square())
        distance.fill_diagonal_(torch.inf)
        metadata_neighbor_count = min(
            max(1, self.cross_farm_neighbors), len(station_capacity_mw) - 1
        )
        geographic_neighbors = distance.topk(
            metadata_neighbor_count, largest=False, dim=-1
        ).indices
        self.register_buffer(
            "geographic_neighbor_index", geographic_neighbors, persistent=False
        )
        self.register_buffer("station_distance_km", distance, persistent=False)
        self.register_buffer(
            "history_time_hours",
            torch.arange(-(history_steps - 1), 1, dtype=torch.float32) / 4.0,
        )
        self.register_buffer(
            "target_lead_hours",
            torch.arange(1, forecast_steps + 1, dtype=torch.float32) / 4.0,
        )
        self.register_buffer(
            "temporal_basis",
            dct_basis(forecast_steps, temporal_components),
        )
        if spatial_basis is None:
            basis = torch.empty(0, len(capacity), dtype=torch.float32)
        else:
            basis = torch.as_tensor(np.asarray(spatial_basis), dtype=torch.float32)
            if basis.ndim != 2 or basis.shape[1] != len(capacity):
                raise ValueError("spatial_basis must have shape [components, stations]")
            gram = basis @ basis.T
            if not torch.allclose(gram, torch.eye(len(basis)), atol=2e-4, rtol=2e-4):
                raise ValueError("spatial_basis rows must be orthonormal")
        self.register_buffer("spatial_basis", basis)

        history_input_channels = 1 if cross_farm_context_mode == "none" else 2
        self.station_history_encoder = SharedHistoryEncoder(
            d_model,
            history_tcn_layers,
            dropout,
            input_channels=history_input_channels,
        )
        self.cluster_history_encoder = SharedHistoryEncoder(
            d_model, history_tcn_layers, dropout
        )
        # The three regional-composition modes instantiate exactly the same
        # active modules.  The control sees only redundant aggregate history;
        # the statistics mode additionally sees deterministic distribution
        # summaries; and DeepSets additionally sees the station-history set.
        # This nesting isolates information gain without a capacity change.
        self.composition_statistics_encoder: nn.Module | None = None
        self.composition_element_encoder: nn.Module | None = None
        self.composition_set_encoder: nn.Module | None = None
        self.composition_global_fusion: nn.Module | None = None
        if cross_farm_context_mode in self.composition_modes:
            self.composition_statistics_encoder = nn.Sequential(
                nn.Linear(25, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )
            self.composition_element_encoder = nn.Sequential(
                nn.Linear(5, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.composition_set_encoder = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )
            self.composition_global_fusion = nn.Sequential(
                nn.Linear(3 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        self.target_embedding = TargetTimeEmbedding(
            d_model, forecast_hours=forecast_steps / 4.0
        )
        if self.trajectory_conditioning != "none":
            self.trajectory_projection = nn.Sequential(
                nn.Linear(3, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.trajectory_projection = None
        self.trajectory_response_projection: nn.Module | None = None
        if self.trajectory_conditioning != "none":
            # Ensure the trajectory feature is live in every correction fusion
            # branch, including explicit additive/non-additive controls.
            self.trajectory_response_projection = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
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
        if self.weather_gate_mode == "monotonic_sigmoid":
            self.weather_gate_center_raw = nn.Parameter(torch.zeros(()))
            initial_slope = torch.tensor(3.5, dtype=torch.float32)
            self.weather_gate_slope_raw = nn.Parameter(
                torch.log(torch.expm1(initial_slope))
            )
        else:
            self.register_parameter("weather_gate_center_raw", None)
            self.register_parameter("weather_gate_slope_raw", None)
        if correction_fusion_mode == "blackbox":
            input_count = (
                4
                if self.matched_conditioning_backbone
                or correction_conditioning == "history_weather"
                else 2
            )
            if self.trajectory_conditioning != "none":
                input_count += 1
            if self.trajectory_interaction:
                if self.trajectory_conditioning == "none":
                    raise ValueError("trajectory_interaction requires trajectory_conditioning")
                input_count += 1
            self.correction_response = nn.Sequential(
                nn.Linear(input_count * d_model, 2 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.full_correction_response: nn.Module | None = None
            if self.specialist_mode == "dual_head":
                self.full_correction_response = copy.deepcopy(self.correction_response)
                self.full_correction_response.apply(self._reset_cloned_module)
        else:
            # The explicit modes instantiate the same modules so their parameter
            # counts are exactly matched.  With rank=d_model/2 this block differs
            # from the black-box response by only 452 parameters at d_model=64.
            def branch_response(input_count: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Linear(input_count * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                )

            # Legacy explicit modes keep their original d_model/2 rank so the
            # new gated multiplicative model is paired directly with the
            # already-trained ungated multiplicative checkpoints.  The new
            # control-object operators use the common configured bottleneck.
            legacy_interaction_rank = max(1, d_model // 2)
            self.weather_main_response = branch_response(2)
            self.state_main_response = branch_response(3)
            self.interaction_state_projection = nn.Linear(
                3 * d_model, legacy_interaction_rank
            )
            self.interaction_weather_projection = nn.Linear(
                2 * d_model, legacy_interaction_rank
            )
            self.interaction_response = nn.Sequential(
                nn.Linear(legacy_interaction_rank, d_model),
                nn.LayerNorm(d_model),
            )
            # Horizon-adaptive operators use one independent positive gate per
            # lead.  Legacy explicit checkpoints contained four unused gate
            # logits, so preserve that shape for strict checkpoint loading.
            # For the new operators, 2*sigmoid(0)=1 matches the ungated
            # operator at initialization without monotonicity or smoothness.
            horizon_control_modes = {
                "horizon_film",
                "horizon_gated_multiplicative",
                "horizon_conditioned_query",
                "horizon_attention_bias",
                "horizon_conditional_lora",
            }
            gate_count = (
                forecast_steps
                if correction_fusion_mode in horizon_control_modes
                else 4
            )
            self.horizon_gate_logits = nn.Parameter(torch.zeros(gate_count))

            rank = self.interaction_rank
            self.film_state_down: nn.Module | None = None
            self.film_state_output: nn.Module | None = None
            self.query_state_down: nn.Module | None = None
            self.query_state_output: nn.Module | None = None
            self.attention_state_projection: nn.Module | None = None
            self.attention_weather_projection: nn.Module | None = None
            self.lora_state_projection: nn.Module | None = None
            self.lora_down: nn.Module | None = None
            self.lora_up: nn.Module | None = None
            self.horizon_interaction_state_projection: nn.Module | None = None
            if correction_fusion_mode == "horizon_film":
                self.film_state_down = nn.Sequential(
                    nn.Linear(2 * d_model, rank), nn.GELU()
                )
                self.film_state_output = nn.Linear(rank, 2 * d_model)
            elif correction_fusion_mode == "horizon_gated_multiplicative":
                self.horizon_interaction_state_projection = nn.Linear(
                    2 * d_model, legacy_interaction_rank
                )
            elif correction_fusion_mode == "horizon_conditioned_query":
                self.query_state_down = nn.Sequential(
                    nn.Linear(2 * d_model, rank), nn.GELU()
                )
                self.query_state_output = nn.Linear(rank, d_model)
            elif correction_fusion_mode == "horizon_attention_bias":
                self.attention_state_projection = nn.Linear(2 * d_model, rank)
                self.attention_weather_projection = nn.Linear(d_model, rank)
            elif correction_fusion_mode == "horizon_conditional_lora":
                self.lora_state_projection = nn.Linear(2 * d_model, rank)
                self.lora_down = nn.Linear(d_model, rank, bias=False)
                self.lora_up = nn.Linear(rank, d_model, bias=False)
        self.trajectory_decoder = nn.Sequential(
            *[
                TemporalResidualBlock(d_model, 2**layer, dropout)
                for layer in range(trajectory_tcn_layers)
            ]
        )
        self.correction_head = nn.Conv1d(d_model, 1, kernel_size=1)
        nn.init.normal_(self.correction_head.weight, std=1e-3)
        nn.init.zeros_(self.correction_head.bias)
        self.full_trajectory_decoder: nn.Module | None = None
        self.full_correction_head: nn.Conv1d | None = None
        if self.specialist_mode == "dual_head":
            self.full_trajectory_decoder = copy.deepcopy(self.trajectory_decoder)
            self.full_trajectory_decoder.apply(self._reset_cloned_module)
            self.full_correction_head = copy.deepcopy(self.correction_head)
            nn.init.normal_(self.full_correction_head.weight, std=1e-3)
            nn.init.zeros_(self.full_correction_head.bias)

    @staticmethod
    def _reset_cloned_module(module: nn.Module) -> None:
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()

    def _decode_response(
        self, response: torch.Tensor, *, include_head_bias: bool
    ) -> torch.Tensor:
        batch, station_count = response.shape[:2]
        decoded = self.trajectory_decoder(
            response.reshape(
                batch * station_count, self.forecast_steps, self.d_model
            ).transpose(1, 2)
        )
        raw = F.conv1d(
            decoded,
            self.correction_head.weight,
            self.correction_head.bias if include_head_bias else None,
        )
        return raw.squeeze(1).reshape(batch, station_count, self.forecast_steps)

    def _decode_full_response(
        self, response: torch.Tensor, *, include_head_bias: bool
    ) -> torch.Tensor:
        if self.full_trajectory_decoder is None or self.full_correction_head is None:
            raise RuntimeError("The full specialist decoder is missing")
        batch, station_count = response.shape[:2]
        decoded = self.full_trajectory_decoder(
            response.reshape(
                batch * station_count, self.forecast_steps, self.d_model
            ).transpose(1, 2)
        )
        raw = F.conv1d(
            decoded,
            self.full_correction_head.weight,
            self.full_correction_head.bias if include_head_bias else None,
        )
        return raw.squeeze(1).reshape(batch, station_count, self.forecast_steps)

    def _weather_view_gate(self, reference: torch.Tensor) -> torch.Tensor:
        lead = self.target_lead_hours.to(reference)
        if self.weather_gate_mode == "fixed":
            gate = ((lead - 1.0) / 2.0).clamp(0.0, 1.0)
        else:
            if self.weather_gate_center_raw is None or self.weather_gate_slope_raw is None:
                raise RuntimeError("Learned weather-gate parameters are missing")
            center = 1.0 + 2.0 * torch.sigmoid(
                self.weather_gate_center_raw.to(reference)
            )
            slope = 0.5 + F.softplus(self.weather_gate_slope_raw.to(reference))
            gate = torch.sigmoid(slope * (lead - center))
        return gate.reshape(1, 1, self.forecast_steps)

    def weather_gate_parameters(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.weather_gate_mode != "monotonic_sigmoid":
            return None, None
        if self.weather_gate_center_raw is None or self.weather_gate_slope_raw is None:
            raise RuntimeError("Learned weather-gate parameters are missing")
        center = 1.0 + 2.0 * torch.sigmoid(self.weather_gate_center_raw)
        slope = 0.5 + F.softplus(self.weather_gate_slope_raw)
        return center, slope

    def _horizon_interaction_gate(self, reference: torch.Tensor) -> torch.Tensor:
        knots = 2.0 * torch.sigmoid(self.horizon_gate_logits.to(reference))
        return knots.reshape(1, 1, self.forecast_steps)

    def _condition_correction_history(
        self,
        station_history: torch.Tensor,
        cluster_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose only the requested state summary to the correction pathway."""
        mode = self.correction_history_mode
        if mode == "full_history":
            return station_history, cluster_history
        if mode == "recent_1h":
            recent_station = station_history[..., -4:]
            recent_cluster = cluster_history[..., -4:]
            station_prefix = recent_station[..., :1].expand(
                *recent_station.shape[:-1], self.history_steps - 4
            )
            cluster_prefix = recent_cluster[..., :1].expand(
                *recent_cluster.shape[:-1], self.history_steps - 4
            )
            return (
                torch.cat((station_prefix, recent_station), dim=-1),
                torch.cat((cluster_prefix, recent_cluster), dim=-1),
            )
        if mode == "cluster_recent_1h":
            recent_cluster = cluster_history[..., -4:]
            cluster_prefix = recent_cluster[..., :1].expand(
                *recent_cluster.shape[:-1], self.history_steps - 4
            )
            cluster = torch.cat((cluster_prefix, recent_cluster), dim=-1)
            station = cluster[:, None, :].expand(
                -1, station_history.shape[1], -1
            )
            return station, cluster
        if mode == "station_current":
            return (
                station_history[..., -1:].expand(-1, -1, self.history_steps),
                cluster_history[..., -1:].expand(-1, self.history_steps),
            )
        if mode == "cluster_current":
            current = cluster_history[..., -1:].expand(-1, self.history_steps)
        else:
            previous = cluster_history[..., -2:-1]
            current = torch.cat(
                (
                    previous.expand(-1, self.history_steps - 1),
                    cluster_history[..., -1:],
                ),
                dim=-1,
            )
        station = current[:, None, :].expand(
            -1, station_history.shape[1], -1
        )
        return station, current

    def train(self, mode: bool = True) -> "IncrementalCorrectionModel":
        super().train(mode)
        if self.freeze_history_baseline:
            self.baseline_model.eval()
        return self

    def _cross_farm_history(
        self,
        station_history: torch.Tensor,
        weather_station_trajectory: torch.Tensor,
        weather_batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Construct a parameter-free one-hop context for every target farm."""
        mode = self.cross_farm_context_mode
        if mode == "none":
            raise RuntimeError("Cross-farm history was requested with mode=none")
        if mode == "self_duplicate":
            # This is the information-matched control: the second input channel
            # is active but contains no information beyond the self trajectory.
            return station_history
        if mode in self.composition_modes:
            # Regional composition is injected through its own matched global
            # branch.  Keep the per-target second channel strictly redundant
            # so these models retain the same self-state input as G1.
            return station_history
        if mode == "global_other":
            return (
                station_history.sum(dim=1, keepdim=True) - station_history
            ) / float(station_history.shape[1] - 1)
        if mode == "geographic":
            return station_history[:, self.geographic_neighbor_index, :].mean(dim=2)

        if weather_station_trajectory.ndim != 3:
            raise ValueError(
                "weather_station_trajectory must have shape [cycles,stations,time]"
            )
        trajectory = weather_station_trajectory.detach().float()
        pairwise_mse = (
            trajectory[:, :, None, :] - trajectory[:, None, :, :]
        ).square().mean(dim=-1)
        diagonal = torch.eye(
            pairwise_mse.shape[1], dtype=torch.bool, device=pairwise_mse.device
        ).unsqueeze(0)
        pairwise_mse = pairwise_mse.masked_fill(diagonal, torch.inf)
        unique_neighbors = pairwise_mse.topk(
            self.cross_farm_neighbors, largest=False, dim=-1
        ).indices
        sample_neighbors = unique_neighbors.index_select(0, weather_batch_index)
        batch_index = torch.arange(
            station_history.shape[0], device=station_history.device
        ).reshape(-1, 1, 1)
        return station_history[batch_index, sample_neighbors, :].mean(dim=2)

    def _regional_composition_inputs(
        self,
        station_history: torch.Tensor,
        cluster_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nested statistics and set inputs for composition ablations.

        The statistics vector has five summaries (mean, population standard
        deviation, and three quartiles) at each of four recent times plus the
        same five summaries for the final 15-minute ramp.  The set input holds
        four recent values and that ramp for every station.  Inputs are already
        capacity-normalized factors in the model cache.
        """
        recent_station = station_history[..., -4:].float()
        recent_cluster = cluster_history[..., -4:].float()
        cluster_ramp = recent_cluster[..., -1:] - recent_cluster[..., -2:-1]

        zero_cluster = torch.zeros_like(recent_cluster)
        redundant_time = torch.stack(
            (
                recent_cluster,
                zero_cluster,
                recent_cluster,
                recent_cluster,
                recent_cluster,
            ),
            dim=-1,
        )
        redundant_ramp = torch.cat(
            (
                cluster_ramp,
                torch.zeros_like(cluster_ramp),
                cluster_ramp,
                cluster_ramp,
                cluster_ramp,
            ),
            dim=-1,
        )
        redundant_statistics = torch.cat(
            (redundant_time.reshape(len(station_history), -1), redundant_ramp),
            dim=-1,
        )
        redundant_set = torch.cat((recent_cluster, cluster_ramp), dim=-1)[:, None]
        redundant_set = redundant_set.expand(-1, station_history.shape[1], -1)

        if self.cross_farm_context_mode == "composition_control":
            return redundant_statistics, redundant_set

        quantiles = torch.quantile(
            recent_station,
            torch.tensor(
                (0.25, 0.50, 0.75),
                device=recent_station.device,
                dtype=recent_station.dtype,
            ),
            dim=1,
        ).permute(1, 2, 0)
        time_statistics = torch.cat(
            (
                recent_station.mean(dim=1, keepdim=False).unsqueeze(-1),
                recent_station.std(dim=1, unbiased=False).unsqueeze(-1),
                quantiles,
            ),
            dim=-1,
        )
        station_ramp = recent_station[..., -1] - recent_station[..., -2]
        ramp_quantiles = torch.quantile(
            station_ramp,
            torch.tensor(
                (0.25, 0.50, 0.75),
                device=station_ramp.device,
                dtype=station_ramp.dtype,
            ),
            dim=1,
        ).transpose(0, 1)
        ramp_statistics = torch.cat(
            (
                station_ramp.mean(dim=1, keepdim=True),
                station_ramp.std(dim=1, unbiased=False, keepdim=True),
                ramp_quantiles,
            ),
            dim=-1,
        )
        actual_statistics = torch.cat(
            (time_statistics.reshape(len(station_history), -1), ramp_statistics),
            dim=-1,
        )
        if self.cross_farm_context_mode == "composition_statistics":
            return actual_statistics, redundant_set
        actual_set = torch.cat((recent_station, station_ramp.unsqueeze(-1)), dim=-1)
        return actual_statistics, actual_set

    def _regional_composition_state(
        self,
        station_history: torch.Tensor,
        cluster_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.composition_statistics_encoder is None
            or self.composition_element_encoder is None
            or self.composition_set_encoder is None
        ):
            raise RuntimeError("Regional composition encoders are missing")
        statistics_input, set_input = self._regional_composition_inputs(
            station_history, cluster_history
        )
        statistics_state = self.composition_statistics_encoder(statistics_input)
        element_state = self.composition_element_encoder(set_input)
        set_state = self.composition_set_encoder(
            torch.cat(
                (element_state.mean(dim=1), element_state.max(dim=1).values),
                dim=-1,
            )
        )
        return statistics_state, set_state, statistics_input

    def load_history_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        model_config = checkpoint.get("model_config", {})
        if str(model_config.get("architecture")) != "forced_evolution" or str(
            model_config.get("fusion_mode")
        ) != "history_only":
            raise ValueError("The incremental baseline checkpoint must be a history-only forced-evolution model")
        self.baseline_model.load_state_dict(checkpoint["model_state"], strict=True)
        if self.baseline_anchor_enabled:
            with torch.no_grad():
                for name, value in self.baseline_model.state_dict().items():
                    getattr(self, self._anchor_state_buffers[name]).copy_(value)
        self.baseline_model.requires_grad_(not self.freeze_history_baseline)
        if self.freeze_history_baseline:
            self.baseline_model.eval()

    def _anchored_baseline(
        self,
        *,
        station_history: torch.Tensor,
        cluster_history: torch.Tensor,
        origin_phase: torch.Tensor,
    ) -> Mapping[str, torch.Tensor | None] | None:
        if not self.baseline_anchor_enabled:
            return None
        anchor_state = {
            name: getattr(self, buffer_name)
            for name, buffer_name in self._anchor_state_buffers.items()
        }
        was_training = self.baseline_model.training
        self.baseline_model.eval()
        try:
            with torch.no_grad():
                return functional_call(
                    self.baseline_model,
                    anchor_state,
                    kwargs={
                        "station_history": station_history,
                        "cluster_history": cluster_history,
                        "origin_phase": origin_phase,
                    },
                    strict=True,
                )
        finally:
            self.baseline_model.train(was_training)

    def _project_correction(self, correction_factor: torch.Tensor) -> torch.Tensor:
        correction = correction_factor
        if self.projection_mode in {"temporal", "joint"}:
            coefficients = torch.einsum("bst,qt->bsq", correction, self.temporal_basis)
            correction = torch.einsum("bsq,qt->bst", coefficients, self.temporal_basis)
        if self.projection_mode in {"spatial", "joint"}:
            correction_mw = correction * self.capacity_mw.reshape(1, -1, 1)
            coefficients = torch.einsum("bst,ms->bmt", correction_mw, self.spatial_basis)
            correction_mw = torch.einsum("bmt,ms->bst", coefficients, self.spatial_basis)
            correction = correction_mw / self.capacity_mw.clamp_min(1e-6).reshape(1, -1, 1)
        return correction

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
        batch, station_count = station_history.shape[:2]
        if station_history.shape != (batch, len(self.capacity_mw), self.history_steps):
            raise ValueError("station_history has an invalid shape")
        if cluster_history.shape != (batch, self.history_steps):
            raise ValueError("cluster_history has an invalid shape")
        if weather_batch_index.shape != (batch,):
            raise ValueError("weather_batch_index has an invalid shape")
        baseline_context = (
            torch.no_grad() if self.freeze_history_baseline else nullcontext()
        )
        with baseline_context:
            baseline = self.baseline_model(
                station_history=station_history,
                cluster_history=cluster_history,
                origin_phase=origin_phase,
            )
        anchor_baseline = self._anchored_baseline(
            station_history=station_history,
            cluster_history=cluster_history,
            origin_phase=origin_phase,
        )
        baseline_station = baseline["station_factor"]
        assert baseline_station is not None

        # Power-space trajectory condition.  The first difference uses the
        # actually observed origin power; all later differences use the
        # frozen history-only forecast itself.  No future target is accessed.
        trajectory_token = None
        if self.trajectory_conditioning != "none":
            origin_power = station_history[..., -1:].to(baseline_station)
            previous = torch.cat((origin_power, baseline_station[..., :-1]), dim=-1)
            delta = baseline_station - previous
            input_mode = self.trajectory_input_mode
            if input_mode == "matched":
                input_mode = self.trajectory_conditioning
            if input_mode == "zero":
                baseline_value = torch.zeros_like(baseline_station)
                delta = torch.zeros_like(delta)
            elif input_mode == "level":
                baseline_value = baseline_station
                delta = torch.zeros_like(delta)
            else:
                baseline_value = baseline_station
            lead = self.target_lead_hours.to(baseline_station).reshape(
                1, 1, self.forecast_steps, 1
            ).expand(batch, station_count, -1, -1) / 6.0
            raw_trajectory = torch.cat(
                (baseline_value.unsqueeze(-1), delta.unsqueeze(-1), lead), dim=-1
            )
            trajectory_token = self.trajectory_projection(raw_trajectory)

        unique_weather = self.forcing_encoder(weather)
        weather_forcing = unique_weather["forcing"].index_select(0, weather_batch_index)
        weather_query_time_hours = weather_time_offset_hours
        station_embedding = self.forcing_encoder.station_embedding()
        correction_station_history, correction_cluster_history = (
            self._condition_correction_history(station_history, cluster_history)
        )
        if self.cross_farm_context_mode == "none":
            station_encoder_input = correction_station_history
        else:
            cross_farm_history = self._cross_farm_history(
                correction_station_history,
                unique_weather["hourly_station_factor"],
                weather_batch_index,
            )
            station_encoder_input = torch.stack(
                (correction_station_history, cross_farm_history), dim=2
            )
        history_tokens, station_state = self.station_history_encoder(
            station_encoder_input, station_embedding
        )
        _, global_state = self.cluster_history_encoder(correction_cluster_history)
        composition_statistics_state = None
        composition_set_state = None
        composition_statistics_input = None
        if self.cross_farm_context_mode in self.composition_modes:
            if self.composition_global_fusion is None:
                raise RuntimeError("Regional composition fusion is missing")
            (
                composition_statistics_state,
                composition_set_state,
                composition_statistics_input,
            ) = self._regional_composition_state(
                correction_station_history, correction_cluster_history
            )
            global_state = self.composition_global_fusion(
                torch.cat(
                    (
                        global_state,
                        composition_statistics_state,
                        composition_set_state,
                    ),
                    dim=-1,
                )
            )
        target_times = self.target_lead_hours.to(station_history).reshape(1, -1).expand(
            batch, -1
        )
        if self.weather_query.mode == "raw_linear":
            # Samples sharing an NWP cycle differ only in origin phase.  Build
            # the union of their requested quarter-hour coordinates, run the
            # expensive grid/station encoder once per unique cycle and time,
            # then gather each sample's 24-point path before the shared path
            # encoder.  This is numerically the same raw-space interpolation
            # baseline without encoding four duplicate origin views.
            query_coordinate = target_times - weather_time_offset_hours[:, :1]
            union_coordinate = torch.unique(query_coordinate.flatten(), sorted=True)
            context_coordinate = (
                weather_time_offset_hours - weather_time_offset_hours[:, :1]
            )
            reference_context = context_coordinate[:1]
            if not torch.allclose(
                context_coordinate,
                reference_context.expand_as(context_coordinate),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    "raw_linear requires common native forcing intervals within a batch"
                )
            unique_cycles = weather.shape[0]
            union_steps = union_coordinate.numel()
            union_times = union_coordinate.reshape(1, -1).expand(unique_cycles, -1)
            native_times = reference_context.expand(unique_cycles, -1)
            lower_index, upper_index = self.weather_query._bracketing_indices(
                union_times, native_times
            )

            def gather_raw(indices: torch.Tensor) -> torch.Tensor:
                gather_index = indices.reshape(
                    unique_cycles, union_steps, 1, 1, 1
                ).expand(
                    unique_cycles,
                    union_steps,
                    *weather.shape[2:],
                )
                return torch.gather(weather, dim=1, index=gather_index)

            lower_weather = gather_raw(lower_index)
            upper_weather = gather_raw(upper_index)
            lower_time = torch.gather(native_times, 1, lower_index)
            upper_time = torch.gather(native_times, 1, upper_index)
            duration = upper_time - lower_time
            phase = torch.where(
                duration.abs() > 1e-8,
                (union_times - lower_time) / duration.clamp_min(1e-8),
                torch.zeros_like(duration),
            ).clamp(0.0, 1.0)
            raw_target_weather = lower_weather + phase.reshape(
                unique_cycles, union_steps, 1, 1, 1
            ).to(lower_weather) * (upper_weather - lower_weather)
            aligned_frames = self.forcing_encoder.encode_station_frames(
                raw_target_weather
            )["station_frames"]
            assert aligned_frames is not None
            sampled_frames = aligned_frames.index_select(0, weather_batch_index)
            query_index = torch.searchsorted(union_coordinate, query_coordinate)
            recovered_coordinate = union_coordinate.index_select(
                0, query_index.flatten()
            ).reshape_as(query_coordinate)
            if not torch.allclose(
                recovered_coordinate, query_coordinate, atol=1e-6, rtol=0.0
            ):
                raise RuntimeError("Failed to recover raw-linear target coordinates")
            gather_index = query_index[:, None, :, None].expand(
                batch,
                station_count,
                self.forecast_steps,
                self.d_model,
            )
            sampled_frames = torch.gather(sampled_frames, dim=2, index=gather_index)
            weather_forcing = self.forcing_encoder.path_encoder(sampled_frames)
            weather_query_time_hours = target_times
        target_token = self.target_embedding(target_times, origin_phase)
        target_token = target_token[:, None] + station_embedding.reshape(
            1, station_count, 1, self.d_model
        )
        target_token = target_token.expand(-1, -1, self.forecast_steps, -1)
        uses_weather = self.correction_conditioning in {
            "history_weather",
            "weather_only",
        }
        uses_history = self.correction_conditioning in {
            "history_weather",
            "history_only",
        }
        if uses_history:
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
        else:
            history_response = torch.zeros_like(target_token)
            global_response = torch.zeros_like(target_token)
        state_input = torch.cat(
            (history_response, global_response, target_token), dim=-1
        )
        # New horizon-control operators receive only encoded recent state here.
        # Target time enters exclusively through the common gate and weather
        # path, so an operator cannot become an unlabelled horizon-only branch.
        state_control = torch.cat(
            (
                station_state[:, :, None].expand(
                    -1, -1, self.forecast_steps, -1
                ),
                global_state[:, None, None].expand(
                    -1, station_count, self.forecast_steps, -1
                ),
            ),
            dim=-1,
        )
        horizon_modes = {
            "horizon_film",
            "horizon_gated_multiplicative",
            "horizon_conditioned_query",
            "horizon_attention_bias",
            "horizon_conditional_lora",
        }
        interaction_gate = (
            self._horizon_interaction_gate(target_token)
            if self.correction_fusion_mode in horizon_modes
            else None
        )
        conditioned_query = None
        conditioned_attention_bias = None
        if self.correction_fusion_mode == "horizon_conditioned_query":
            if self.query_state_down is None or self.query_state_output is None:
                raise RuntimeError("Conditioned-query modules are missing")
            assert interaction_gate is not None
            query_update = self.query_state_output(
                self.query_state_down(state_control)
            )
            conditioned_query = (
                target_token + interaction_gate.unsqueeze(-1) * query_update
            )
        elif self.correction_fusion_mode == "horizon_attention_bias":
            if (
                self.attention_state_projection is None
                or self.attention_weather_projection is None
            ):
                raise RuntimeError("Attention-bias modules are missing")
            assert interaction_gate is not None
            state_compatibility = self.attention_state_projection(state_control)
            weather_compatibility = self.attention_weather_projection(weather_forcing)
            conditioned_attention_bias = torch.einsum(
                "bnqr,bnsr->bnqs", state_compatibility, weather_compatibility
            ) / np.sqrt(float(self.interaction_rank))
            conditioned_attention_bias = (
                conditioned_attention_bias * interaction_gate.unsqueeze(-1)
            )

        weather_attention_shift = None
        weather_view_gate = None
        near_weather_response = None
        full_weather_response = None
        if uses_weather:
            if self.weather_view_mode == "legacy":
                wants_attention = (
                    self.correction_fusion_mode
                    in {"horizon_conditioned_query", "horizon_attention_bias"}
                    and not self.training
                )
                weather_result = self.weather_query(
                    target_token,
                    weather_forcing,
                    target_times,
                    weather_query_time_hours,
                    attention_query=conditioned_query,
                    attention_bias=conditioned_attention_bias,
                    return_attention=wants_attention,
                )
                if wants_attention:
                    if not isinstance(weather_result, tuple):
                        raise RuntimeError("Weather attention weights were not returned")
                    weather_response, conditioned_weights = weather_result
                    base_result = self.weather_query(
                        target_token,
                        weather_forcing,
                        target_times,
                        weather_time_offset_hours,
                        return_attention=True,
                    )
                    if not isinstance(base_result, tuple):
                        raise RuntimeError("Base weather attention weights were not returned")
                    _, base_weights = base_result
                    weather_attention_shift = (
                        (conditioned_weights - base_weights)
                        .abs()
                        .sum(dim=-1)
                        .mean(dim=(1, 2))
                    )
                else:
                    if isinstance(weather_result, tuple):
                        raise RuntimeError("Unexpected weather attention tuple")
                    weather_response = weather_result
            else:
                wants_near = self.weather_view_mode in {"nearest", "horizon_mix"}
                wants_full = self.weather_view_mode in {"full", "horizon_mix"}
                if wants_near:
                    result = self.weather_query(
                        target_token,
                        weather_forcing,
                        target_times,
                        weather_time_offset_hours,
                        mode_override="nearest",
                    )
                    if isinstance(result, tuple):
                        raise RuntimeError("Unexpected nearest-weather attention tuple")
                    near_weather_response = result
                if wants_full:
                    result = self.weather_query(
                        target_token,
                        weather_forcing,
                        target_times,
                        weather_time_offset_hours,
                        mode_override="full_attention",
                    )
                    if isinstance(result, tuple):
                        raise RuntimeError("Unexpected full-weather attention tuple")
                    full_weather_response = result
                if self.specialist_mode == "dual_head":
                    weather_view_gate = self._weather_view_gate(target_token)
                    if self.weather_view_mode == "full":
                        assert full_weather_response is not None
                        near_weather_response = full_weather_response
                    elif self.weather_view_mode == "nearest":
                        assert near_weather_response is not None
                        full_weather_response = near_weather_response
                    assert near_weather_response is not None
                    assert full_weather_response is not None
                    weather_response = (
                        (1.0 - weather_view_gate.unsqueeze(-1))
                        * near_weather_response
                        + weather_view_gate.unsqueeze(-1) * full_weather_response
                    )
                elif self.weather_view_mode == "horizon_mix":
                    assert near_weather_response is not None
                    assert full_weather_response is not None
                    weather_view_gate = self._weather_view_gate(target_token)
                    weather_response = (
                        (1.0 - weather_view_gate.unsqueeze(-1))
                        * near_weather_response
                        + weather_view_gate.unsqueeze(-1) * full_weather_response
                    )
                elif self.weather_view_mode == "full":
                    assert full_weather_response is not None
                    weather_response = full_weather_response
                else:
                    assert near_weather_response is not None
                    weather_response = near_weather_response
        else:
            weather_response = torch.zeros_like(target_token)
        trajectory_response = None
        if trajectory_token is not None:
            if self.trajectory_response_projection is None:
                raise RuntimeError("Trajectory response projection is missing")
            trajectory_response = self.trajectory_response_projection(trajectory_token)
            if not self.strict_additive:
                weather_response = weather_response + trajectory_response
            # Keep the trajectory input live for additive and non-additive
            # state branches as well as for the black-box branch.
            state_input = torch.cat(
                (
                    history_response + trajectory_response,
                    global_response + trajectory_response,
                    target_token,
                ),
                dim=-1,
            )
        if self.matched_conditioning_backbone or uses_history:
            response_input = torch.cat(
                (history_response, global_response, weather_response, target_token),
                dim=-1,
            )
        else:
            response_input = torch.cat((weather_response, target_token), dim=-1)
        if trajectory_token is not None:
            response_input = torch.cat((response_input, trajectory_token), dim=-1)
            if self.trajectory_interaction:
                response_input = torch.cat(
                    (response_input, trajectory_token * weather_response), dim=-1
                )
        weather_main_correction = None
        state_main_correction = None
        interaction_correction = None
        interaction_update_norm = None
        near_specialist_correction = None
        full_specialist_correction = None
        if self.correction_fusion_mode == "blackbox":
            if self.specialist_mode == "dual_head":
                if (
                    self.full_correction_response is None
                    or near_weather_response is None
                    or full_weather_response is None
                    or weather_view_gate is None
                ):
                    raise RuntimeError("Dual weather specialists are incomplete")
                if self.matched_conditioning_backbone or uses_history:
                    near_input = torch.cat(
                        (
                            history_response,
                            global_response,
                            near_weather_response,
                            target_token,
                        ),
                        dim=-1,
                    )
                    full_input = torch.cat(
                        (
                            history_response,
                            global_response,
                            full_weather_response,
                            target_token,
                        ),
                        dim=-1,
                    )
                else:
                    near_input = torch.cat((near_weather_response, target_token), dim=-1)
                    full_input = torch.cat((full_weather_response, target_token), dim=-1)
                near_raw = self._decode_response(
                    self.correction_response(near_input), include_head_bias=True
                )
                full_raw = self._decode_full_response(
                    self.full_correction_response(full_input), include_head_bias=True
                )
                near_specialist_correction = self._project_correction(near_raw)
                full_specialist_correction = self._project_correction(full_raw)
                correction = (
                    (1.0 - weather_view_gate) * near_specialist_correction
                    + weather_view_gate * full_specialist_correction
                )
                raw_correction = None
            else:
                response = self.correction_response(response_input)
                raw_correction = self._decode_response(response, include_head_bias=True)
        else:
            weather_input = torch.cat((weather_response, target_token), dim=-1)
            weather_main = self.weather_main_response(weather_input)
            state_main = self.state_main_response(state_input)
            if self.correction_fusion_mode == "asymmetric_modulation":
                # One correction, not two identifiable output components:
                # recent state scales the weather-centred representation before
                # the shared nonlinear trajectory decoder.
                modulated_weather = weather_main * (1.0 + torch.tanh(state_main))
                raw_correction = self._decode_response(
                    modulated_weather, include_head_bias=True
                )
            elif self.correction_fusion_mode == "horizon_film":
                if self.film_state_down is None or self.film_state_output is None:
                    raise RuntimeError("FiLM modules are missing")
                assert interaction_gate is not None
                gamma, beta = self.film_state_output(
                    self.film_state_down(state_control)
                ).chunk(2, dim=-1)
                modulation = torch.tanh(gamma) * weather_main + torch.tanh(beta)
                gated_modulation = interaction_gate.unsqueeze(-1) * modulation
                interaction_update_norm = gated_modulation.abs().mean(dim=(1, 3))
                raw_correction = self._decode_response(
                    weather_main + gated_modulation, include_head_bias=True
                )
            elif self.correction_fusion_mode in {
                "horizon_conditioned_query",
                "horizon_attention_bias",
            }:
                raw_correction = self._decode_response(
                    weather_main, include_head_bias=True
                )
            elif self.correction_fusion_mode == "horizon_conditional_lora":
                if (
                    self.lora_state_projection is None
                    or self.lora_down is None
                    or self.lora_up is None
                ):
                    raise RuntimeError("Conditional-LoRA modules are missing")
                assert interaction_gate is not None
                state_scale = torch.tanh(self.lora_state_projection(state_control))
                lora_update = self.lora_up(
                    state_scale * self.lora_down(weather_main)
                )
                gated_update = interaction_gate.unsqueeze(-1) * lora_update
                interaction_update_norm = gated_update.abs().mean(dim=(1, 3))
                raw_correction = self._decode_response(
                    weather_main + gated_update, include_head_bias=True
                )
            else:
                weather_main_correction = self._decode_response(
                    weather_main, include_head_bias=False
                ) + self.correction_head.bias.reshape(1, 1, 1)
            if self.correction_fusion_mode == "additive":
                state_main_correction = self._decode_response(
                    state_main, include_head_bias=False
                )
                raw_correction = weather_main_correction + state_main_correction
            elif self.correction_fusion_mode in {
                "multiplicative",
                "horizon_gated_multiplicative",
            }:
                if self.correction_fusion_mode == "horizon_gated_multiplicative":
                    if self.horizon_interaction_state_projection is None:
                        raise RuntimeError("Horizon interaction projection is missing")
                    interaction_state = self.horizon_interaction_state_projection(
                        state_control
                    )
                else:
                    interaction_state = self.interaction_state_projection(state_input)
                interaction = self.interaction_response(
                    interaction_state * self.interaction_weather_projection(weather_input)
                )
                interaction_update_norm = interaction.abs().mean(dim=(1, 3))
                interaction_correction = self._decode_response(
                    interaction, include_head_bias=False
                )
                if self.correction_fusion_mode == "horizon_gated_multiplicative":
                    assert interaction_gate is not None
                    interaction_correction = (
                        interaction_correction * interaction_gate
                    )
                    interaction_update_norm = (
                        interaction_update_norm
                        * interaction_gate.reshape(1, self.forecast_steps)
                    )
                raw_correction = weather_main_correction + interaction_correction
        if raw_correction is not None:
            correction = self._project_correction(raw_correction)
        weather_main_correction = (
            self._project_correction(weather_main_correction)
            if weather_main_correction is not None
            else None
        )
        state_main_correction = (
            self._project_correction(state_main_correction)
            if state_main_correction is not None
            else None
        )
        interaction_correction = (
            self._project_correction(interaction_correction)
            if interaction_correction is not None
            else None
        )
        station_factor = baseline_station + correction
        if self.output_clamp_min is not None:
            station_factor = station_factor.clamp_min(self.output_clamp_min)
        cluster_factor = (
            station_factor * self.capacity_mw.reshape(1, -1, 1)
        ).sum(dim=1) / self.total_capacity_mw
        near_specialist_cluster = None
        full_specialist_cluster = None
        if near_specialist_correction is not None and full_specialist_correction is not None:
            near_station = (baseline_station + near_specialist_correction).clamp_min(0.0)
            full_station = (baseline_station + full_specialist_correction).clamp_min(0.0)
            near_specialist_cluster = (
                near_station * self.capacity_mw.reshape(1, -1, 1)
            ).sum(dim=1) / self.total_capacity_mw
            full_specialist_cluster = (
                full_station * self.capacity_mw.reshape(1, -1, 1)
            ).sum(dim=1) / self.total_capacity_mw
        return {
            "station_factor": station_factor,
            "cluster_factor": cluster_factor,
            "baseline_station_factor": baseline_station,
            "baseline_cluster_factor": baseline["cluster_factor"],
            "anchor_baseline_station_factor": (
                anchor_baseline["station_factor"]
                if anchor_baseline is not None
                else None
            ),
            "anchor_baseline_cluster_factor": (
                anchor_baseline["cluster_factor"]
                if anchor_baseline is not None
                else None
            ),
            "weather_correction_factor": correction,
            "trajectory_conditioning_token": trajectory_token,
            "trajectory_response_token": trajectory_response,
            "near_specialist_correction_factor": near_specialist_correction,
            "full_specialist_correction_factor": full_specialist_correction,
            "near_specialist_cluster_factor": near_specialist_cluster,
            "full_specialist_cluster_factor": full_specialist_cluster,
            "weather_view_gate": (
                weather_view_gate.reshape(1, self.forecast_steps).expand(batch, -1)
                if weather_view_gate is not None
                else None
            ),
            "baseline_anchor_abs_drift_factor": (
                (baseline["cluster_factor"] - anchor_baseline["cluster_factor"]).abs()
                if anchor_baseline is not None
                else None
            ),
            "weather_main_correction_factor": weather_main_correction,
            "state_main_correction_factor": state_main_correction,
            "state_weather_interaction_correction_factor": interaction_correction,
            "horizon_interaction_gate": interaction_gate,
            "interaction_update_abs_mean": interaction_update_norm,
            "weather_attention_shift_l1": weather_attention_shift,
            "composition_statistics_state": composition_statistics_state,
            "composition_set_state": composition_set_state,
            "composition_statistics_input": composition_statistics_input,
            "hourly_station_factor_unique": unique_weather["hourly_station_factor"],
            "spatial_attention_unique": unique_weather["spatial_attention"],
            "regional_spatial_attention_unique": unique_weather[
                "regional_spatial_attention"
            ],
            "spatial_scale_attention_unique": unique_weather["spatial_scale_attention"],
        }


def build_incremental_correction_model(
    model_config: Mapping[str, Any],
    static: Mapping[str, Sequence[float]],
    *,
    history_steps: int = 24,
    forecast_steps: int = 24,
) -> IncrementalCorrectionModel:
    baseline_config = dict(model_config)
    baseline_config.update(
        architecture="forced_evolution",
        fusion_mode="history_only",
        output_level="hierarchical",
        dropout=float(model_config.get("baseline_dropout", model_config["dropout"])),
        # Spatial/forcing ablations belong to the correction branch.  The
        # pretrained history baseline contains unused weather modules in its
        # state dict, so its construction contract must stay checkpoint-
        # compatible even when the correction encoder is changed.
        station_query_mode=str(
            model_config.get("baseline_station_query_mode", "learned_cells")
        ),
        station_neighborhood_cells=int(
            model_config.get("baseline_station_neighborhood_cells", 3)
        ),
        include_wind_geometry=bool(
            model_config.get("baseline_include_wind_geometry", True)
        ),
        forcing_path_mode=str(
            model_config.get("baseline_forcing_path_mode", "difference")
        ),
        time_query_mode=str(
            model_config.get("baseline_time_query_mode", "full_attention")
        ),
    )
    baseline = build_forced_evolution_model(
        baseline_config,
        static,
        history_steps=history_steps,
        forecast_steps=forecast_steps,
    )
    projection_mode = str(model_config.get("projection_mode", "full"))
    spatial_basis: np.ndarray | None = None
    if projection_mode in {"spatial", "joint"}:
        basis_path = Path(str(model_config["spatial_basis_path"])).expanduser().resolve()
        all_components = np.load(basis_path)
        spatial_count = int(model_config.get("spatial_components", len(all_components)))
        spatial_basis = np.asarray(all_components[:spatial_count], dtype=np.float32)
    return IncrementalCorrectionModel(
        baseline,
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
        correction_conditioning=str(
            model_config.get("correction_conditioning", "history_weather")
        ),
        correction_history_mode=str(
            model_config.get("correction_history_mode", "full_history")
        ),
        cross_farm_context_mode=str(
            model_config.get("cross_farm_context_mode", "none")
        ),
        cross_farm_neighbors=int(model_config.get("cross_farm_neighbors", 4)),
        correction_fusion_mode=str(
            model_config.get("correction_fusion_mode", "blackbox")
        ),
        interaction_rank=int(model_config.get("interaction_rank", 16)),
        matched_conditioning_backbone=bool(
            model_config.get("matched_conditioning_backbone", False)
        ),
        baseline_anchor_enabled=bool(
            model_config.get("baseline_anchor_enabled", False)
        ),
        projection_mode=projection_mode,
        temporal_components=int(model_config.get("temporal_components", forecast_steps)),
        spatial_basis=spatial_basis,
        freeze_history_baseline=bool(
            model_config.get("freeze_history_baseline", True)
        ),
        station_latitude=static["station_latitude"],
        station_longitude=static["station_longitude"],
        weather_view_mode=str(model_config.get("weather_view_mode", "legacy")),
        weather_gate_mode=str(model_config.get("weather_gate_mode", "fixed")),
        specialist_mode=str(model_config.get("specialist_mode", "shared_head")),
        bracket_time_feature_mode=str(
            model_config.get("bracket_time_feature_mode", "relative_distances")
        ),
        bracket_mapper_activation=str(
            model_config.get("bracket_mapper_activation", "gelu")
        ),
        trajectory_conditioning=str(
            model_config.get("trajectory_conditioning", "none")
        ),
        trajectory_interaction=bool(
            model_config.get("trajectory_interaction", False)
        ),
        trajectory_input_mode=str(
            model_config.get("trajectory_input_mode", "matched")
        ),
        strict_additive=bool(model_config.get("strict_additive", False)),
        output_clamp_min=(
            None
            if model_config.get("output_clamp_min", 0.0) is None
            else float(model_config.get("output_clamp_min", 0.0))
        ),
    )
