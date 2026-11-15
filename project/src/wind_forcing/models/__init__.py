from wind_forcing.models.forced_evolution import (
    ForcedEvolutionModel,
    SpatialPretrainModel,
    StationForcingEncoder,
    build_forced_evolution_model,
    build_spatial_pretrain_model,
)
from wind_forcing.models.baselines import (
    build_history_baseline,
    build_weather_baseline,
)
from wind_forcing.models.controlled_transition import (
    ControlledTransitionModel,
    build_controlled_transition_model,
)
from wind_forcing.models.incremental_correction import (
    IncrementalCorrectionModel,
    build_incremental_correction_model,
)
from wind_forcing.models.aggregate_cluster import build_aggregate_model
from wind_forcing.models.local_to_global import (
    LocalToGlobalModel,
    build_local_to_global_model,
)
from wind_forcing.models.local_state_identifiability import (
    LocalStateIdentifiabilityModel,
    build_local_state_identifiability_model,
)


def forecast_model_uses_weather(model_config):
    architecture = str(model_config.get("architecture", "forced_evolution"))
    if architecture.startswith("history_"):
        return False
    if architecture == "forced_evolution":
        return str(model_config.get("fusion_mode", "state_conditioned")) != "history_only"
    return True


def build_forecast_model(
    model_config,
    static,
    *,
    history_steps=24,
    forecast_steps=24,
):
    architecture = str(model_config.get("architecture", "forced_evolution"))
    if architecture == "forced_evolution":
        return build_forced_evolution_model(
            model_config,
            static,
            history_steps=history_steps,
            forecast_steps=forecast_steps,
        )
    if architecture == "controlled_transition":
        return build_controlled_transition_model(
            model_config,
            static,
            history_steps=history_steps,
            forecast_steps=forecast_steps,
        )
    if architecture == "incremental_correction":
        return build_incremental_correction_model(
            model_config,
            static,
            history_steps=history_steps,
            forecast_steps=forecast_steps,
        )
    if architecture == "local_to_global":
        return build_local_to_global_model(
            model_config,
            static,
            history_steps=history_steps,
            forecast_steps=forecast_steps,
        )
    if architecture == "local_state_identifiability":
        return build_local_state_identifiability_model(
            model_config,
            static,
            history_steps=history_steps,
            forecast_steps=forecast_steps,
        )
    if architecture.startswith("history_"):
        return build_history_baseline(
            model_config, static, history_steps, forecast_steps
        )
    if architecture.startswith("weather_"):
        return build_weather_baseline(model_config, static, forecast_steps)
    raise ValueError(f"Unsupported model.architecture: {architecture}")

__all__ = [
    "ForcedEvolutionModel",
    "ControlledTransitionModel",
    "IncrementalCorrectionModel",
    "LocalToGlobalModel",
    "LocalStateIdentifiabilityModel",
    "SpatialPretrainModel",
    "StationForcingEncoder",
    "build_forced_evolution_model",
    "build_forecast_model",
    "build_spatial_pretrain_model",
    "forecast_model_uses_weather",
    "build_aggregate_model",
]
