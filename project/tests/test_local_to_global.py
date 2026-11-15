from __future__ import annotations

import numpy as np
import torch

from wind_forcing.models.local_to_global import build_local_to_global_model


def static() -> dict[str, np.ndarray]:
    return {
        "station_capacity_mw": np.asarray([10.0, 20.0], dtype=np.float32),
        "station_latitude": np.asarray([1.25, 1.75], dtype=np.float32),
        "station_longitude": np.asarray([0.25, 0.75], dtype=np.float32),
        "weather_latitude": np.asarray([2.0, 1.0], dtype=np.float64),
        "weather_longitude": np.asarray([0.0, 1.0], dtype=np.float64),
        "nwp_mean": np.zeros(14, dtype=np.float32),
        "nwp_std": np.ones(14, dtype=np.float32),
    }


def config(
    state_mode: str = "local_recent", state_operator: str | None = None
) -> dict[str, object]:
    result: dict[str, object] = {
        "weather_channels": 14, "weather_hidden_channels": 8, "d_model": 8,
        "history_visible_steps": 12, "state_visible_steps": 4,
        "history_tcn_layers": 1, "trajectory_tcn_layers": 1,
        "attention_heads": 2, "dropout": 0.0, "state_mode": state_mode,
        "station_specific_mode": "none", "station_embedding_dim": 4,
        "modulation_amplitude": 0.25, "backbone_lr_multiplier": 0.1,
    }
    if state_operator is not None:
        result["state_interaction_operator"] = state_operator
    return result


def inputs() -> dict[str, torch.Tensor]:
    return {
        "weather": torch.randn(1, 8, 14, 2, 2),
        "weather_batch_index": torch.tensor([0]),
        "weather_time_offset_hours": torch.arange(8, dtype=torch.float32).reshape(1, -1),
        "station_history": torch.rand(1, 2, 24),
        "cluster_history": torch.rand(1, 24),
        "origin_phase": torch.tensor([0]),
    }


def test_raw_and_precomputed_bilinear_are_identical() -> None:
    model = build_local_to_global_model(config(), static(), history_steps=24, forecast_steps=24)
    model.eval()
    values = inputs()
    local = model._local_weather(values["weather"])
    with torch.no_grad():
        grid_output = model(**values)["cluster_factor"]
        values["weather"] = local
        local_output = model(**values)["cluster_factor"]
    torch.testing.assert_close(grid_output, local_output)


def test_capacity_sum_and_bounds() -> None:
    model = build_local_to_global_model(config(), static(), history_steps=24, forecast_steps=24)
    model.eval()
    with torch.no_grad():
        output = model(**inputs())
    station = output["station_factor"]
    assert station is not None
    assert bool(((station >= 0) & (station <= 1)).all())
    expected = (station * model.capacity_mw.reshape(1, -1, 1)).sum(1) / model.total_capacity_mw
    torch.testing.assert_close(output["cluster_factor"], expected)


def test_identity_state_initialization_matches_no_state() -> None:
    local = build_local_to_global_model(config("local_recent"), static(), history_steps=24, forecast_steps=24)
    none = build_local_to_global_model(config("none"), static(), history_steps=24, forecast_steps=24)
    none.load_state_dict(local.state_dict())
    local.eval(); none.eval()
    values = inputs()
    with torch.no_grad():
        torch.testing.assert_close(local(**values)["cluster_factor"], none(**values)["cluster_factor"])


def test_zero_weather_latent_has_no_state_only_correction() -> None:
    model = build_local_to_global_model(config("local_recent"), static(), history_steps=24, forecast_steps=24)
    for name, parameter in model.named_parameters():
        if name.startswith(("weather_encoder.", "forcing_path.", "weather_query.")):
            parameter.data.zero_()
    # Make the state modulator non-trivial; it still only multiplies zero.
    model.state_scale[-1].weight.data.fill_(0.1)
    model.eval()
    first = inputs()
    second = {key: value.clone() for key, value in first.items()}
    second["station_history"] = torch.rand_like(second["station_history"])
    second["cluster_history"] = torch.rand_like(second["cluster_history"])
    with torch.no_grad():
        a = model(**first)["local_weather_correction_logit"]
        b = model(**second)["local_weather_correction_logit"]
    torch.testing.assert_close(a, b)


def test_zero_invariant_interaction_is_exact_for_different_states() -> None:
    model = build_local_to_global_model(
        config("local_recent", "zero_invariant_mlp"),
        static(), history_steps=24, forecast_steps=24,
    )
    assert model.state_interaction is not None
    model.state_interaction[-1].weight.data.normal_()
    model.state_interaction[-1].bias.data.normal_()
    weather = torch.zeros(3, 2, 24, 8)
    state = torch.randn(3, 2, 8)
    changed, update = model._apply_state_interaction(weather, state)
    torch.testing.assert_close(update, torch.zeros_like(update), atol=0.0, rtol=0.0)
    torch.testing.assert_close(changed, weather, atol=0.0, rtol=0.0)


def test_concat_interaction_permits_a_state_only_update() -> None:
    model = build_local_to_global_model(
        config("local_recent", "concat_mlp"),
        static(), history_steps=24, forecast_steps=24,
    )
    assert model.state_interaction is not None
    for parameter in model.state_interaction.parameters():
        parameter.data.fill_(0.1)
    weather = torch.zeros(2, 2, 24, 8)
    state = torch.stack((torch.zeros(2, 8), torch.ones(2, 8)))
    _, update = model._apply_state_interaction(weather, state)
    assert not torch.allclose(update[0], update[1])


def test_new_operator_identity_initialization_accepts_only_new_keys() -> None:
    base = build_local_to_global_model(
        config("none", "none"), static(), history_steps=24, forecast_steps=24
    )
    candidate = build_local_to_global_model(
        config("local_recent", "zero_invariant_mlp"),
        static(), history_steps=24, forecast_steps=24,
    )
    result = candidate.load_state_dict(base.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert result.missing_keys
    assert all(key.startswith("state_interaction.") for key in result.missing_keys)
    base.eval(); candidate.eval()
    values = inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            candidate(**values)["cluster_factor"],
            base(**values)["cluster_factor"],
        )


def test_operator_and_state_mode_must_be_consistent() -> None:
    with np.testing.assert_raises(ValueError):
        build_local_to_global_model(
            config("none", "concat_mlp"), static(), history_steps=24, forecast_steps=24
        )
    with np.testing.assert_raises(ValueError):
        build_local_to_global_model(
            config("local_recent", "none"), static(), history_steps=24, forecast_steps=24
        )
