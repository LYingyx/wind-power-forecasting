from __future__ import annotations

import numpy as np
import torch

from wind_forcing.models.local_state_identifiability import (
    build_local_state_identifiability_model,
)


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
    source: str = "local_power",
    context: str = "static",
    mode: str = "strict",
    spatial: str = "precomputed_bilinear",
) -> dict[str, object]:
    return {
        "weather_channels": 14, "weather_hidden_channels": 8, "d_model": 8,
        "history_visible_steps": 12, "state_visible_steps": 4,
        "history_tcn_layers": 1, "trajectory_tcn_layers": 1,
        "attention_heads": 2, "dropout": 0.0,
        "response_state_source": source, "station_context_mode": context,
        "response_interaction_mode": mode, "station_embedding_dim": 4,
        "modulation_amplitude": 0.25, "backbone_lr_multiplier": 0.25,
        "fine_tune_policy": "all",
        "spatial_weather_mode": spatial, "station_neighborhood_cells": 3,
    }


def inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    station = torch.rand(batch, 2, 24)
    cluster = (station * torch.tensor([10.0, 20.0])[None, :, None]).sum(1) / 30.0
    return {
        "weather": torch.randn(batch, 8, 2, 14),
        "weather_batch_index": torch.arange(batch),
        "weather_time_offset_hours": torch.arange(8, dtype=torch.float32)[None].expand(batch, -1),
        "station_history": station,
        "cluster_history": cluster,
        "state_station_history": station.clone(),
        "state_cluster_history": cluster.clone(),
        "origin_phase": torch.zeros(batch, dtype=torch.long),
        "past_weather": torch.randn(batch, 6, 2, 14),
        "past_weather_batch_index": torch.arange(batch),
        "past_weather_time_offset_hours": torch.arange(-5, 1, dtype=torch.float32)[None].expand(batch, -1),
        "past_station_power": torch.rand(batch, 2, 6),
    }


def build(source: str = "local_power", context: str = "static", mode: str = "strict"):
    return build_local_state_identifiability_model(
        config(source, context, mode), static(), history_steps=24, forecast_steps=24
    )


def grid_static() -> dict[str, np.ndarray]:
    values = static()
    values["station_latitude"] = np.asarray([1.75, 2.25], dtype=np.float32)
    values["station_longitude"] = np.asarray([2.25, 2.75], dtype=np.float32)
    values["weather_latitude"] = np.asarray(
        [4.0, 3.0, 2.0, 1.0, 0.0, -1.0], dtype=np.float64
    )
    values["weather_longitude"] = np.asarray(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64
    )
    return values


def test_strict_interaction_is_exactly_zero_invariant() -> None:
    model = build()
    for parameter in model.response_interaction.parameters():
        parameter.data.normal_()
    weather = torch.zeros(3, 2, 24, 8)
    condition = torch.randn(3, 2, 8)
    changed, update = model._conditioned_weather(weather, condition)
    torch.testing.assert_close(changed, torch.zeros_like(changed), atol=0.0, rtol=0.0)
    torch.testing.assert_close(update, torch.zeros_like(update), atol=0.0, rtol=0.0)


def test_state_encoder_is_in_full_learning_rate_group() -> None:
    model = build()
    groups = model.optimizer_parameter_groups(1e-3, 1e-4)
    state_ids = {id(parameter) for parameter in model.state_encoder.parameters()}
    assert state_ids.issubset({id(parameter) for parameter in groups[0]["params"]})
    assert groups[0]["lr"] == 1e-3
    assert groups[1]["lr"] == 2.5e-4


def test_baseline_uses_actual_history_when_response_state_is_donor() -> None:
    model = build().eval()
    values = inputs()
    donor = {key: value.clone() for key, value in values.items()}
    donor["state_station_history"] = donor["state_station_history"].flip(0)
    donor["state_cluster_history"] = donor["state_cluster_history"].flip(0)
    with torch.no_grad():
        first = model(**values)
        second = model(**donor)
    torch.testing.assert_close(
        first["baseline_station_factor"], second["baseline_station_factor"]
    )


def test_io_pair_identification_has_support_query_shape() -> None:
    model = build("io_pair_id", "static", "strict").eval()
    with torch.no_grad():
        output = model(**inputs())
    assert output["response_identification_station_factor"].shape == (2, 2, 3)
    assert output["station_factor"].shape == (2, 2, 24)


def test_far_station_permutation_has_no_fixed_points() -> None:
    model = build()
    assert bool(torch.all(model.far_station_index != torch.arange(2)))


def test_faithful_concat_is_not_zero_weather_invariant() -> None:
    model = build("local_power", "none", "faithful_concat")
    weather = torch.zeros(2, 2, 24, 8)
    state = torch.stack((torch.zeros(2, 8), torch.ones(2, 8)))
    changed, _ = model._conditioned_weather(weather, state)
    assert not torch.allclose(changed[0], changed[1])


def test_additive_control_has_no_weather_state_interaction() -> None:
    model = build("local_power", "none", "additive_main_effect")
    weather_a = torch.randn(2, 2, 24, 8)
    weather_b = torch.randn(2, 2, 24, 8)
    state_a = torch.randn(2, 2, 8)
    state_b = torch.randn(2, 2, 8)
    aa, _ = model._conditioned_weather(weather_a, state_a)
    ab, _ = model._conditioned_weather(weather_a, state_b)
    ba, _ = model._conditioned_weather(weather_b, state_a)
    bb, _ = model._conditioned_weather(weather_b, state_b)
    # Additive separability implies a zero difference-in-differences.
    torch.testing.assert_close(aa - ab - ba + bb, torch.zeros_like(aa), atol=1e-6, rtol=1e-5)


def test_preserve_policy_freezes_only_state_response_modules() -> None:
    values = config("local_power", "none", "faithful_concat")
    values["fine_tune_policy"] = "freeze_state_response"
    model = build_local_state_identifiability_model(
        values, static(), history_steps=24, forecast_steps=24
    )
    assert not any(parameter.requires_grad for parameter in model.state_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.response_interaction.parameters())
    assert any(parameter.requires_grad for parameter in model.weather_encoder.parameters())
    model.train()
    assert not model.state_encoder.training
    assert not model.response_interaction.training


def test_raw_bilinear_and_surrounding_weather_share_output_contract() -> None:
    values = inputs()
    values["weather"] = torch.randn(2, 8, 14, 6, 6)
    for spatial in (
        "raw_bilinear",
        "fixed_gaussian_4x4",
        "learned_4x4_wind",
    ):
        model = build_local_state_identifiability_model(
            config("local_power", "none", "faithful_concat", spatial),
            grid_static(),
            history_steps=24,
            forecast_steps=24,
        ).eval()
        with torch.no_grad():
            output = model(**values)
        assert output["station_factor"].shape == (2, 2, 24)
        assert output["cluster_factor"].shape == (2, 24)
        attention = output["spatial_attention_unique"]
        assert attention is not None
        expected_tokens = 4 if spatial == "raw_bilinear" else 16
        assert attention.shape == (2, 8, 2, expected_tokens)


def test_additive_control_remains_separable_with_learned_4x4_weather() -> None:
    model = build_local_state_identifiability_model(
        config("local_power", "none", "additive_main_effect", "learned_4x4_wind"),
        grid_static(),
        history_steps=24,
        forecast_steps=24,
    )
    weather_a = torch.randn(2, 2, 24, 8)
    weather_b = torch.randn(2, 2, 24, 8)
    state_a = torch.randn(2, 2, 8)
    state_b = torch.randn(2, 2, 8)
    aa, _ = model._conditioned_weather(weather_a, state_a)
    ab, _ = model._conditioned_weather(weather_a, state_b)
    ba, _ = model._conditioned_weather(weather_b, state_a)
    bb, _ = model._conditioned_weather(weather_b, state_b)
    torch.testing.assert_close(
        aa - ab - ba + bb, torch.zeros_like(aa), atol=1e-6, rtol=1e-5
    )


def test_output_additive_is_separable_after_complete_decoder() -> None:
    model = build_local_state_identifiability_model(
        config("local_power", "none", "output_additive", "learned_4x4_wind"),
        grid_static(),
        history_steps=24,
        forecast_steps=24,
    ).eval()
    base = inputs()
    weather_a = torch.randn(2, 8, 14, 6, 6)
    weather_b = torch.randn(2, 8, 14, 6, 6)
    state_a = torch.rand(2, 2, 24)
    state_b = torch.rand(2, 2, 24)

    def correction(weather: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        values = {key: value.clone() for key, value in base.items()}
        values["weather"] = weather
        values["state_station_history"] = state
        with torch.no_grad():
            output = model(**values)
        return output["local_weather_correction_logit"]

    aa = correction(weather_a, state_a)
    ab = correction(weather_a, state_b)
    ba = correction(weather_b, state_a)
    bb = correction(weather_b, state_b)
    torch.testing.assert_close(
        aa - ab - ba + bb, torch.zeros_like(aa), atol=2e-6, rtol=1e-5
    )


def test_independent_output_additive_matches_interaction_parameter_count() -> None:
    interaction = build("local_power", "none", "faithful_concat")
    additive = build("local_power", "none", "independent_output_additive")
    interaction_count = sum(
        parameter.numel() for parameter in interaction.response_interaction.parameters()
    )
    additive_count = sum(
        parameter.numel() for parameter in additive.response_interaction.parameters()
    )
    assert interaction_count == additive_count == 408
    assert sum(parameter.numel() for parameter in interaction.parameters()) == sum(
        parameter.numel() for parameter in additive.parameters()
    )


def test_independent_output_additive_is_separable_after_complete_decoder() -> None:
    model = build_local_state_identifiability_model(
        config(
            "local_power",
            "none",
            "independent_output_additive",
            "fixed_gaussian_4x4",
        ),
        grid_static(),
        history_steps=24,
        forecast_steps=24,
    ).eval()
    base = inputs()
    weather_a = torch.randn(2, 8, 14, 6, 6)
    weather_b = torch.randn(2, 8, 14, 6, 6)
    state_a = torch.rand(2, 2, 24)
    state_b = torch.rand(2, 2, 24)

    def correction(weather: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        values = {key: value.clone() for key, value in base.items()}
        values["weather"] = weather
        values["state_station_history"] = state
        with torch.no_grad():
            output = model(**values)
        return output["local_weather_correction_logit"]

    aa = correction(weather_a, state_a)
    ab = correction(weather_a, state_b)
    ba = correction(weather_b, state_a)
    bb = correction(weather_b, state_b)
    torch.testing.assert_close(
        aa - ab - ba + bb, torch.zeros_like(aa), atol=2e-6, rtol=1e-5
    )


def test_history_only_is_exactly_weather_invariant() -> None:
    values = config("none", "none", "none", "raw_bilinear")
    values["source_role_mode"] = "history_only"
    model = build_local_state_identifiability_model(
        values, grid_static(), history_steps=24, forecast_steps=24
    ).eval()
    first = inputs()
    first["weather"] = torch.randn(2, 8, 14, 6, 6)
    second = {key: value.clone() for key, value in first.items()}
    second["weather"] = torch.randn_like(first["weather"]) * 10.0
    with torch.no_grad():
        first_output = model(**first)["station_factor"]
        second_output = model(**second)["station_factor"]
    torch.testing.assert_close(first_output, second_output, atol=0.0, rtol=0.0)


def test_anchor_weather_is_exactly_pre_origin_history_invariant() -> None:
    values = config("none", "none", "none", "raw_bilinear")
    values["source_role_mode"] = "anchor_weather"
    model = build_local_state_identifiability_model(
        values, grid_static(), history_steps=24, forecast_steps=24
    ).eval()
    first = inputs()
    first["weather"] = torch.randn(2, 8, 14, 6, 6)
    second = {key: value.clone() for key, value in first.items()}
    second["station_history"][:, :, :-1] = torch.rand_like(
        second["station_history"][:, :, :-1]
    )
    second["cluster_history"][:, :-1] = torch.rand_like(
        second["cluster_history"][:, :-1]
    )
    with torch.no_grad():
        first_output = model(**first)["station_factor"]
        second_output = model(**second)["station_factor"]
    torch.testing.assert_close(first_output, second_output, atol=0.0, rtol=0.0)


def test_default_full_source_role_preserves_legacy_state_dict() -> None:
    legacy = build("local_power", "none", "faithful_concat")
    values = config("local_power", "none", "faithful_concat")
    values["source_role_mode"] = "full"
    explicit = build_local_state_identifiability_model(
        values, static(), history_steps=24, forecast_steps=24
    )
    explicit.load_state_dict(legacy.state_dict(), strict=True)
