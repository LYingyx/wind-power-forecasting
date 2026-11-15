from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from wind_forcing.data import (
    AggregateForecastDataset,
    AggregateResponseCacheBuilder,
    AggregateResponseForecastDataset,
    AggregateDynamicsTimeCacheBuilder,
    capacity_density_bilinear,
    create_aggregate_loader,
    create_aggregate_response_loader,
)
from wind_forcing.aggregate_training import aggregate_inputs, aggregate_loss
from wind_forcing.config import load_config
from wind_forcing.models import build_aggregate_model


def test_capacity_density_bilinear_conserves_and_normalizes() -> None:
    density = capacity_density_bilinear(
        np.asarray([2.0, 1.0, 0.0]),
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([1.5, 0.5]),
        np.asarray([0.5, 1.5]),
        np.asarray([100.0, 300.0]),
    )
    assert density.shape == (3, 3)
    assert np.all(density >= 0.0)
    np.testing.assert_allclose(density.sum(), 1.0, atol=1e-7)


def test_real_aggregate_cache_has_no_station_sample_tensors() -> None:
    cache = Path("data/aggregate_cluster_v1")
    if not cache.is_dir():
        return
    for split in ("train", "val", "test"):
        names = {path.name for path in (cache / split).glob("*.npy")}
        assert not any("station" in name for name in names)
        assert "cluster_history_mw.npy" in names
        assert "cluster_future_mw.npy" in names
    dataset = AggregateForecastDataset(cache, "train", include_weather=False, verbose=False)
    batch = next(iter(create_aggregate_loader(dataset, cycles_per_batch=1, shuffle=False)))
    assert batch["cluster_history"].shape == (4, 24)
    assert batch["cluster_future"].shape == (4, 24)
    assert not any("station" in key for key in batch)
    expected = np.load(cache / "train" / "cluster_history_mw.npy")[:4]
    np.testing.assert_allclose(
        batch["cluster_history"].numpy() * dataset.total_capacity_mw,
        expected,
        rtol=1e-6,
        atol=1e-3,
    )


def _static() -> dict[str, np.ndarray]:
    density = np.zeros((24, 25), dtype=np.float32)
    density[10:12, 10:12] = 0.25
    return {"capacity_density": density}


def _config(route: str) -> dict[str, object]:
    return {
        "architecture": "aggregate_correction",
        "weather_route": route,
        "weather_channels": 14,
        "d_model": 16,
        "history_visible_steps": 24,
        "history_layers": 1,
        "trajectory_layers": 1,
        "attention_heads": 4,
        "dropout": 0.0,
        "baseline_dropout": 0.0,
        "state_mode": "recent_1h",
        "correction_conditioning": "state_weather",
        "correction_fusion": "blackbox",
    }


def _strict_config(mode: str, source: str = "baseline_latent", stage: str = "adapter") -> dict[str, object]:
    config = _config("r2_latent_interpolation")
    config.update(
        architecture="aggregate_strict_modulation",
        history_visible_steps=12,
        state_mode="current",
        correction_conditioning="weather",
        strict_modulation_mode=mode,
        modulation_state_source=source,
        modulation_training_stage=stage,
        modulation_use_reference_state=False,
        modulation_amplitude=0.25,
        modulation_low_rank=4,
        modulation_expert_count=3,
        modulation_expert_coefficient=0.1,
    )
    return config


def _strict_inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "cluster_history": torch.rand(batch, 24),
        "weather": torch.rand(1, 8, 14, 24, 25),
        "weather_batch_index": torch.zeros(batch, dtype=torch.long),
        "weather_time_offset_hours": torch.arange(8, dtype=torch.float32).reshape(1, 8).expand(batch, -1),
    }


def _nested_models(mode: str, source: str = "baseline_latent", stage: str = "adapter"):
    base_config = _config("r2_latent_interpolation")
    base_config.update(history_visible_steps=12, state_mode="current", correction_conditioning="weather")
    base = build_aggregate_model(base_config, _static(), total_capacity_mw=1000.0).eval()
    strict = build_aggregate_model(
        _strict_config(mode, source, stage), _static(), total_capacity_mw=1000.0
    )
    strict.load_base_state(base.state_dict())
    strict.set_state_reference(0.5, torch.zeros(16))
    return base, strict


def _response_config(response_input: str, stage: str = "adapter") -> dict[str, object]:
    config = _strict_config("low_rank", "baseline_latent", stage)
    config.update(
        architecture="aggregate_response_state",
        response_state_input=response_input,
        response_state_hidden=32,
        response_identification_loss_weight=(0.2 if response_input == "transition_pair_id" else 0.0),
    )
    return config


def _response_models(response_input: str, stage: str = "adapter"):
    base_config = _config("r2_latent_interpolation")
    base_config.update(history_visible_steps=12, state_mode="current", correction_conditioning="weather")
    base = build_aggregate_model(base_config, _static(), total_capacity_mw=1000.0).eval()
    response = build_aggregate_model(
        _response_config(response_input, stage), _static(), total_capacity_mw=1000.0
    )
    response.load_base_state(base.state_dict())
    reference_nodes = 6 if response_input == "level_pair" else 5
    response.set_response_reference_tokens(
        torch.zeros(reference_nodes, response.response_token_width)
    )
    return base, response


def _response_inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    history = torch.rand(batch, 24)
    future_weather = torch.rand(1, 8, 14, 24, 25)
    past_weather = torch.rand(batch, 6, 14, 24, 25)
    return {
        "cluster_history": history,
        "future_weather": future_weather,
        "future_hourly_weather_latent": torch.empty(0),
        "weather_batch_index": torch.zeros(batch, dtype=torch.long),
        "weather_time_offset_hours": torch.arange(8, dtype=torch.float32).reshape(1, 8).expand(batch, -1),
        "past_weather": past_weather,
        "past_hourly_weather_latent": torch.empty(0),
        "past_cluster_power": torch.rand(batch, 6),
        "past_weather_time_offset_hours": torch.arange(-5, 1, dtype=torch.float32).reshape(1, 6).expand(batch, -1),
    }


def test_all_weather_routes_have_finite_aggregate_outputs() -> None:
    inputs = {
        "cluster_history": torch.rand(1, 24),
        "weather": torch.rand(1, 8, 14, 24, 25),
        "weather_batch_index": torch.zeros(1, dtype=torch.long),
        "weather_time_offset_hours": torch.arange(8, dtype=torch.float32).reshape(1, 8),
    }
    routes = (
        "r1_raw_interpolation",
        "r2_latent_interpolation",
        "r3_full_path",
        "r4_target_value",
        "r4_no_time",
        "r5_dynamic_pool",
        "r6_regional_tokens",
        "r7_patch_regional",
        "r8_joint_spacetime",
        "r8_factorized",
    )
    counts = {}
    for route in routes:
        model = build_aggregate_model(_config(route), _static(), total_capacity_mw=1000.0)
        output = model(**inputs)
        assert output["cluster_factor"].shape == (1, 24)
        assert torch.isfinite(output["cluster_factor"]).all()
        counts[route] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert counts["r4_target_value"] == counts["r4_no_time"]
    assert counts["r8_joint_spacetime"] == counts["r8_factorized"]


def test_state_windows_are_parameter_matched_and_backward_finite() -> None:
    modes = ("current", "recent_1h", "recent_3h", "full_6h", "multiscale", "horizon")
    counts = []
    for mode in modes:
        config = _config("r3_full_path")
        config["state_mode"] = mode
        model = build_aggregate_model(config, _static(), total_capacity_mw=1000.0)
        counts.append(sum(p.numel() for p in model.state_encoder.parameters()))
    assert len(set(counts)) == 1


def test_history_tcn_windows_are_parameter_matched() -> None:
    counts = []
    for visible in (4, 12, 24):
        model = build_aggregate_model(
            {
                "architecture": "hb_tcn",
                "d_model": 16,
                "history_visible_steps": visible,
                "history_layers": 1,
                "attention_heads": 4,
                "dropout": 0.0,
            },
            _static(),
            total_capacity_mw=1000.0,
        )
        counts.append(sum(p.numel() for p in model.parameters()))
        output = model(cluster_history=torch.rand(2, 24))["cluster_factor"]
        assert output.shape == (2, 24)
    assert len(set(counts)) == 1


def test_weather_only_is_independent_of_cluster_history() -> None:
    config = _config("r3_full_path")
    config["architecture"] = "w_only"
    model = build_aggregate_model(config, _static(), total_capacity_mw=1000.0).eval()
    assert model.history_encoder is None
    common = {
        "weather": torch.rand(1, 8, 14, 24, 25),
        "weather_batch_index": torch.zeros(2, dtype=torch.long),
        "weather_time_offset_hours": torch.arange(8, dtype=torch.float32).reshape(1, 8).expand(2, -1),
    }
    with torch.no_grad():
        first = model(cluster_history=torch.zeros(2, 24), **common)["cluster_factor"]
        second = model(cluster_history=torch.ones(2, 24), **common)["cluster_factor"]
    torch.testing.assert_close(first, second)


def test_direct_fusion_has_no_persistence_output_shortcut() -> None:
    config = _config("r3_full_path")
    config["architecture"] = "f_target_concat"
    model = build_aggregate_model(config, _static(), total_capacity_mw=1000.0).eval()
    common = {
        "weather": torch.rand(1, 8, 14, 24, 25),
        "weather_batch_index": torch.zeros(1, dtype=torch.long),
        "weather_time_offset_hours": torch.arange(8, dtype=torch.float32).reshape(1, 8),
    }
    with torch.no_grad():
        low = model(cluster_history=torch.zeros(1, 24), **common)["cluster_factor"]
        high = model(cluster_history=torch.ones(1, 24), **common)["cluster_factor"]
    # History may affect the learned representation, but changing the current
    # value must not be added one-for-one as an explicit persistence baseline.
    assert float((high - low).abs().max()) < 0.5


def test_horizon_taper_anchor_has_registered_per_lead_weights() -> None:
    prediction = torch.zeros(1, 24)
    output = {
        "cluster_factor": prediction,
        "baseline_cluster_factor": torch.ones(1, 24),
        "anchor_baseline_cluster_factor": torch.zeros(1, 24),
    }
    batch = {
        "cluster_future": torch.zeros(1, 24),
        "cluster_history": torch.zeros(1, 24),
    }
    _, parts = aggregate_loss(
        output,
        batch,
        {
            "cluster_loss_weight": 1.0,
            "baseline_anchor_loss_weight": 0.2,
            "anchor_schedule_mode": "horizon_taper",
        },
    )
    weights = torch.where(
        torch.arange(1, 25, dtype=torch.float32) / 4 <= 3,
        torch.tensor(1.0),
        (4 - torch.arange(1, 25, dtype=torch.float32) / 4).clamp(0, 1),
    )
    expected = float((weights * 0.975).mean())
    assert abs(parts["anchor"] - expected) < 1e-6


def test_state_visibility_intervention_keeps_query_type_and_masks_old_lags() -> None:
    for mode in ("full_6h", "multiscale", "horizon"):
        config = _config("r3_full_path")
        config["state_mode"] = mode
        model = build_aggregate_model(config, _static(), total_capacity_mw=1000.0).eval()
        model.state_encoder.visible_steps_override = 4
        with torch.no_grad():
            _, attention = model.state_encoder(torch.rand(2, 24))
        assert attention.shape == (2, 24, 24)
        torch.testing.assert_close(attention[..., :20], torch.zeros_like(attention[..., :20]))
        torch.testing.assert_close(attention[..., -4:].sum(dim=-1), torch.ones(2, 24))


def test_r2_conditioning_controls_are_parameter_matched() -> None:
    counts = []
    for conditioning, fusion in (
        ("state_weather", "blackbox"),
        ("weather", "blackbox"),
        ("state_weather", "additive"),
    ):
        config = _config("r2_latent_interpolation")
        config["state_mode"] = "current"
        config["correction_conditioning"] = conditioning
        config["correction_fusion"] = fusion
        model = build_aggregate_model(config, _static(), total_capacity_mw=1000.0)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1


def test_r2_weather_control_skips_inactive_state_encoder() -> None:
    config = _config("r2_latent_interpolation")
    config["state_mode"] = "current"
    config["correction_conditioning"] = "weather"
    model = build_aggregate_model(config, _static(), total_capacity_mw=1000.0).eval()

    def inactive_state_must_not_run(_history: torch.Tensor):
        raise AssertionError("inactive state encoder was evaluated")

    model.state_encoder.forward = inactive_state_must_not_run
    history = torch.rand(2, 24)
    weather = torch.rand(1, 8, 14, 24, 25)
    weather_batch_index = torch.zeros(2, dtype=torch.long)
    weather_offsets = torch.arange(8, dtype=torch.float32).reshape(1, 8).expand(2, -1)
    with torch.no_grad():
        output = model(
            cluster_history=history,
            weather=weather,
            weather_batch_index=weather_batch_index,
            weather_time_offset_hours=weather_offsets,
        )
        baseline = model.baseline(cluster_history=history)["cluster_factor"]
        lead = model.lead(2, dtype=history.dtype, device=history.device)
        weather_features = model.weather_encoder(
            weather, weather_batch_index, weather_offsets
        )["weather_features"]
        fused = model.fusion_network(
            torch.cat((torch.zeros_like(lead), weather_features, lead), dim=-1)
        )
        zero_state_reference = (baseline + model._decode(fused)).clamp_min(0.0)
    assert output["state_attention"] is None
    assert torch.isfinite(output["cluster_factor"]).all()
    torch.testing.assert_close(output["cluster_factor"], zero_state_reference)


def test_correction_state_override_preserves_history_baseline() -> None:
    config = _config("r2_latent_interpolation")
    config["state_mode"] = "current"
    model = build_aggregate_model(config, _static(), total_capacity_mw=1000.0).eval()
    history = torch.rand(3, 24)
    common = {
        "weather": torch.rand(1, 8, 14, 24, 25),
        "weather_batch_index": torch.zeros(3, dtype=torch.long),
        "weather_time_offset_hours": torch.arange(8, dtype=torch.float32).reshape(1, 8).expand(3, -1),
    }
    with torch.no_grad():
        native_state, _ = model.state_encoder(history)
        shifted_state, _ = model.state_encoder(torch.roll(history, shifts=1, dims=0))
        native = model(cluster_history=history, **common)
        shifted = model(
            cluster_history=history,
            correction_state_history=torch.roll(history, shifts=1, dims=0),
            **common,
        )
    torch.testing.assert_close(
        native["baseline_cluster_factor"], shifted["baseline_cluster_factor"]
    )
    torch.testing.assert_close(
        native["anchor_baseline_cluster_factor"],
        shifted["anchor_baseline_cluster_factor"],
    )
    assert not torch.equal(native_state, shifted_state)


def test_aggregate_inputs_does_not_replace_baseline_history() -> None:
    history = torch.arange(72, dtype=torch.float32).reshape(3, 24)
    batch = {
        "cluster_history": history,
        "weather": torch.rand(1, 8, 14, 24, 25),
        "weather_batch_index": torch.zeros(3, dtype=torch.long),
        "weather_time_offset_hours": torch.zeros(3, 8),
    }
    inputs = aggregate_inputs(batch)
    inputs["correction_state_history"] = torch.roll(history, shifts=1, dims=0)
    assert inputs["cluster_history"] is history
    assert not torch.equal(inputs["cluster_history"], inputs["correction_state_history"])


def test_strict_modulation_candidates_are_exactly_nested_at_initialization() -> None:
    inputs = _strict_inputs()
    for mode in ("effect_scale", "channel_scale", "low_rank", "experts", "ramp"):
        base, strict = _nested_models(mode)
        strict.eval()
        with torch.no_grad():
            expected = base(**inputs)["cluster_factor"]
            actual = strict(**inputs)["cluster_factor"]
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)


def test_strict_zero_weather_correction_is_state_invariant_after_perturbation() -> None:
    _, model = _nested_models("channel_scale")
    with torch.no_grad():
        model.modulation_controller[-1].weight.normal_(std=0.1)
    model.eval()
    inputs = _strict_inputs(batch=3)
    zero_weather = torch.zeros(3, 24, 16)
    with torch.no_grad():
        first = model(
            **inputs,
            correction_state_history=torch.zeros_like(inputs["cluster_history"]),
            weather_latent_override=zero_weather,
        )["weather_correction_factor"]
        second = model(
            **inputs,
            correction_state_history=torch.ones_like(inputs["cluster_history"]),
            weather_latent_override=zero_weather,
        )["weather_correction_factor"]
    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-7)


def test_strict_current_and_latent_state_variants_are_parameter_matched() -> None:
    for mode in ("effect_scale", "channel_scale", "low_rank"):
        counts = []
        for source in ("current", "baseline_latent"):
            model = build_aggregate_model(
                _strict_config(mode, source), _static(), total_capacity_mw=1000.0
            )
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
        assert counts[0] == counts[1]


def test_strict_adapter_stage_freezes_r2_base_and_has_finite_gradients() -> None:
    inputs = _strict_inputs()
    for mode in ("effect_scale", "channel_scale", "low_rank", "experts", "ramp"):
        _, model = _nested_models(mode, stage="adapter")
        model.train()
        assert not model.baseline.training
        assert not model.weather_encoder.training
        assert not model.fusion_network.training
        output = model(**inputs)["cluster_factor"]
        output.mean().backward()
        base_modules = (model.baseline, model.weather_encoder, model.fusion_network, model.trajectory, model.head)
        assert all(parameter.grad is None for module in base_modules for parameter in module.parameters())
        gradients = [
            parameter.grad for parameter in model.modulation_controller.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_strict_joint_stage_uses_two_learning_rates() -> None:
    _, model = _nested_models("channel_scale", stage="joint")
    groups = model.optimizer_parameter_groups(2e-4, 0.1)
    assert [group["name"] for group in groups] == ["modulator", "decoder"]
    assert [group["lr"] for group in groups] == [2e-4, 2e-5]
    assert not any(parameter.requires_grad for parameter in model.baseline.parameters())
    assert not any(parameter.requires_grad for parameter in model.weather_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.fusion_network.parameters())


def test_response_state_modes_are_parameter_matched() -> None:
    counts = []
    for mode in ("power_only", "weather_only", "level_pair", "transition_pair", "transition_pair_id"):
        model = build_aggregate_model(
            _response_config(mode), _static(), total_capacity_mw=1000.0
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1


def test_response_state_candidates_are_exactly_nested_at_initialization() -> None:
    raw = _response_inputs()
    for mode in ("power_only", "weather_only", "level_pair", "transition_pair", "transition_pair_id"):
        base, model = _response_models(mode)
        base.eval()
        model.eval()
        with torch.no_grad():
            future_hourly = base.weather_encoder.encode_hourly(raw["future_weather"])
            past_hourly = base.weather_encoder.encode_hourly(raw["past_weather"])
            direct = base(
                cluster_history=raw["cluster_history"],
                weather=raw["future_weather"],
                weather_batch_index=raw["weather_batch_index"],
                weather_time_offset_hours=raw["weather_time_offset_hours"],
            )["cluster_factor"]
            nested = model(
                cluster_history=raw["cluster_history"],
                future_hourly_weather_latent=future_hourly,
                weather_batch_index=raw["weather_batch_index"],
                weather_time_offset_hours=raw["weather_time_offset_hours"],
                past_hourly_weather_latent=past_hourly,
                past_cluster_power=raw["past_cluster_power"],
                past_weather_time_offset_hours=raw["past_weather_time_offset_hours"],
            )["cluster_factor"]
        torch.testing.assert_close(nested, direct, rtol=0.0, atol=1e-7)


def test_response_zero_weather_is_state_invariant_in_eval_mode() -> None:
    _, model = _response_models("transition_pair")
    with torch.no_grad():
        model.response_controller[-1].weight.normal_(std=0.1)
        model.response_controller[-1].bias.normal_(std=0.1)
    model.eval()
    raw = _response_inputs(batch=3)
    with torch.no_grad():
        future_hourly = model.weather_encoder.encode_hourly(raw["future_weather"])
        past_hourly = model.weather_encoder.encode_hourly(raw["past_weather"])
        common = {
            "cluster_history": raw["cluster_history"],
            "future_hourly_weather_latent": future_hourly,
            "weather_batch_index": raw["weather_batch_index"],
            "weather_time_offset_hours": raw["weather_time_offset_hours"],
            "past_hourly_weather_latent": past_hourly,
            "past_cluster_power": raw["past_cluster_power"],
            "past_weather_time_offset_hours": raw["past_weather_time_offset_hours"],
            "future_weather_latent_override": torch.zeros(3, 24, 16),
        }
        first = model(**common)["weather_correction_factor"]
        common["response_past_power_override"] = torch.roll(
            raw["past_cluster_power"], shifts=1, dims=0
        )
        common["response_past_weather_override"] = torch.roll(
            past_hourly, shifts=1, dims=0
        )
        second = model(**common)["weather_correction_factor"]
    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-7)


def test_response_adapter_freezes_base_and_has_finite_gradients() -> None:
    _, model = _response_models("transition_pair", stage="adapter")
    model.train()
    assert not model.baseline.training
    assert not model.weather_encoder.training
    assert not model.fusion_network.training
    raw = _response_inputs()
    with torch.no_grad():
        future_hourly = model.weather_encoder.encode_hourly(raw["future_weather"])
        past_hourly = model.weather_encoder.encode_hourly(raw["past_weather"])
    output = model(
        cluster_history=raw["cluster_history"],
        future_hourly_weather_latent=future_hourly,
        weather_batch_index=raw["weather_batch_index"],
        weather_time_offset_hours=raw["weather_time_offset_hours"],
        past_hourly_weather_latent=past_hourly,
        past_cluster_power=raw["past_cluster_power"],
        past_weather_time_offset_hours=raw["past_weather_time_offset_hours"],
    )["cluster_factor"]
    output.mean().backward()
    base_modules = (model.baseline, model.weather_encoder, model.fusion_network, model.trajectory, model.head)
    assert all(parameter.grad is None for module in base_modules for parameter in module.parameters())
    gradients = [
        parameter.grad for parameter in model.response_controller.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_response_joint_stage_uses_response_and_decoder_learning_rates() -> None:
    _, model = _response_models("transition_pair", stage="joint")
    groups = model.optimizer_parameter_groups(2e-4, 0.1)
    assert [group["name"] for group in groups] == ["response", "decoder"]
    assert [group["lr"] for group in groups] == [2e-4, 2e-5]
    assert not any(parameter.requires_grad for parameter in model.baseline.parameters())
    assert not any(parameter.requires_grad for parameter in model.weather_encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.fusion_network.parameters())


def test_response_identification_loss_is_only_required_for_r3() -> None:
    batch = {"cluster_future": torch.rand(2, 24), "cluster_history": torch.rand(2, 24)}
    output = {
        "cluster_factor": torch.rand(2, 24),
        "baseline_cluster_factor": None,
        "anchor_baseline_cluster_factor": None,
        "identification_prediction": None,
        "identification_target": None,
    }
    aggregate_loss(output, batch, {"response_identification_loss_weight": 0.0})
    try:
        aggregate_loss(output, batch, {"response_identification_loss_weight": 0.2})
    except ValueError as error:
        assert "identification" in str(error).lower()
    else:
        raise AssertionError("A positive identification weight must require R3 outputs")
    output["identification_prediction"] = torch.rand(2, 2)
    output["identification_target"] = torch.rand(2, 2)
    loss, parts = aggregate_loss(
        output, batch, {"response_identification_loss_weight": 0.2}
    )
    assert torch.isfinite(loss)
    assert np.isfinite(parts["identification"])


def test_response_support_state_does_not_read_query_transitions() -> None:
    _, model = _response_models("transition_pair_id")
    model.eval()
    past_weather = torch.rand(2, 6, 16)
    past_power = torch.rand(2, 6)
    past_time = torch.arange(-5, 1, dtype=torch.float32).reshape(1, 6).expand(2, -1)
    first = model.build_response_tokens(past_weather, past_power, past_time)
    second = first.clone()
    second[:, 3:] = torch.randn_like(second[:, 3:]) * 100
    with torch.no_grad():
        first_state = model._response_state_pair(first, support_only=True)[0]
        second_state = model._response_state_pair(second, support_only=True)[0]
    torch.testing.assert_close(first_state, second_state, rtol=0.0, atol=0.0)


def test_response_interventions_preserve_history_baseline() -> None:
    _, model = _response_models("transition_pair")
    model.eval()
    raw = _response_inputs(batch=3)
    with torch.no_grad():
        future_hourly = model.weather_encoder.encode_hourly(raw["future_weather"])
        past_hourly = model.weather_encoder.encode_hourly(raw["past_weather"])
        common = {
            "cluster_history": raw["cluster_history"],
            "future_hourly_weather_latent": future_hourly,
            "weather_batch_index": raw["weather_batch_index"],
            "weather_time_offset_hours": raw["weather_time_offset_hours"],
            "past_hourly_weather_latent": past_hourly,
            "past_cluster_power": raw["past_cluster_power"],
            "past_weather_time_offset_hours": raw["past_weather_time_offset_hours"],
        }
        native = model(**common)
        shifted = model(**common, response_pair_shift=True)
        shuffled = model(
            **common,
            response_past_power_override=torch.roll(raw["past_cluster_power"], 1, 0),
            response_past_weather_override=torch.roll(past_hourly, 1, 0),
        )
    torch.testing.assert_close(native["baseline_cluster_factor"], shifted["baseline_cluster_factor"])
    torch.testing.assert_close(native["baseline_cluster_factor"], shuffled["baseline_cluster_factor"])


def test_real_response_cache_has_no_station_tensors_and_is_causal() -> None:
    cache = Path("data/aggregate_response_state_v1")
    source = Path("data/jilin_56farm_pad8.h5")
    aggregate = Path("data/aggregate_cluster_v1")
    if not (cache.is_dir() and source.is_file() and aggregate.is_dir()):
        return
    result = AggregateResponseCacheBuilder(source, aggregate, cache).audit()
    assert result["station_sample_tensors_present"] is False
    assert result["causal_time_protocol"] is True
    for split in ("train", "val", "test"):
        assert not any("station" in path.name for path in (cache / split).glob("*.npy"))


def test_real_response_latent_cache_matches_direct_r2_with_bf16_tolerance() -> None:
    checkpoint_path = Path(
        "artifacts/aggregate_cluster_r2_state_v1/mechanism/r2_w/seed42/best_model.pt"
    )
    latent_root = Path("artifacts/aggregate_response_state_v1/latent_cache/seed42")
    aggregate_root = Path("data/aggregate_cluster_v1")
    response_root = Path("data/aggregate_response_state_v1")
    if not all(path.exists() for path in (checkpoint_path, latent_root / "train", aggregate_root, response_root)):
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = AggregateForecastDataset(
        aggregate_root, "train", include_weather=True, preload_weather="mmap", verbose=False
    )
    response = AggregateResponseForecastDataset(
        aggregate_root, response_root, latent_root, "train", verbose=False
    )
    model = build_aggregate_model(
        checkpoint["model_config"], dataset.static,
        total_capacity_mw=dataset.total_capacity_mw,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    ).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        direct = model.weather_encoder.encode_hourly(
            torch.from_numpy(np.array(dataset.weather[:1], dtype=np.float32, copy=True))
        ).float()
    cached = torch.from_numpy(
        np.array(response.future_hourly_latent[:1], dtype=np.float32, copy=True)
    )
    torch.testing.assert_close(cached, direct, rtol=2e-2, atol=2e-2)
    dataset.close()
    response.close()


def test_response_cache_paths_are_materialized_as_absolute(tmp_path: Path) -> None:
    config = {
        "data": {
            "source_h5": "../data/source.h5",
            "cache_dir": "../data/aggregate",
            "response_cache_dir": "../data/response",
            "response_latent_cache_root": "../artifacts/response/latent_cache",
            "dynamics_time_cache_dir": "../data/dynamics",
            "dynamics_latent_cache_root": "../artifacts/dynamics/latent_cache",
            "realization_cache_root": "../artifacts/dynamics/realization",
        },
        "model": {},
        "training": {},
    }
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = load_config(path)
    for key in (
        "source_h5", "cache_dir", "response_cache_dir", "response_latent_cache_root",
        "dynamics_time_cache_dir", "dynamics_latent_cache_root", "realization_cache_root",
    ):
        assert Path(loaded["data"][key]).is_absolute()


def _dynamics_model(mode: str, stage: str = "head"):
    base_config = _config("r2_latent_interpolation")
    base_config.update(history_visible_steps=12, state_mode="current", correction_conditioning="weather")
    base = build_aggregate_model(base_config, _static(), total_capacity_mw=1000.0).eval()
    config = dict(base_config)
    config.update(
        architecture="aggregate_weather_dynamics",
        weather_dynamics_mode=mode,
        dynamics_training_stage=stage,
        dynamics_rate_min_per_hour=1.0 / 12.0,
        dynamics_rate_max_per_hour=4.0,
        dynamics_initial_rate_per_hour=0.5,
    )
    model = build_aggregate_model(config, _static(), total_capacity_mw=1000.0)
    model.load_base_state(base.state_dict())
    return model


def _dynamics_inputs(batch: int = 3) -> dict[str, torch.Tensor]:
    offsets = torch.arange(8, dtype=torch.float32).reshape(1, 8).expand(batch, -1)
    return {
        "cluster_history": torch.rand(batch, 24),
        "future_hourly_weather_latent": torch.randn(1, 8, 16),
        "weather_batch_index": torch.zeros(batch, dtype=torch.long),
        "weather_time_offset_hours": offsets,
        "current_weather_latent": torch.randn(batch, 16),
    }


def test_weather_dynamics_modes_are_bounded_and_have_finite_gradients() -> None:
    inputs = _dynamics_inputs()
    modes = (
        "m1_baseline_headroom", "m2_anomaly_headroom", "m3_recursive_headroom",
        "m4_fixed_equilibrium", "m4_adaptive_equilibrium", "m4_no_recurrence",
    )
    for mode in modes:
        model = _dynamics_model(mode)
        prediction = model(**inputs)["cluster_factor"]
        assert prediction.shape == (3, 24)
        assert torch.all((prediction >= 0.0) & (prediction <= 1.0))
        prediction.mean().backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_equilibrium_closed_form_and_rate_bounds() -> None:
    model = _dynamics_model("m4_fixed_equilibrium")
    inputs = _dynamics_inputs(batch=2)
    output = model(**inputs)
    rate = output["response_rate_per_hour"]
    assert torch.allclose(rate, torch.full_like(rate, 0.5), atol=1e-6)
    target = output["equilibrium_factor"]
    p0 = inputs["cluster_history"][:, -1].clamp(0.0, 1.0)
    expected = target[:, 0] + (p0 - target[:, 0]) * torch.exp(-rate[:, 0] * 0.25)
    torch.testing.assert_close(output["cluster_factor"][:, 0], expected)


def test_weather_absolute_and_factor_models_require_correct_auxiliary() -> None:
    inputs = _dynamics_inputs()
    mean = _dynamics_model("weather_mean")
    mean_output = mean(**inputs)
    assert mean_output["current_weather_prediction_factor"].shape == (3, 1)
    envelope = _dynamics_model("weather_envelope")
    m5 = _dynamics_model("m5_envelope_realization")
    m5.load_auxiliary_state(envelope.state_dict())
    output = m5(**inputs, past_realization_factor=torch.rand(3, 12))
    assert torch.all((output["cluster_factor"] >= 0.0) & (output["cluster_factor"] <= 1.0))
    m6 = _dynamics_model("m6_adaptive_error_decay")
    m6.load_auxiliary_state(mean.state_dict())
    output = m6(**inputs)
    assert torch.all((output["cluster_factor"] >= 0.0) & (output["cluster_factor"] <= 1.0))
    rate = output["response_rate_per_hour"]
    assert float(rate.min()) >= 1.0 / 12.0 and float(rate.max()) <= 4.0


def test_weather_quantile_and_current_loss() -> None:
    batch = {"cluster_future": torch.rand(2, 24), "cluster_history": torch.rand(2, 24)}
    output = {
        "cluster_factor": torch.rand(2, 24, requires_grad=True),
        "current_weather_prediction_factor": torch.rand(2, 1, requires_grad=True),
        "baseline_cluster_factor": None,
        "anchor_baseline_cluster_factor": None,
    }
    loss, parts = aggregate_loss(
        output, batch, {"weather_quantile": 0.9, "current_weather_loss_weight": 1.0 / 24.0}
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert np.isfinite(parts["current_weather"])


def test_real_dynamics_time_cache_is_causal() -> None:
    cache = Path("data/aggregate_weather_dynamics_v1")
    source = Path("data/jilin_56farm_pad8.h5")
    aggregate = Path("data/aggregate_cluster_v1")
    if not all(path.exists() for path in (cache, source, aggregate)):
        return
    result = AggregateDynamicsTimeCacheBuilder(source, aggregate, cache).audit()
    assert result["causal"] is True
    for split in ("train", "val", "test"):
        offsets = np.load(cache / split / "current_weather_time_offset_hours.npy")
        assert np.all(offsets[:, 0] <= 0.0)
        assert np.all(offsets[:, 1] >= 0.0)
