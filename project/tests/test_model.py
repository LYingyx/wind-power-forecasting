from __future__ import annotations

import numpy as np
import torch

from wind_forcing.models import (
    build_forecast_model,
    build_forced_evolution_model,
    build_spatial_pretrain_model,
)
from wind_forcing.models.components import (
    ForcingPathEncoder,
    RegionalContextEncoder,
    WeatherFrameEncoder,
    WeatherTargetQuery,
    build_station_patch_metadata,
)
from wind_forcing.models.controlled_transition import (
    ControlledMambaTransition,
    SelectiveStateSpaceTransition,
)
from wind_forcing.training import average_model_states, deterministic_loss
from scripts.evaluate_forced_evolution import subset_weather_nodes


def _static() -> dict[str, np.ndarray]:
    latitude = np.arange(47.0, 45.0, -0.25, dtype=np.float32)
    longitude = np.arange(122.0, 124.0, 0.25, dtype=np.float32)
    return {
        "weather_latitude": latitude,
        "weather_longitude": longitude,
        "station_latitude": np.asarray([46.5, 46.25, 45.75], dtype=np.float32),
        "station_longitude": np.asarray([122.5, 123.0, 123.25], dtype=np.float32),
        "station_capacity_mw": np.asarray([100.0, 150.0, 200.0], dtype=np.float32),
        "nwp_mean": np.zeros(14, dtype=np.float32),
        "nwp_std": np.ones(14, dtype=np.float32),
    }


def _config() -> dict[str, object]:
    return {
        "architecture": "forced_evolution",
        "weather_channels": 14,
        "weather_hidden_channels": 8,
        "d_model": 16,
        "station_query_mode": "learned_cells",
        "station_neighborhood_cells": 3,
        "include_wind_geometry": True,
        "include_region_context": False,
        "forcing_path_mode": "tcn",
        "time_query_mode": "full_attention",
        "fusion_mode": "state_conditioned",
        "output_level": "hierarchical",
        "history_tcn_layers": 2,
        "trajectory_tcn_layers": 2,
        "attention_heads": 4,
        "dropout": 0.0,
    }


def test_checkpoint_state_averaging_handles_float_and_discrete_tensors() -> None:
    states = [
        {
            "weight": torch.tensor([1.0, 3.0], dtype=torch.float32),
            "counter": torch.tensor(2, dtype=torch.long),
        },
        {
            "weight": torch.tensor([3.0, 5.0], dtype=torch.float32),
            "counter": torch.tensor(4, dtype=torch.long),
        },
    ]
    averaged = average_model_states(states)
    torch.testing.assert_close(averaged["weight"], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(averaged["counter"], torch.tensor(4))


def test_spatial_and_full_models_preserve_hierarchical_shapes() -> None:
    torch.manual_seed(3)
    static = _static()
    weather = torch.randn(1, 8, 14, 8, 8)
    spatial = build_spatial_pretrain_model(_config(), static)
    spatial_output = spatial(weather)
    assert spatial_output["hourly_station_factor"].shape == (1, 3, 8)
    assert spatial_output["cluster_factor"].shape == (1, 8)
    assert spatial_output["spatial_attention"].shape == (1, 8, 3, 16)
    torch.testing.assert_close(
        spatial_output["spatial_attention"].sum(dim=-1),
        torch.ones(1, 8, 3),
    )

    model = build_forced_evolution_model(_config(), static)
    history = torch.rand(2, 3, 24)
    output = model(
        weather=weather,
        weather_batch_index=torch.zeros(2, dtype=torch.long),
        weather_time_offset_hours=torch.tensor(
            [[0.75 + step for step in range(8)]] * 2
        ),
        station_history=history,
        cluster_history=torch.rand(2, 24),
        origin_phase=torch.tensor([1, 2]),
    )
    assert output["station_factor"].shape == (2, 3, 24)
    assert output["cluster_factor"].shape == (2, 24)
    capacity = torch.as_tensor(static["station_capacity_mw"]).reshape(1, -1, 1)
    expected_cluster = (output["station_factor"] * capacity).sum(dim=1) / capacity.sum()
    torch.testing.assert_close(output["cluster_factor"], expected_cluster)
    output["cluster_factor"].mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_station_patch_modes_have_explicit_cell_geometry() -> None:
    latitude = np.arange(48.0, 39.75, -0.25, dtype=np.float32)
    longitude = np.arange(119.0, 128.25, 0.25, dtype=np.float32)
    station_latitude = np.asarray([43.12], dtype=np.float32)
    station_longitude = np.asarray([123.18], dtype=np.float32)

    expected_tokens = {
        ("nearest", 1): 1,
        ("bilinear", 1): 4,
        ("learned_cells", 1): 4,
        ("learned_cells", 3): 16,
        ("learned_cells", 5): 36,
    }
    for (query_mode, cell_count), token_count in expected_tokens.items():
        indices, geometry, weights = build_station_patch_metadata(
            latitude,
            longitude,
            station_latitude,
            station_longitude,
            query_mode,
            cell_count,
        )
        assert indices.shape == (1, token_count)
        assert geometry.shape == (1, token_count, 5)
        assert weights.shape == (1, token_count)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

    _, _, bilinear_weights = build_station_patch_metadata(
        latitude,
        longitude,
        station_latitude,
        station_longitude,
        "bilinear",
        1,
    )
    north = int(np.searchsorted(-latitude, -station_latitude[0], side="right") - 1)
    west = int(np.searchsorted(longitude, station_longitude[0], side="right") - 1)
    north_fraction = (station_latitude[0] - latitude[north + 1]) / (
        latitude[north] - latitude[north + 1]
    )
    west_fraction = (station_longitude[0] - longitude[west]) / (
        longitude[west + 1] - longitude[west]
    )
    expected_bilinear = np.asarray(
        [
            (1.0 - west_fraction) * north_fraction,
            west_fraction * north_fraction,
            (1.0 - west_fraction) * (1.0 - north_fraction),
            west_fraction * (1.0 - north_fraction),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(bilinear_weights[0], expected_bilinear, atol=1e-6)


def test_pointwise_weather_encoder_has_no_spatial_leakage() -> None:
    torch.manual_seed(7)
    encoder = WeatherFrameEncoder(
        nwp_channels=14,
        hidden_channels=8,
        d_model=16,
        nwp_mean=np.zeros(14, dtype=np.float32),
        nwp_std=np.ones(14, dtype=np.float32),
    ).eval()
    weather = torch.randn(1, 2, 14, 6, 7)
    perturbed = weather.clone()
    perturbed[:, :, :, 1, 2] += 10.0
    with torch.no_grad():
        encoded = encoder(weather)
        encoded_perturbed = encoder(perturbed)
    torch.testing.assert_close(encoded[:, :, :, 4, 5], encoded_perturbed[:, :, :, 4, 5])
    assert not torch.allclose(encoded[:, :, :, 1, 2], encoded_perturbed[:, :, :, 1, 2])


def test_regional_context_is_a_separate_spatial_mixing_branch() -> None:
    torch.manual_seed(11)
    encoder = RegionalContextEncoder(d_model=8).eval()
    features = torch.randn(1, 2, 8, 7, 7)
    perturbed = features.clone()
    perturbed[:, :, :, 0, 0] += 5.0
    with torch.no_grad():
        context = encoder(features)
        perturbed_context = encoder(perturbed)
    assert context.shape == (1, 2, 8)
    assert not torch.allclose(context, perturbed_context)


def test_station_conditioned_multiscale_query_exposes_both_scales() -> None:
    torch.manual_seed(17)
    static = _static()
    config = _config()
    config.update(regional_pool_size=2, regional_neighborhood_cells=1)
    weather = torch.randn(1, 8, 14, 8, 8)
    output = build_spatial_pretrain_model(config, static)(weather)

    assert output["spatial_attention"].shape == (1, 8, 3, 16)
    assert output["regional_spatial_attention"].shape == (1, 8, 3, 4)
    assert output["spatial_scale_attention"].shape == (1, 8, 3, 2)
    torch.testing.assert_close(
        output["regional_spatial_attention"].sum(dim=-1),
        torch.ones(1, 8, 3),
    )
    torch.testing.assert_close(
        output["spatial_scale_attention"].sum(dim=-1),
        torch.ones(1, 8, 3),
    )
    output["cluster_factor"].mean().backward()


def test_fixed_station_queries_return_expected_attention_sizes() -> None:
    static = _static()
    weather = torch.randn(1, 8, 14, 8, 8)
    for query_mode, expected_tokens in (("nearest", 1), ("bilinear", 4)):
        config = _config()
        config.update(
            station_query_mode=query_mode,
            station_neighborhood_cells=1,
            include_wind_geometry=False,
        )
        output = build_spatial_pretrain_model(config, static)(weather)
        assert output["spatial_attention"].shape == (1, 8, 3, expected_tokens)
        torch.testing.assert_close(
            output["spatial_attention"].sum(dim=-1),
            torch.ones(1, 8, 3),
        )


def test_fixed_gaussian_and_raw_bilinear_spatial_controls() -> None:
    torch.manual_seed(23)
    static = _static()
    static["station_latitude"] = static["station_latitude"] + 0.08
    static["station_longitude"] = static["station_longitude"] + 0.11
    weather = torch.randn(1, 8, 14, 8, 8, requires_grad=True)
    outputs = {}
    models = {}
    for mode, cells in (("fixed_gaussian", 3), ("raw_bilinear", 1), ("bilinear", 1)):
        config = _config()
        config.update(
            station_query_mode=mode,
            station_neighborhood_cells=cells,
            include_wind_geometry=False,
            forcing_path_mode="difference",
        )
        models[mode] = build_spatial_pretrain_model(config, static).eval()
        outputs[mode] = models[mode](weather)
        torch.testing.assert_close(
            outputs[mode]["spatial_attention"].sum(dim=-1),
            torch.ones_like(outputs[mode]["spatial_attention"][..., 0]),
        )
        assert torch.all(outputs[mode]["spatial_attention"] >= 0)
    assert torch.all(outputs["fixed_gaussian"]["spatial_attention"] > 0)
    assert outputs["fixed_gaussian"]["spatial_attention"].shape[-1] == 16
    assert outputs["raw_bilinear"]["spatial_attention"].shape[-1] == 4

    # Raw and latent bilinear have identical parameters.  Loading the same
    # state isolates the non-commuting interpolation/point-encoding order.
    models["raw_bilinear"].load_state_dict(models["bilinear"].state_dict())
    raw = models["raw_bilinear"](weather)["cluster_factor"]
    latent = models["bilinear"](weather)["cluster_factor"]
    assert not torch.allclose(raw, latent)
    (raw.mean() + latent.mean()).backward()
    assert weather.grad is not None and torch.isfinite(weather.grad).all()


def test_cross_resolution_mechanism_controls_are_parameter_matched() -> None:
    torch.manual_seed(29)
    query = torch.randn(2, 3, 24, 16)
    context = torch.randn(2, 3, 8, 16)
    query_times = torch.arange(1, 25, dtype=torch.float32).reshape(1, -1) / 4
    query_times = query_times.expand(2, -1)
    context_times = torch.arange(8, dtype=torch.float32).reshape(1, -1).expand(2, -1)
    variants = {
        "t2": WeatherTargetQuery(16, 4, 0.0, "bracketing_mlp"),
        "phase_off": WeatherTargetQuery(
            16, 4, 0.0, "bracketing_mlp", bracket_time_feature_mode="zero"
        ),
        "linear": WeatherTargetQuery(
            16, 4, 0.0, "bracketing_mlp", bracket_mapper_activation="identity"
        ),
    }
    counts = {sum(p.numel() for p in module.parameters()) for module in variants.values()}
    assert len(counts) == 1
    captured: list[torch.Tensor] = []
    handle = variants["phase_off"].local_mlp[0].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    for module in variants.values():
        output = module(query, context, query_times, context_times)
        assert output.shape == query.shape and torch.isfinite(output).all()
    handle.remove()
    assert captured and torch.count_nonzero(captured[0][..., -2:]) == 0
    assert isinstance(variants["linear"].local_mlp[1], torch.nn.Identity)

    actual = ForcingPathEncoder(16, "difference", 4, 0.0, difference_input="actual")
    zero = ForcingPathEncoder(16, "difference", 4, 0.0, difference_input="zero")
    zero.load_state_dict(actual.state_dict())
    forcing = torch.randn(2, 3, 8, 16, requires_grad=True)
    encoded_inputs: list[torch.Tensor] = []
    handle = zero.encoder[0].register_forward_pre_hook(
        lambda _module, args: encoded_inputs.append(args[0].detach())
    )
    zero_output = zero(forcing)
    handle.remove()
    assert zero_output.shape == forcing.shape
    assert torch.count_nonzero(encoded_inputs[0][..., 16:]) == 0
    assert sum(p.numel() for p in actual.parameters()) == sum(
        p.numel() for p in zero.parameters()
    )
    zero_output.mean().backward()
    assert forcing.grad is not None and torch.isfinite(forcing.grad).all()


def test_new_cross_resolution_defaults_preserve_legacy_state_contract() -> None:
    static = _static()
    legacy_config = _config()
    explicit_config = dict(legacy_config)
    explicit_config.update(
        forcing_difference_input="actual",
        bracket_time_feature_mode="relative_distances",
        bracket_mapper_activation="gelu",
    )
    legacy = build_forced_evolution_model(legacy_config, static).eval()
    explicit = build_forced_evolution_model(explicit_config, static).eval()
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    inputs = _forecast_inputs()
    with torch.no_grad():
        legacy_output = legacy(**inputs)["cluster_factor"]
        explicit_output = explicit(**inputs)["cluster_factor"]
    torch.testing.assert_close(legacy_output, explicit_output)


def test_weather_node_subsets_preserve_time_alignment_and_boundaries() -> None:
    batch = {
        "weather": torch.arange(2 * 8 * 3).reshape(2, 8, 3),
        "weather_time_offset_hours": torch.arange(4 * 8).reshape(4, 8),
        "weather_station_target": torch.arange(2 * 5 * 8).reshape(2, 5, 8),
        "cluster_history": torch.ones(4, 24),
    }
    retained = subset_weather_nodes(batch, (0, 2, 4, 6))
    assert retained["weather"].shape == (2, 4, 3)
    assert retained["weather_time_offset_hours"].shape == (4, 4)
    torch.testing.assert_close(retained["weather"][:, :, 0], batch["weather"][:, (0, 2, 4, 6), 0])
    torch.testing.assert_close(
        retained["weather_time_offset_hours"],
        batch["weather_time_offset_hours"][:, (0, 2, 4, 6)],
    )
    torch.testing.assert_close(
        retained["weather_station_target"],
        batch["weather_station_target"][..., (0, 2, 4, 6)],
    )
    assert retained["cluster_history"] is batch["cluster_history"]
    try:
        subset_weather_nodes(batch, (0, 8))
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-bound weather node was accepted")


def _forecast_inputs() -> dict[str, torch.Tensor]:
    return {
        "weather": torch.randn(1, 8, 14, 8, 8),
        "weather_batch_index": torch.zeros(2, dtype=torch.long),
        "weather_time_offset_hours": torch.tensor(
            [[0.75 + step for step in range(8)]] * 2
        ),
        "station_history": torch.rand(2, 3, 24),
        "cluster_history": torch.rand(2, 24),
        "origin_phase": torch.tensor([1, 2]),
    }


def test_all_target_time_queries_and_fusions_produce_quarter_hour_outputs() -> None:
    static = _static()
    inputs = _forecast_inputs()
    for mode in (
        "nearest",
        "latent_linear",
        "bracketing_mlp",
        "bracketing_attention",
        "full_attention",
    ):
        config = _config()
        config["time_query_mode"] = mode
        output = build_forecast_model(config, static)(**inputs)
        assert output["cluster_factor"].shape == (2, 24)

    for mode in (
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
    ):
        config = _config()
        config["fusion_mode"] = mode
        model = build_forecast_model(config, static)
        selected_inputs = dict(inputs)
        if mode == "history_only":
            for key in ("weather", "weather_batch_index", "weather_time_offset_hours"):
                selected_inputs.pop(key)
        output = model(**selected_inputs)
        assert output["station_factor"].shape == (2, 3, 24)
        assert output["cluster_factor"].shape == (2, 24)


def test_cross_resolution_adapters_are_shape_and_parameter_matched() -> None:
    static = _static()
    inputs = _forecast_inputs()
    parameter_counts = set()
    for mode in (
        "raw_linear",
        "latent_linear",
        "bracketing_mlp",
        "nearest",
        "full_attention",
    ):
        config = _config()
        config.update(
            architecture="incremental_correction",
            time_query_mode=mode,
            weather_view_mode="legacy",
            correction_conditioning="history_weather",
            correction_history_mode="cluster_recent_1h",
            correction_fusion_mode="blackbox",
            matched_conditioning_backbone=True,
            baseline_anchor_enabled=True,
            freeze_history_baseline=False,
        )
        model = build_forecast_model(config, static)
        output = model(**inputs)
        assert output["station_factor"].shape == (2, 3, 24)
        assert output["cluster_factor"].shape == (2, 24)
        # T0 reuses the encoder for interpolated raw frames, but the native
        # hourly auxiliary target must retain its original eight anchors.
        assert output["hourly_station_factor_unique"].shape == (1, 3, 8)
        parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
        output["cluster_factor"].mean().backward()
        assert model.forcing_encoder.frame_encoder.network[0].weight.grad is not None
    assert len(parameter_counts) == 1

    control_counts = set()
    for updates in (
        {},
        {"bracket_time_feature_mode": "zero"},
        {"bracket_mapper_activation": "identity"},
        {"forcing_difference_input": "zero"},
    ):
        config = _config()
        config.update(
            architecture="incremental_correction",
            time_query_mode="bracketing_mlp",
            weather_view_mode="legacy",
            correction_conditioning="history_weather",
            correction_history_mode="cluster_recent_1h",
            correction_fusion_mode="blackbox",
            matched_conditioning_backbone=True,
            baseline_anchor_enabled=True,
            freeze_history_baseline=False,
            **updates,
        )
        model = build_forecast_model(config, static)
        output = model(**inputs)
        assert output["cluster_factor"].shape == (2, 24)
        assert torch.isfinite(output["cluster_factor"]).all()
        control_counts.add(sum(parameter.numel() for parameter in model.parameters()))
    assert len(control_counts) == 1


def test_incremental_correction_is_additive_frozen_and_projectable(tmp_path) -> None:
    static = _static()
    inputs = _forecast_inputs()
    basis_path = tmp_path / "basis.npy"
    np.save(basis_path, np.eye(3, dtype=np.float32))
    for projection_mode in ("full", "temporal", "spatial", "joint"):
        config = _config()
        config.update(
            architecture="incremental_correction",
            correction_conditioning="history_weather",
            projection_mode=projection_mode,
            temporal_components=4,
            spatial_components=2,
            spatial_basis_path=str(basis_path),
        )
        model = build_forecast_model(config, static)
        output = model(**inputs)
        assert output["station_factor"].shape == (2, 3, 24)
        assert output["weather_correction_factor"].shape == (2, 3, 24)
        assert all(
            not parameter.requires_grad
            for parameter in model.baseline_model.parameters()
        )
        model.train()
        assert model.baseline_model.training is False
        output["cluster_factor"].mean().backward()
        assert model.correction_head.weight.grad is not None


def test_explicit_correction_fusions_are_output_additive_and_parameter_matched() -> None:
    static = _static()
    inputs = _forecast_inputs()
    models = {}
    for fusion_mode in (
        "additive",
        "multiplicative",
        "asymmetric_modulation",
    ):
        config = _config()
        config.update(
            architecture="incremental_correction",
            correction_conditioning="history_weather",
            correction_history_mode="cluster_recent_1h",
            correction_fusion_mode=fusion_mode,
            matched_conditioning_backbone=True,
            projection_mode="full",
            temporal_components=24,
        )
        models[fusion_mode] = build_forecast_model(config, static).eval()

    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }
    assert len(parameter_counts) == 1

    for fusion_mode, model in models.items():
        output = model(**inputs)
        weather_main = output["weather_main_correction_factor"]
        if fusion_mode == "asymmetric_modulation":
            assert weather_main is None
            assert output["state_main_correction_factor"] is None
            assert output["state_weather_interaction_correction_factor"] is None
            assert output["horizon_interaction_gate"] is None
            assert output["weather_correction_factor"].shape == (2, 3, 24)
        elif fusion_mode == "additive":
            assert weather_main.shape == (2, 3, 24)
            second = output["state_main_correction_factor"]
            assert output["state_weather_interaction_correction_factor"] is None
            assert output["horizon_interaction_gate"] is None
            torch.testing.assert_close(
                output["weather_correction_factor"], weather_main + second
            )
        else:
            assert weather_main.shape == (2, 3, 24)
            second = output["state_weather_interaction_correction_factor"]
            assert output["state_main_correction_factor"] is None
            assert output["horizon_interaction_gate"] is None
            torch.testing.assert_close(
                output["weather_correction_factor"], weather_main + second
            )
        output["cluster_factor"].mean().backward()
        assert model.correction_head.weight.grad is not None

    blackbox_config = _config()
    blackbox_config.update(
        architecture="incremental_correction",
        correction_conditioning="history_weather",
        correction_history_mode="cluster_recent_1h",
        correction_fusion_mode="blackbox",
        matched_conditioning_backbone=True,
        projection_mode="full",
        temporal_components=24,
    )
    blackbox = build_forecast_model(blackbox_config, static)
    explicit_count = next(iter(parameter_counts))
    blackbox_count = sum(parameter.numel() for parameter in blackbox.parameters())
    assert explicit_count - blackbox_count == 7 * int(blackbox_config["d_model"]) + 4


def test_horizon_adaptive_control_objects_share_gate_and_weather_only_limit() -> None:
    static = _static()
    inputs = _forecast_inputs()
    modes = (
        "horizon_film",
        "horizon_gated_multiplicative",
        "horizon_conditioned_query",
        "horizon_attention_bias",
        "horizon_conditional_lora",
    )
    parameter_counts = []
    for fusion_mode in modes:
        config = _config()
        config.update(
            architecture="incremental_correction",
            correction_conditioning="history_weather",
            correction_history_mode="cluster_recent_1h",
            correction_fusion_mode=fusion_mode,
            interaction_rank=16,
            matched_conditioning_backbone=True,
            time_query_mode="full_attention",
            projection_mode="full",
            temporal_components=24,
        )
        model = build_forecast_model(config, static).eval()
        parameter_counts.append(sum(p.numel() for p in model.parameters()))
        output = model(**inputs)
        gate = output["horizon_interaction_gate"]
        assert gate.shape == (1, 1, 24)
        torch.testing.assert_close(gate, torch.ones_like(gate))
        assert output["weather_correction_factor"].shape == (2, 3, 24)
        if fusion_mode in {
            "horizon_conditioned_query",
            "horizon_attention_bias",
        }:
            shift = output["weather_attention_shift_l1"]
            assert shift.shape == (2, 24)
            assert torch.all(shift >= 0.0)
        else:
            assert output["weather_attention_shift_l1"] is None
        if fusion_mode not in {
            "horizon_conditioned_query",
            "horizon_attention_bias",
        }:
            assert output["interaction_update_abs_mean"].shape == (2, 24)

        # With g=0, correction is a weather-only path and must not change when
        # the correction-side recent state is perturbed.  The total forecast is
        # not compared because the frozen baseline legitimately sees history.
        with torch.no_grad():
            model.horizon_gate_logits.fill_(-100.0)
        weather_only = model(**inputs)
        changed = dict(inputs)
        changed["station_history"] = inputs["station_history"] + 0.37
        changed["cluster_history"] = inputs["cluster_history"] - 0.21
        changed_output = model(**changed)
        torch.testing.assert_close(
            weather_only["weather_correction_factor"],
            changed_output["weather_correction_factor"],
            atol=2e-6,
            rtol=2e-6,
        )

    # The toy fixture omits most of the production weather encoder, so compare
    # the operator budget in absolute terms here; the formal smoke audit checks
    # the requested <1% spread on the complete ~1.29M-parameter models.
    assert max(parameter_counts) - min(parameter_counts) < 2_000


def test_joint_incremental_training_keeps_baseline_explicit(tmp_path) -> None:
    static = _static()
    config = _config()
    basis_path = tmp_path / "basis.npy"
    np.save(basis_path, np.eye(3, dtype=np.float32))
    config.update(
        architecture="incremental_correction",
        correction_conditioning="history_weather",
        projection_mode="full",
        temporal_components=24,
        spatial_components=3,
        spatial_basis_path=str(basis_path),
        freeze_history_baseline=False,
    )
    model = build_forecast_model(config, static)
    output = model(**_forecast_inputs())
    assert output["baseline_station_factor"].shape == (2, 3, 24)
    assert output["baseline_cluster_factor"].shape == (2, 24)
    assert any(
        parameter.requires_grad for parameter in model.baseline_model.parameters()
    )


def test_incremental_history_state_ladder_exposes_only_requested_information() -> None:
    static = _static()
    station = torch.arange(2 * 3 * 24, dtype=torch.float32).reshape(2, 3, 24)
    cluster = torch.arange(2 * 24, dtype=torch.float32).reshape(2, 24)
    conditioned = {}
    for mode in (
        "full_history",
        "recent_1h",
        "cluster_recent_1h",
        "cluster_current",
        "station_current",
        "cluster_current_ramp15m",
    ):
        config = _config()
        config.update(
            architecture="incremental_correction",
            correction_conditioning="history_weather",
            correction_history_mode=mode,
            matched_conditioning_backbone=True,
            projection_mode="full",
            temporal_components=24,
        )
        model = build_forecast_model(config, static)
        conditioned[mode] = model._condition_correction_history(station, cluster)

    torch.testing.assert_close(conditioned["full_history"][0], station)
    torch.testing.assert_close(conditioned["full_history"][1], cluster)

    recent_station, recent_cluster = conditioned["recent_1h"]
    torch.testing.assert_close(recent_station[..., -4:], station[..., -4:])
    torch.testing.assert_close(recent_cluster[..., -4:], cluster[..., -4:])
    torch.testing.assert_close(
        recent_station[..., :-4],
        station[..., -4:-3].expand(-1, -1, 20),
    )
    torch.testing.assert_close(
        recent_cluster[..., :-4], cluster[..., -4:-3].expand(-1, 20)
    )

    aggregate_recent_station, aggregate_recent_cluster = conditioned[
        "cluster_recent_1h"
    ]
    torch.testing.assert_close(
        aggregate_recent_cluster[..., -4:], cluster[..., -4:]
    )
    torch.testing.assert_close(
        aggregate_recent_cluster[..., :-4],
        cluster[..., -4:-3].expand(-1, 20),
    )
    torch.testing.assert_close(
        aggregate_recent_station,
        aggregate_recent_cluster[:, None, :].expand(-1, 3, -1),
    )

    cluster_station, cluster_only = conditioned["cluster_current"]
    torch.testing.assert_close(
        cluster_only, cluster[..., -1:].expand(-1, 24)
    )
    torch.testing.assert_close(
        cluster_station,
        cluster[..., -1:].reshape(2, 1, 1).expand(-1, 3, 24),
    )

    current_station, current_cluster = conditioned["station_current"]
    torch.testing.assert_close(
        current_station, station[..., -1:].expand(-1, -1, 24)
    )
    torch.testing.assert_close(
        current_cluster, cluster[..., -1:].expand(-1, 24)
    )

    ramp_station, ramp_cluster = conditioned["cluster_current_ramp15m"]
    torch.testing.assert_close(
        ramp_cluster[..., :-1], cluster[..., -2:-1].expand(-1, 23)
    )
    torch.testing.assert_close(ramp_cluster[..., -1], cluster[..., -1])
    torch.testing.assert_close(
        ramp_station, ramp_cluster[:, None, :].expand(-1, 3, -1)
    )


def test_cross_farm_recent_contexts_are_parameter_matched_and_information_explicit() -> None:
    static = _static()
    inputs = _forecast_inputs()
    histories = torch.arange(2 * 3 * 24, dtype=torch.float32).reshape(2, 3, 24)
    weather_trajectory = torch.stack(
        (
            torch.zeros(8),
            torch.full((8,), 0.1),
            torch.full((8,), 10.0),
        ),
        dim=0,
    ).unsqueeze(0)
    models = {}
    contexts = {}
    for mode in (
        "self_duplicate",
        "geographic",
        "global_other",
        "weather_coherent",
    ):
        config = _config()
        config.update(
            architecture="incremental_correction",
            correction_conditioning="history_weather",
            correction_history_mode="recent_1h",
            cross_farm_context_mode=mode,
            cross_farm_neighbors=1,
            matched_conditioning_backbone=True,
            projection_mode="full",
            temporal_components=24,
        )
        model = build_forecast_model(config, static)
        models[mode] = model
        contexts[mode] = model._cross_farm_history(
            histories, weather_trajectory, torch.zeros(2, dtype=torch.long)
        )
        assert model(**inputs)["cluster_factor"].shape == (2, 24)

    assert len(
        {
            sum(parameter.numel() for parameter in model.parameters())
            for model in models.values()
        }
    ) == 1
    torch.testing.assert_close(contexts["self_duplicate"], histories)
    torch.testing.assert_close(
        contexts["global_other"],
        (histories.sum(dim=1, keepdim=True) - histories) / 2.0,
    )
    geographic_index = models["geographic"].geographic_neighbor_index
    torch.testing.assert_close(
        contexts["geographic"], histories[:, geographic_index, :].mean(dim=2)
    )
    expected_weather_index = torch.tensor([1, 0, 1])
    torch.testing.assert_close(
        contexts["weather_coherent"], histories[:, expected_weather_index, :]
    )


def test_regional_composition_ladder_is_nested_permutation_invariant_and_matched() -> None:
    static = _static()
    inputs = _forecast_inputs()
    station = torch.arange(2 * 3 * 24, dtype=torch.float32).reshape(2, 3, 24) / 100.0
    cluster = torch.rand(2, 24)
    models = {}
    encoded_inputs = {}
    for mode in (
        "composition_control",
        "composition_statistics",
        "composition_deepsets",
    ):
        config = _config()
        config.update(
            architecture="incremental_correction",
            correction_conditioning="history_weather",
            correction_history_mode="recent_1h",
            cross_farm_context_mode=mode,
            cross_farm_neighbors=1,
            matched_conditioning_backbone=True,
            projection_mode="full",
            temporal_components=24,
        )
        model = build_forecast_model(config, static).eval()
        models[mode] = model
        encoded_inputs[mode] = model._regional_composition_inputs(station, cluster)
        output = model(**inputs)
        assert output["cluster_factor"].shape == (2, 24)
        assert output["composition_statistics_state"].shape == (2, 16)
        assert output["composition_set_state"].shape == (2, 16)

    assert len(
        {
            sum(parameter.numel() for parameter in model.parameters())
            for model in models.values()
        }
    ) == 1
    control_statistics, control_set = encoded_inputs["composition_control"]
    statistics_statistics, statistics_set = encoded_inputs["composition_statistics"]
    deepsets_statistics, deepsets_set = encoded_inputs["composition_deepsets"]
    assert control_statistics.shape == statistics_statistics.shape == (2, 25)
    assert control_set.shape == statistics_set.shape == deepsets_set.shape == (2, 3, 5)
    assert not torch.allclose(control_statistics, statistics_statistics)
    torch.testing.assert_close(statistics_statistics, deepsets_statistics)
    torch.testing.assert_close(control_set, statistics_set)
    assert not torch.allclose(statistics_set, deepsets_set)

    model = models["composition_deepsets"]
    statistics_state, set_state, _ = model._regional_composition_state(station, cluster)
    permuted_statistics_state, permuted_set_state, _ = model._regional_composition_state(
        station[:, torch.tensor([2, 0, 1])], cluster
    )
    torch.testing.assert_close(statistics_state, permuted_statistics_state)
    torch.testing.assert_close(set_state, permuted_set_state)


def test_matched_incremental_factorial_masks_share_one_backbone(tmp_path) -> None:
    static = _static()
    inputs = _forecast_inputs()
    models = {}
    for conditioning in (
        "context_only",
        "history_only",
        "weather_only",
        "history_weather",
    ):
        config = _config()
        config.update(
            architecture="incremental_correction",
            correction_conditioning=conditioning,
            matched_conditioning_backbone=True,
            projection_mode="full",
            temporal_components=24,
            freeze_history_baseline=False,
        )
        models[conditioning] = build_forecast_model(config, static).eval()

    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }
    assert len(parameter_counts) == 1
    history_output = models["history_only"](**inputs)
    weather_perturbed = dict(inputs)
    weather_perturbed["weather"] = inputs["weather"] + 10.0
    history_changed = models["history_only"](**weather_perturbed)
    torch.testing.assert_close(
        history_output["weather_correction_factor"],
        history_changed["weather_correction_factor"],
    )

    weather_output = models["weather_only"](**inputs)
    history_perturbed = dict(inputs)
    history_perturbed["station_history"] = inputs["station_history"] + 0.2
    history_perturbed["cluster_history"] = inputs["cluster_history"] + 0.2
    weather_changed = models["weather_only"](**history_perturbed)
    torch.testing.assert_close(
        weather_output["weather_correction_factor"],
        weather_changed["weather_correction_factor"],
    )


def test_pretrained_baseline_anchor_is_explicit_and_supervised() -> None:
    static = _static()
    config = _config()
    config.update(
        architecture="incremental_correction",
        correction_conditioning="history_weather",
        matched_conditioning_backbone=True,
        baseline_anchor_enabled=True,
        projection_mode="full",
        temporal_components=24,
        freeze_history_baseline=False,
    )
    model = build_forecast_model(config, static)
    checkpoint = {
        "model_config": {"architecture": "forced_evolution", "fusion_mode": "history_only"},
        "model_state": model.baseline_model.state_dict(),
    }
    model.load_history_checkpoint(checkpoint)
    with torch.no_grad():
        model.baseline_model.power_delta.bias.add_(0.1)
    output = model(**_forecast_inputs())
    assert output["anchor_baseline_station_factor"].shape == (2, 3, 24)
    assert output["anchor_baseline_cluster_factor"].shape == (2, 24)
    batch = {
        "station_future": torch.rand(2, 3, 24),
        "cluster_future": torch.rand(2, 24),
        "weather_station_target": torch.rand(1, 3, 8),
    }
    loss, parts = deterministic_loss(
        output,
        batch,
        {
            "station_loss_weight": 1.0,
            "cluster_loss_weight": 2.0,
            "auxiliary_weather_loss_weight": 0.0,
            "baseline_anchor_loss_weight": 0.1,
        },
    )
    assert parts["baseline_anchor"] > 0.0
    loss.backward()
    assert model.baseline_model.power_delta.bias.grad is not None


def test_horizon_weather_view_gate_matches_preregistered_schedule() -> None:
    static = _static()
    config = _config()
    config.update(
        architecture="incremental_correction",
        correction_conditioning="history_weather",
        correction_history_mode="recent_1h",
        matched_conditioning_backbone=True,
        freeze_history_baseline=False,
        weather_view_mode="horizon_mix",
        weather_gate_mode="fixed",
        specialist_mode="shared_head",
    )
    model = build_forecast_model(config, static).eval()
    output = model(**_forecast_inputs())
    gate = output["weather_view_gate"]
    assert gate.shape == (2, 24)
    lead = torch.arange(1, 25, dtype=torch.float32) / 4.0
    expected = ((lead - 1.0) / 2.0).clamp(0.0, 1.0)
    torch.testing.assert_close(gate, expected.reshape(1, -1).expand(2, -1))
    assert output["near_specialist_correction_factor"] is None
    assert output["full_specialist_correction_factor"] is None


def test_learned_weather_gate_is_monotonic_and_bounded() -> None:
    static = _static()
    config = _config()
    config.update(
        architecture="incremental_correction",
        correction_conditioning="history_weather",
        correction_history_mode="recent_1h",
        matched_conditioning_backbone=True,
        freeze_history_baseline=False,
        weather_view_mode="horizon_mix",
        weather_gate_mode="monotonic_sigmoid",
        specialist_mode="shared_head",
    )
    model = build_forecast_model(config, static)
    gate = model._weather_view_gate(torch.zeros(1))
    assert gate.shape == (1, 1, 24)
    assert torch.all(gate[..., 1:] >= gate[..., :-1])
    assert torch.all((gate > 0.0) & (gate < 1.0))
    center, slope = model.weather_gate_parameters()
    assert center is not None and slope is not None
    assert 1.0 <= float(center.detach()) <= 3.0
    assert float(slope.detach()) >= 0.5
    gate.mean().backward()
    assert model.weather_gate_center_raw.grad is not None
    assert model.weather_gate_slope_raw.grad is not None


def test_dual_weather_specialists_have_matched_capacity_and_diagnostics() -> None:
    static = _static()
    models = []
    for weather_view_mode in ("full", "horizon_mix"):
        config = _config()
        config.update(
            architecture="incremental_correction",
            correction_conditioning="history_weather",
            correction_history_mode="recent_1h",
            matched_conditioning_backbone=True,
            freeze_history_baseline=False,
            weather_view_mode=weather_view_mode,
            weather_gate_mode="fixed",
            specialist_mode="dual_head",
        )
        models.append(build_forecast_model(config, static).eval())
    assert sum(parameter.numel() for parameter in models[0].parameters()) == sum(
        parameter.numel() for parameter in models[1].parameters()
    )
    for model in models:
        output = model(**_forecast_inputs())
        assert output["weather_view_gate"].shape == (2, 24)
        assert output["near_specialist_correction_factor"].shape == (2, 3, 24)
        assert output["full_specialist_correction_factor"].shape == (2, 3, 24)
        output["cluster_factor"].mean().backward()
        assert model.full_correction_head is not None
        assert model.full_correction_head.weight.grad is not None


def test_horizon_anchor_taper_preserves_scalar_mode_and_zeros_long_leads() -> None:
    baseline_station = torch.full((1, 2, 24), 0.2, requires_grad=True)
    baseline_cluster = torch.full((1, 24), 0.2, requires_grad=True)
    output = {
        "station_factor": torch.zeros(1, 2, 24, requires_grad=True),
        "cluster_factor": torch.zeros(1, 24, requires_grad=True),
        "hourly_station_factor_unique": None,
        "baseline_station_factor": baseline_station,
        "baseline_cluster_factor": baseline_cluster,
        "anchor_baseline_station_factor": torch.zeros(1, 2, 24),
        "anchor_baseline_cluster_factor": torch.zeros(1, 24),
        "multirate_station_factor": None,
        "multirate_cluster_factor": None,
    }
    batch = {
        "station_future": torch.zeros(1, 2, 24),
        "cluster_future": torch.zeros(1, 24),
        "weather_station_target": torch.zeros(1, 2, 8),
    }
    settings = {
        "station_loss_weight": 1.0,
        "cluster_loss_weight": 2.0,
        "auxiliary_weather_loss_weight": 0.0,
        "baseline_anchor_loss_weight": 0.2,
        "anchor_schedule_mode": "scalar",
    }
    _, scalar_parts = deterministic_loss(output, batch, settings)
    expected = 3.0 * torch.nn.functional.smooth_l1_loss(
        torch.full((1,), 0.2), torch.zeros(1), beta=0.05
    )
    assert abs(scalar_parts["baseline_anchor"] - float(expected)) < 1e-7
    settings["anchor_schedule_mode"] = "horizon_taper"
    tapered_loss, tapered_parts = deterministic_loss(output, batch, settings)
    assert 0.0 < tapered_parts["baseline_anchor"] < scalar_parts["baseline_anchor"]
    tapered_loss.backward()
    assert baseline_cluster.grad is not None
    assert torch.all(baseline_cluster.grad[..., :12] != 0.0)
    assert torch.all(baseline_cluster.grad[..., 15:] == 0.0)


def test_correction_forcing_ablation_preserves_history_checkpoint_contract() -> None:
    static = _static()
    history_config = _config()
    history_config.update(
        architecture="forced_evolution",
        fusion_mode="history_only",
        forcing_path_mode="difference",
        time_query_mode="full_attention",
    )
    history_model = build_forecast_model(history_config, static)
    correction_config = dict(history_config)
    correction_config.update(
        architecture="incremental_correction",
        forcing_path_mode="independent",
        time_query_mode="nearest",
        station_query_mode="bilinear",
        station_neighborhood_cells=1,
        include_wind_geometry=False,
        correction_conditioning="history_weather",
        correction_history_mode="cluster_recent_1h",
        baseline_station_query_mode="learned_cells",
        baseline_station_neighborhood_cells=3,
        baseline_include_wind_geometry=True,
        baseline_forcing_path_mode="difference",
        baseline_time_query_mode="full_attention",
    )
    correction_model = build_forecast_model(correction_config, static)
    correction_model.load_history_checkpoint(
        {
            "model_config": {
                "architecture": "forced_evolution",
                "fusion_mode": "history_only",
            },
            "model_state": history_model.state_dict(),
        }
    )
    assert set(correction_model.baseline_model.state_dict()) == set(
        history_model.state_dict()
    )


def test_distillation_loss_supports_plain_and_disagreement_weighted_teacher() -> None:
    output = {
        "station_factor": torch.full((2, 3, 4), 0.4, requires_grad=True),
        "cluster_factor": torch.full((2, 4), 0.4, requires_grad=True),
        "hourly_station_factor_unique": None,
        "multirate_station_factor": None,
        "multirate_cluster_factor": None,
    }
    batch = {
        "station_future": torch.full((2, 3, 4), 0.5),
        "cluster_future": torch.full((2, 4), 0.5),
        "weather_station_target": torch.rand(1, 3, 8),
        "teacher_station_factor_mean": torch.full((2, 3, 4), 0.45),
        "teacher_cluster_factor_mean": torch.full((2, 4), 0.45),
        "teacher_station_factor_std": torch.full((2, 3, 4), 0.02),
        "teacher_cluster_factor_std": torch.full((2, 4), 0.01),
    }
    settings = {
        "station_loss_weight": 1.0,
        "cluster_loss_weight": 2.0,
        "auxiliary_weather_loss_weight": 0.0,
        "distillation_weight": 0.5,
    }
    supervised_loss, _ = deterministic_loss(
        output,
        batch,
        {**settings, "distillation_weight": 0.0},
    )
    plain_loss, plain_parts = deterministic_loss(output, batch, settings)
    aware_loss, aware_parts = deterministic_loss(
        output,
        batch,
        {
            **settings,
            "distillation_uncertainty_aware": True,
            "distillation_station_scale": 0.02,
            "distillation_cluster_scale": 0.01,
        },
    )
    assert plain_parts["distillation"] > 0.0
    assert aware_parts["distillation"] > 0.0
    assert plain_loss > supervised_loss
    assert torch.isfinite(plain_loss)
    assert torch.isfinite(aware_loss)
    (plain_loss + aware_loss).backward()
    assert output["cluster_factor"].grad is not None


def test_distillation_loss_can_ablate_station_and_cluster_teacher_targets() -> None:
    output = {
        "station_factor": torch.full((2, 3, 4), 0.4, requires_grad=True),
        "cluster_factor": torch.full((2, 4), 0.4, requires_grad=True),
        "hourly_station_factor_unique": None,
        "multirate_station_factor": None,
        "multirate_cluster_factor": None,
    }
    batch = {
        "station_future": torch.full((2, 3, 4), 0.5),
        "cluster_future": torch.full((2, 4), 0.5),
        "weather_station_target": torch.rand(1, 3, 8),
        "teacher_station_factor_mean": torch.full((2, 3, 4), 0.45),
        "teacher_cluster_factor_mean": torch.full((2, 4), 0.45),
    }
    settings = {
        "station_loss_weight": 1.0,
        "cluster_loss_weight": 2.0,
        "auxiliary_weather_loss_weight": 0.0,
        "distillation_weight": 0.5,
    }
    station_only, station_parts = deterministic_loss(
        output,
        batch,
        {
            **settings,
            "distillation_station_loss_weight": 1.0,
            "distillation_cluster_loss_weight": 0.0,
        },
    )
    cluster_only, cluster_parts = deterministic_loss(
        output,
        batch,
        {
            **settings,
            "distillation_station_loss_weight": 0.0,
            "distillation_cluster_loss_weight": 2.0,
        },
    )
    both, both_parts = deterministic_loss(output, batch, settings)
    assert station_parts["distillation"] > 0.0
    assert cluster_parts["distillation"] > 0.0
    assert both_parts["distillation"] > station_parts["distillation"]
    assert both_parts["distillation"] > cluster_parts["distillation"]
    assert torch.isfinite(station_only + cluster_only + both)

def test_controlled_transition_backbones_share_the_forecast_contract() -> None:
    static = _static()
    inputs = _forecast_inputs()
    for transition_type in ("gru", "selective_ssm", "controlled_koopman"):
        config = _config()
        config.update(
            architecture="controlled_transition",
            transition_type=transition_type,
            multirate_enabled=False,
        )
        model = build_forecast_model(config, static)
        output = model(**inputs)
        assert output["station_factor"].shape == (2, 3, 24)
        assert output["cluster_factor"].shape == (2, 24)
        output["cluster_factor"].mean().backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_selective_ssm_is_causal_across_transition_steps() -> None:
    torch.manual_seed(13)
    transition = SelectiveStateSpaceTransition(d_model=8, dropout=0.0).eval()
    initial = torch.randn(1, 2, 8)
    control = torch.randn(1, 2, 6, 8)
    target = torch.randn(1, 2, 6, 8)
    perturbed = control.clone()
    perturbed[:, :, 3] += 3.0
    with torch.no_grad():
        baseline = transition(initial, control, target, 0.25)
        changed = transition(initial, perturbed, target, 0.25)
    torch.testing.assert_close(baseline[:, :, :3], changed[:, :, :3])
    assert not torch.allclose(baseline[:, :, 3:], changed[:, :, 3:])


def test_official_mamba_wrapper_preserves_causality(monkeypatch) -> None:
    import sys
    import types

    class FakeMamba(torch.nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.projection = torch.nn.Linear(d_model, d_model)

        def forward(self, values):
            return self.projection(values).cumsum(dim=1)

    module = types.ModuleType("mamba_ssm")
    module.Mamba = FakeMamba
    module.Mamba2 = FakeMamba
    monkeypatch.setitem(sys.modules, "mamba_ssm", module)
    initial = torch.randn(1, 2, 8)
    control = torch.randn(1, 2, 6, 8)
    target = torch.randn(1, 2, 6, 8)
    for version in ("mamba", "mamba2"):
        transition = ControlledMambaTransition(
            8,
            0.0,
            version=version,
            d_state=4,
            d_conv=2,
            expand=2,
            headdim=8,
        ).eval()
        perturbed = control.clone()
        perturbed[:, :, 3] += 3.0
        baseline = transition(initial, control, target, 0.25)
        changed = transition(initial, perturbed, target, 0.25)
        assert baseline.shape == (1, 2, 6, 8)
        torch.testing.assert_close(baseline[:, :, :3], changed[:, :, :3])
        assert not torch.allclose(baseline[:, :, 3:], changed[:, :, 3:])
        baseline.mean().backward()


def test_multirate_transition_branch_has_supervised_consistency_loss() -> None:
    static = _static()
    inputs = _forecast_inputs()
    config = _config()
    config.update(
        architecture="controlled_transition",
        transition_type="selective_ssm",
        multirate_enabled=True,
    )
    output = build_forecast_model(config, static)(**inputs)
    assert output["multirate_station_factor"].shape == (2, 3, 6)
    assert output["multirate_cluster_factor"].shape == (2, 6)
    batch = {
        "station_future": torch.rand(2, 3, 24),
        "cluster_future": torch.rand(2, 24),
        "weather_station_target": torch.rand(1, 3, 8),
    }
    loss, parts = deterministic_loss(
        output,
        batch,
        {
            "station_loss_weight": 1.0,
            "cluster_loss_weight": 2.0,
            "auxiliary_weather_loss_weight": 0.0,
            "multirate_target_loss_weight": 0.25,
            "multirate_consistency_loss_weight": 0.1,
        },
    )
    assert torch.isfinite(loss)
    assert parts["multirate_target"] > 0.0
    assert parts["multirate_consistency"] >= 0.0


def test_ramp_loss_supervises_origin_to_future_power_changes() -> None:
    station_future = torch.tensor([[[0.3, 0.5, 0.4], [0.2, 0.4, 0.7]]])
    cluster_future = station_future.mean(dim=1)
    output = {
        "station_factor": torch.full_like(station_future, 0.3),
        "cluster_factor": torch.full_like(cluster_future, 0.3),
        "hourly_station_factor_unique": None,
        "multirate_station_factor": None,
        "multirate_cluster_factor": None,
    }
    batch = {
        "station_history": torch.tensor([[[0.1], [0.2]]]),
        "cluster_history": torch.tensor([[0.15]]),
        "station_future": station_future,
        "cluster_future": cluster_future,
    }
    settings = {
        "station_loss_weight": 1.0,
        "cluster_loss_weight": 2.0,
        "auxiliary_weather_loss_weight": 0.0,
        "ramp_loss_weight": 0.2,
    }
    loss_with_ramp, parts = deterministic_loss(output, batch, settings)
    settings["ramp_loss_weight"] = 0.0
    loss_without_ramp, _ = deterministic_loss(output, batch, settings)

    assert parts["ramp"] > 0.0
    torch.testing.assert_close(
        loss_with_ramp - loss_without_ramp,
        torch.as_tensor(0.2 * parts["ramp"]),
    )


def test_direct_cluster_and_all_baseline_families_have_common_output_contract() -> None:
    static = _static()
    inputs = _forecast_inputs()
    config = _config()
    config["output_level"] = "direct_cluster"
    direct = build_forecast_model(config, static)(**inputs)
    assert direct["station_factor"] is None
    assert direct["cluster_factor"].shape == (2, 24)

    history_architectures = (
        "history_persistence",
        "history_linear",
        "history_dlinear",
        "history_lstm",
        "history_tcn",
        "history_transformer",
        "history_patchtst",
        "history_tft",
        "history_itransformer",
    )
    for architecture in history_architectures:
        baseline_config = _config()
        baseline_config["architecture"] = architecture
        output = build_forecast_model(baseline_config, static)(
            station_history=inputs["station_history"],
            cluster_history=inputs["cluster_history"],
        )
        assert output["station_factor"].shape == (2, 3, 24)
        assert output["cluster_factor"].shape == (2, 24)

    weather_architectures = (
        "weather_nearest",
        "weather_bilinear",
        "weather_resnet2d",
        "weather_3dcnn",
        "weather_convlstm",
        "weather_vit",
        "weather_swin",
        "weather_video_swin",
    )
    for architecture in weather_architectures:
        baseline_config = _config()
        baseline_config.update(architecture=architecture, output_level="direct_cluster")
        output = build_forecast_model(baseline_config, static)(**inputs)
        assert output["station_factor"] is None
        assert output["cluster_factor"].shape == (2, 24)
