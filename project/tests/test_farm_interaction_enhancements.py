from __future__ import annotations

import numpy as np
import pytest
import torch

from wind_forcing.models.local_state_identifiability import (
    build_local_state_identifiability_model,
)


def static() -> dict[str, np.ndarray]:
    return {
        "station_capacity_mw": np.asarray([10.0, 20.0], dtype=np.float32),
        "station_latitude": np.asarray([2.25, 1.75], dtype=np.float32),
        "station_longitude": np.asarray([1.25, 1.75], dtype=np.float32),
        "weather_latitude": np.asarray([3.0, 2.0, 1.0, 0.0], dtype=np.float64),
        "weather_longitude": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        "nwp_mean": np.zeros(14, dtype=np.float32),
        "nwp_std": np.ones(14, dtype=np.float32),
    }


def config(
    interaction: str,
    *,
    history: str = "pooled_state",
    forcing: str = "legacy",
    context: str = "none",
) -> dict[str, object]:
    return {
        "weather_channels": 14,
        "weather_hidden_channels": 8,
        "d_model": 8,
        "history_visible_steps": 12,
        "state_visible_steps": 12,
        "history_tcn_layers": 1,
        "trajectory_tcn_layers": 1,
        "attention_heads": 2,
        "dropout": 0.0,
        "response_state_source": "local_power",
        "station_context_mode": context,
        "response_interaction_mode": interaction,
        "station_embedding_dim": 4,
        "modulation_amplitude": 1.0,
        "backbone_lr_multiplier": 1.0,
        "fine_tune_policy": "all",
        "spatial_weather_mode": "raw_bilinear",
        "station_neighborhood_cells": 3,
        "source_role_mode": "full",
        "response_history_mode": history,
        "forcing_transition_mode": forcing,
    }


def build(interaction: str, **kwargs):
    return build_local_state_identifiability_model(
        config(interaction, **kwargs), static(), history_steps=24, forecast_steps=24
    )


def inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    station = torch.rand(batch, 2, 24)
    capacity = torch.tensor([10.0, 20.0])
    return {
        "weather": torch.randn(batch, 8, 14, 4, 4),
        "weather_batch_index": torch.arange(batch),
        "weather_time_offset_hours": torch.tensor(
            [[0.25 + index for index in range(8)]] * batch, dtype=torch.float32
        ),
        "station_history": station,
        "cluster_history": (station * capacity[None, :, None]).sum(1) / capacity.sum(),
        "origin_phase": torch.zeros(batch, dtype=torch.long),
    }


@pytest.mark.parametrize("history", ["lead_summary", "weather_cross_attention"])
@pytest.mark.parametrize("interaction", ["faithful_concat", "independent_output_additive"])
def test_temporal_history_modes_have_forecast_contract(history: str, interaction: str) -> None:
    model = build(interaction, history=history).eval()
    with torch.no_grad():
        output = model(**inputs())
    assert output["station_factor"].shape == (2, 2, 24)
    attention = output["history_response_attention"]
    if history == "weather_cross_attention":
        assert attention is not None and attention.shape == (2, 2, 2, 24, 12)
    else:
        assert attention is None


@pytest.mark.parametrize("forcing", ["origin_delta_linear", "origin_delta_tcn"])
def test_forcing_transition_modes_have_forecast_contract(forcing: str) -> None:
    model = build("faithful_concat", forcing=forcing).eval()
    with torch.no_grad():
        output = model(**inputs())
    assert output["station_factor"].shape == (2, 2, 24)


@pytest.mark.parametrize(
    "context", ["metadata_residual", "metadata_film", "id_film"]
)
def test_site_context_modes_have_forecast_contract(context: str) -> None:
    model = build("faithful_concat", context=context).eval()
    with torch.no_grad():
        output = model(**inputs())
    assert output["station_factor"].shape == (2, 2, 24)


@pytest.mark.parametrize(
    ("history", "forcing", "context"),
    [
        ("lead_summary", "legacy", "none"),
        ("weather_cross_attention", "legacy", "none"),
        ("pooled_state", "origin_delta_linear", "none"),
        ("pooled_state", "origin_delta_tcn", "none"),
        ("pooled_state", "legacy", "metadata_residual"),
        ("pooled_state", "legacy", "metadata_film"),
        ("pooled_state", "legacy", "id_film"),
    ],
)
def test_each_additive_control_matches_interaction_parameter_count(
    history: str, forcing: str, context: str
) -> None:
    interaction = build(
        "faithful_concat", history=history, forcing=forcing, context=context
    )
    additive = build(
        "independent_output_additive",
        history=history,
        forcing=forcing,
        context=context,
    )
    assert sum(p.numel() for p in interaction.parameters()) == sum(
        p.numel() for p in additive.parameters()
    )


@pytest.mark.parametrize(
    ("history", "forcing", "context"),
    [
        ("lead_summary", "legacy", "none"),
        ("weather_cross_attention", "legacy", "none"),
        ("pooled_state", "origin_delta_linear", "none"),
        ("pooled_state", "origin_delta_tcn", "none"),
        ("pooled_state", "legacy", "metadata_residual"),
        ("pooled_state", "legacy", "metadata_film"),
        ("pooled_state", "legacy", "id_film"),
    ],
)
def test_additive_controls_are_history_weather_separable(
    history: str, forcing: str, context: str
) -> None:
    model = build(
        "independent_output_additive",
        history=history,
        forcing=forcing,
        context=context,
    ).eval()
    base = inputs()
    weather_a = torch.randn_like(base["weather"])
    weather_b = torch.randn_like(base["weather"])
    history_a = torch.rand_like(base["station_history"])
    history_b = torch.rand_like(base["station_history"])

    def correction(weather: torch.Tensor, station_history: torch.Tensor) -> torch.Tensor:
        values = {key: value.clone() for key, value in base.items()}
        values["weather"] = weather
        values["station_history"] = station_history
        capacity = torch.tensor([10.0, 20.0])
        values["cluster_history"] = (
            station_history * capacity[None, :, None]
        ).sum(1) / capacity.sum()
        with torch.no_grad():
            return model(**values)["local_weather_correction_logit"]

    aa = correction(weather_a, history_a)
    ab = correction(weather_a, history_b)
    ba = correction(weather_b, history_a)
    bb = correction(weather_b, history_b)
    torch.testing.assert_close(
        aa - ab - ba + bb, torch.zeros_like(aa), atol=3e-6, rtol=2e-5
    )


def test_origin_reference_is_linear_extrapolation_to_zero_hours() -> None:
    model = build("faithful_concat", forcing="origin_delta_linear").eval()
    raw = torch.zeros(1, 2, 8, 14)
    raw[:, :, 0] = 2.0
    raw[:, :, 1] = 6.0
    for index in range(2, 8):
        raw[:, :, index] = 6.0 + 4.0 * (index - 1)
    nodes = torch.zeros(1, 2, 8, 8)
    offsets = torch.tensor([[0.5 + index for index in range(8)]])
    captured: list[torch.Tensor] = []

    def hook(_module, arguments, _output):
        captured.append(arguments[0].detach())

    assert model.forcing_delta_encoder is not None
    handle = model.forcing_delta_encoder.register_forward_hook(hook)
    with torch.no_grad():
        model._forcing_transition(nodes, raw, torch.tensor([0]), offsets)
    handle.remove()
    expected_reference = torch.zeros(1, 2, 14)
    torch.testing.assert_close(captured[0][:, :, 0], raw[:, :, 0] - expected_reference)


@pytest.mark.parametrize("history", ["lead_summary", "weather_cross_attention"])
@pytest.mark.parametrize("forcing", ["origin_delta_linear", "origin_delta_tcn"])
@pytest.mark.parametrize("context", ["metadata_residual", "metadata_film", "id_film"])
def test_complete_abc_final_cf_separability_and_parameter_budget(history, forcing, context):
    models = []
    for role in ("faithful_concat", "independent_output_additive"):
        cfg = config(role, history=history, forcing=forcing, context=context)
        cfg["response_output_mode"] = "cf_residual"
        model = build_local_state_identifiability_model(cfg, static(), history_steps=24, forecast_steps=24).eval()
        models.append(model)
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())
    model = models[1]
    # Nonzero head is essential: separability with an inactive head is vacuous.
    with torch.no_grad():
        for p in model.correction_head.parameters():
            p.uniform_(-0.3, 0.3)
    values = inputs()
    ha, hb = torch.rand_like(values["station_history"]), torch.rand_like(values["station_history"])
    wa, wb = torch.randn_like(values["weather"]), torch.randn_like(values["weather"])
    def forward(h, w):
        batch = dict(values, station_history=h, weather=w)
        batch["cluster_history"] = (h * torch.tensor([10.,20.])[None,:,None]).sum(1) / 30
        with torch.no_grad():
            return model(**batch)["station_factor"]
    aa, ab, ba, bb = forward(ha, wa), forward(ha, wb), forward(hb, wa), forward(hb, wb)
    assert (aa - ab).abs().max() > 1e-5
    assert (aa - ba).abs().max() > 1e-5
    torch.testing.assert_close(aa-ab-ba+bb, torch.zeros_like(aa), atol=3e-6, rtol=0)
    grouped = [p for group in model.optimizer_parameter_groups(2e-4, 1e-4) for p in group["params"]]
    assert len(grouped) == len(set(map(id, grouped)))
    assert set(map(id, grouped)) == {id(p) for p in model.parameters() if p.requires_grad}


def test_legacy_strict_loading_and_output_link_are_unchanged():
    cfg = config("faithful_concat")
    legacy = build_local_state_identifiability_model(cfg, static(), history_steps=24, forecast_steps=24).eval()
    cfg["response_output_mode"] = "legacy_logit"
    explicit = build_local_state_identifiability_model(cfg, static(), history_steps=24, forecast_steps=24).eval()
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert not any(k.startswith(("history_summary_encoder.", "response_history_attention.", "forcing_delta_", "response_context_")) for k in legacy.state_dict())
    batch = inputs()
    with torch.no_grad():
        a, b = legacy(**batch), explicit(**batch)
    for key in a:
        if a[key] is not None:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    expected = torch.sigmoid(torch.logit(a["baseline_station_factor"]) + a["local_weather_correction_logit"])
    torch.testing.assert_close(a["station_factor"], expected)


def test_history_attention_survives_evaluation_archive_roundtrip(tmp_path):
    from wind_forcing.training import evaluate_deterministic, save_prediction_archive
    model = build("faithful_concat", history="weather_cross_attention").eval()
    batch = inputs()
    batch.update(station_future=torch.rand(2,2,24), cluster_future=torch.rand(2,24),
                 sample_index=torch.tensor([20,10]), origin_unix=torch.tensor([200,100]))
    records = []
    settings = {"station_loss_weight": 1., "cluster_loss_weight": 0., "auxiliary_weather_loss_weight": 0.}
    evaluate_deterministic(model, [batch], torch.device("cpu"), "fp32", settings,
                           prediction_batches=records, export_station_predictions=True)
    path = tmp_path / "predictions.npz"
    save_prediction_archive(path, records)
    with np.load(path) as archive:
        assert archive["sample_index"].tolist() == [10,20]
        assert archive["history_response_attention_mean"].shape == (2,24,12)
        np.testing.assert_allclose(archive["history_response_attention_mean"].sum(-1), 1., atol=1e-6)


@pytest.mark.parametrize("output",["cf_residual","bounded_cf_mix"])
@pytest.mark.parametrize("readout",["legacy","capacity_aux","level_shape"])
def test_new_readouts_keep_cf_separability_and_matched_capacity(output,readout):
    models=[]
    for role in ("faithful_concat","independent_output_additive"):
        torch.manual_seed(42)
        cfg=config(role,history="weather_cross_attention",forcing="origin_delta_tcn",context="metadata_film")
        cfg.update(response_output_mode=output,response_readout_mode=readout)
        models.append(build_local_state_identifiability_model(cfg,static(),history_steps=24,forecast_steps=24).eval())
    assert sum(p.numel() for p in models[0].parameters())==sum(p.numel() for p in models[1].parameters())
    model=models[1]
    with torch.no_grad():
        for name,p in model.named_parameters():
            if name.startswith(("correction_head.","readout_aux.")):
                p.uniform_(-.5,.5)
    batch=inputs()
    ha,hb=torch.rand_like(batch["station_history"]),torch.rand_like(batch["station_history"])
    wa,wb=torch.randn_like(batch["weather"]),torch.randn_like(batch["weather"])
    def forward(h,w):
        values=dict(batch,station_history=h,weather=w,cluster_history=(h*torch.tensor([10.,20.])[None,:,None]).sum(1)/30)
        with torch.no_grad(): return model(**values)["station_factor"]
    aa,ab,ba,bb=forward(ha,wa),forward(ha,wb),forward(hb,wa),forward(hb,wb)
    assert (aa-ab).abs().max()>1e-5 and (aa-ba).abs().max()>1e-5
    torch.testing.assert_close(aa-ab-ba+bb,torch.zeros_like(aa),atol=3e-6,rtol=0)
    if output=="bounded_cf_mix":
        assert all(((v>=0)&(v<=1)).all() for v in (aa,ab,ba,bb))
    grouped=[p for g in model.optimizer_parameter_groups(2e-4,1e-4) for p in g["params"]]
    assert len(grouped)==len(set(map(id,grouped)))
    assert set(map(id,grouped))=={id(p) for p in model.parameters() if p.requires_grad}


def test_level_shape_head_is_equal_capacity_and_mean_is_auxiliary():
    cfg=config("faithful_concat")
    cfg["response_readout_mode"]="capacity_aux"
    capacity=build_local_state_identifiability_model(cfg,static(),history_steps=24,forecast_steps=24).eval()
    cfg["response_readout_mode"]="level_shape"
    model=build_local_state_identifiability_model(cfg,static(),history_steps=24,forecast_steps=24).eval()
    model.load_state_dict(capacity.state_dict(),strict=True)
    with torch.no_grad():
        model.readout_aux[-1].weight.uniform_(-.5,.5)
        model.correction_head.weight.uniform_(-.5,.5)
    captured=[]
    hook=model.weather_trajectory.register_forward_hook(lambda m,a,o:captured.append(o))
    response,target=torch.randn(1,2,24,8),torch.randn(1,2,24,8)
    value=model._decode_weather_correction(response,target,batch=1,stations=2)
    hook.remove()
    expected=model.readout_aux(captured[0].mean(-1)).reshape(1,2)
    torch.testing.assert_close(value.mean(-1),expected,atol=1e-6,rtol=1e-5)
