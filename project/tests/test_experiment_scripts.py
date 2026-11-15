from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.prepare_transition_base import build_transition_config


def _source() -> dict:
    return {
        "seed": 1,
        "model": {"architecture": "forced_evolution", "output_level": "hierarchical"},
        "training": {"deterministic": {"auxiliary_weather_loss_weight": 0.0}},
    }


def _template() -> dict:
    return {
        "model": {"transition_type": "gru", "multirate_enabled": False},
        "training": {
            "deterministic": {
                "multirate_target_loss_weight": 0.0,
                "multirate_consistency_loss_weight": 0.0,
            }
        },
    }


def test_transition_base_accepts_auxiliary_weather_weight() -> None:
    result = build_transition_config(
        deepcopy(_source()),
        deepcopy(_template()),
        seed=42,
        auxiliary_weather_loss_weight=0.2,
    )
    assert result["seed"] == 42
    assert result["model"]["architecture"] == "controlled_transition"
    assert result["model"]["transition_type"] == "gru"
    assert result["training"]["deterministic"][
        "auxiliary_weather_loss_weight"
    ] == pytest.approx(0.2)


def test_transition_base_optionally_copies_mamba_settings() -> None:
    template = _template()
    template["model"].update(
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
    )
    result = build_transition_config(
        deepcopy(_source()),
        template,
        seed=42,
        auxiliary_weather_loss_weight=0.2,
        include_mamba_settings=True,
    )
    assert result["model"]["mamba_d_state"] == 64
    assert result["model"]["mamba_headdim"] == 64


def test_transition_base_rejects_negative_auxiliary_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_transition_config(
            deepcopy(_source()),
            deepcopy(_template()),
            seed=42,
            auxiliary_weather_loss_weight=-0.1,
        )
