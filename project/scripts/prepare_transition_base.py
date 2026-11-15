from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from wind_forcing.config import load_config


def build_transition_config(
    source: dict,
    template: dict,
    *,
    seed: int,
    auxiliary_weather_loss_weight: float,
    include_mamba_settings: bool = False,
) -> dict:
    if auxiliary_weather_loss_weight < 0.0:
        raise ValueError("auxiliary_weather_loss_weight must be non-negative")
    for config in (source, template):
        config.pop("_config_path", None)
        config.pop("_project_root", None)

    source["seed"] = seed
    source_model = source["model"]
    template_model = template["model"]
    for key in ("transition_type", "multirate_enabled"):
        source_model[key] = template_model[key]
    if include_mamba_settings:
        for key in (
            "mamba_d_state",
            "mamba_d_conv",
            "mamba_expand",
            "mamba_headdim",
        ):
            source_model[key] = template_model[key]
    source_model["architecture"] = "controlled_transition"
    source_model["output_level"] = "hierarchical"

    source_training = source["training"]["deterministic"]
    template_training = template["training"]["deterministic"]
    for key in (
        "multirate_target_loss_weight",
        "multirate_consistency_loss_weight",
    ):
        source_training[key] = template_training[key]
    source_training["auxiliary_weather_loss_weight"] = auxiliary_weather_loss_weight
    source["experiment"] = {"id": f"controlled_transition_base_seed{seed}"}
    return source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add controlled-transition settings to a selected forecast config"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--auxiliary-weather-loss-weight",
        type=float,
        default=0.0,
        help="Hourly station-weather auxiliary loss; defaults to the legacy value 0.0",
    )
    parser.add_argument(
        "--include-mamba-settings",
        action="store_true",
        help="Copy pinned official Mamba/Mamba2 hyperparameters from the template",
    )
    args = parser.parse_args()

    source = build_transition_config(
        load_config(args.source),
        load_config(args.template),
        seed=args.seed,
        auxiliary_weather_loss_weight=args.auxiliary_weather_loss_weight,
        include_mamba_settings=args.include_mamba_settings,
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(source, sort_keys=False, allow_unicode=False)
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"Transition base already differs: {output}")
        print(f"[transition-base] Reusing {output}")
        return
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, output)
    print(f"[transition-base] Created {output}")


if __name__ == "__main__":
    main()
