from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_forced_evolution import WeatherNodeSubsetLoader
from wind_forcing.config import load_config
from wind_forcing.data import ForecastDataset, create_forecast_loader
from wind_forcing.models import build_forecast_model, forecast_model_uses_weather
from wind_forcing.training import (
    evaluate_deterministic,
    resolve_device,
    save_json,
    save_prediction_archive,
    seed_everything,
)


SCENARIOS: dict[str, tuple[int, ...]] = {
    "r1_2hour": (0, 2, 4, 6),
    **{
        f"drop{dropped}": tuple(index for index in range(8) if index != dropped)
        for dropped in range(1, 7)
    },
}


def complete(directory: Path) -> bool:
    return (
        (directory / "COMPLETE").is_file()
        and (directory / "metrics_val.json").is_file()
        and (directory / "metrics_val.json").stat().st_size > 0
        and (directory / "predictions_val.npz").is_file()
        and (directory / "predictions_val.npz").stat().st_size > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all seven weather-node perturbations after one data/model load"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    pending = [name for name in SCENARIOS if not complete(output_root / name)]
    if not pending:
        print("[robustness] all seven scenarios passed resume audit", flush=True)
        return

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    data_config = config["data"]
    training = config["training"]
    print(
        f"[data] loading split=val once cache={data_config['cache_dir']} "
        f"pending_scenarios={pending}",
        flush=True,
    )
    dataset = ForecastDataset(
        data_config["cache_dir"],
        "val",
        preload_weather=data_config["preload_weather"],
        preload_weather_max_gib=float(data_config["preload_weather_max_gib"]),
        include_weather=forecast_model_uses_weather(config["model"]),
    )
    if int(dataset.weather.values.shape[1]) != 8:
        raise ValueError(
            f"Preregistered robustness matrix expects 8 native nodes, got {dataset.weather.values.shape[1]}"
        )
    base_loader = create_forecast_loader(
        dataset,
        cycles_per_batch=2 * int(training["cycles_per_batch"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("data_manifest_identity") != dataset.manifest["preprocess_fingerprint"]:
        raise ValueError("Checkpoint was trained with a different data cache")
    model = build_forecast_model(
        checkpoint["model_config"],
        dataset.static,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = resolve_device(args.device)
    model.to(device)
    settings = {**training["deterministic"], "gradient_clip": training["gradient_clip"]}
    settings["distillation_weight"] = 0.0
    settings["auxiliary_weather_loss_weight"] = 0.0
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    try:
        for scenario in pending:
            directory = output_root / scenario
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "COMPLETE").unlink(missing_ok=True)
            indices = SCENARIOS[scenario]
            loader = WeatherNodeSubsetLoader(base_loader, indices)
            batches: list[dict[str, torch.Tensor]] = []
            metrics = evaluate_deterministic(
                model,
                loader,
                device,
                str(training["precision"]),
                settings,
                max_batches=args.max_batches,
                progress_desc=f"Robustness {scenario}",
                prediction_batches=batches,
                export_station_predictions=False,
            )
            metrics["parameter_count"] = parameter_count
            metrics["model_state_key"] = "model_state"
            metrics["weather_node_perturbation"] = {
                "scenario": scenario,
                "retained_indices": list(indices),
                "dropped_index": int(scenario.removeprefix("drop")) if scenario.startswith("drop") else None,
            }
            save_json(directory / "metrics_val.json", metrics)
            save_prediction_archive(directory / "predictions_val.npz", batches)
            (directory / "COMPLETE").touch()
            print(f"[robustness] complete scenario={scenario}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
