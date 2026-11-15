from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator

import torch

from wind_forcing.config import load_config
from wind_forcing.data import (
    ForecastDataset,
    apply_stratified_weather_permutation,
    create_forecast_loader,
)
from wind_forcing.models import build_forecast_model, forecast_model_uses_weather
from wind_forcing.training import (
    evaluate_deterministic,
    resolve_device,
    save_prediction_archive,
    save_json,
    seed_everything,
)


def parse_weather_node_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("weather node indices must be integers") from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("weather node indices must be non-negative")
    if tuple(sorted(set(indices))) != indices:
        raise argparse.ArgumentTypeError(
            "weather node indices must be unique and strictly increasing"
        )
    return indices


def parse_station_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("station indices must be integers") from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("station indices must be non-empty and non-negative")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("station indices must be unique")
    return indices


def subset_weather_nodes(
    batch: dict[str, torch.Tensor], indices: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    """Return a batch with matching native weather nodes and time offsets."""
    weather = batch.get("weather")
    offsets = batch.get("weather_time_offset_hours")
    if weather is None or offsets is None:
        raise ValueError("Weather-node perturbation requires weather inputs")
    if weather.ndim < 2 or offsets.ndim != 2:
        raise ValueError("Weather tensors have invalid dimensions")
    if weather.shape[1] != offsets.shape[1]:
        raise ValueError("Weather values and time offsets have different node counts")
    if indices[-1] >= weather.shape[1]:
        raise ValueError(
            f"Weather-node index {indices[-1]} exceeds {weather.shape[1] - 1}"
        )
    index = torch.as_tensor(indices, dtype=torch.long)
    selected = dict(batch)
    selected["weather"] = weather.index_select(1, index)
    selected["weather_time_offset_hours"] = offsets.index_select(1, index)
    hourly_target = batch.get("weather_station_target")
    if hourly_target is not None:
        if hourly_target.shape[-1] != weather.shape[1]:
            raise ValueError("Hourly weather target has a different native-node count")
        selected["weather_station_target"] = hourly_target.index_select(-1, index)
    return selected


class WeatherNodeSubsetLoader:
    def __init__(self, loader: Iterable[dict[str, torch.Tensor]], indices: tuple[int, ...]):
        self.loader = loader
        self.indices = indices

    def __len__(self) -> int:
        return len(self.loader)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for batch in self.loader:
            yield subset_weather_nodes(batch, self.indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen deterministic checkpoint")
    parser.add_argument("--config", default="configs/forced_evolution_pad5.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model-state-key",
        default="model_state",
        choices=(
            "model_state",
            "ema_model_state",
            "checkpoint_average_model_state",
        ),
        help="Evaluate a state saved at the same validation-selected epoch",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--predictions-output")
    parser.add_argument("--export-station-predictions", action="store_true")
    parser.add_argument(
        "--metric-station-indices",
        type=parse_station_indices,
        help="Comma-separated station indices used for reported station metrics",
    )
    parser.add_argument(
        "--local-state-intervention",
        choices=("station_roll", "batch_roll", "reference"),
    )
    parser.add_argument(
        "--response-state-intervention",
        choices=("strong_sample", "station_permutation", "reference", "zero"),
    )
    parser.add_argument("--weather-permutation-seed", type=int)
    parser.add_argument("--weather-permutation-power-bins", type=int, default=10)
    node_group = parser.add_mutually_exclusive_group()
    node_group.add_argument(
        "--weather-node-indices",
        type=parse_weather_node_indices,
        help="Comma-separated native weather-node indices retained at inference",
    )
    node_group.add_argument(
        "--drop-weather-node-index",
        type=int,
        help="Drop one native weather node by zero-based index at inference",
    )
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument(
        "--zero-horizon-gate",
        action="store_true",
        help="Set a trained horizon-adaptive interaction gate to zero for ablation",
    )
    gate_group.add_argument(
        "--unit-horizon-gate",
        action="store_true",
        help="Set a trained horizon-adaptive interaction gate to one for ablation",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.split == "test":
        from wind_forcing.experiment_guard import require_test_authorized
        require_test_authorized(config, args.checkpoint)
    seed_everything(int(config["seed"]))
    data_config = config["data"]
    training = config["training"]
    print(
        f"[eval] stage=deterministic split={args.split} checkpoint={args.checkpoint}",
        flush=True,
    )
    print(
        f"[data] loading split={args.split} cache={data_config['cache_dir']} "
        f"preload_weather={data_config['preload_weather']}",
        flush=True,
    )
    dataset = ForecastDataset(
        data_config["cache_dir"],
        args.split,
        preload_weather=data_config["preload_weather"],
        preload_weather_max_gib=float(data_config["preload_weather_max_gib"]),
        include_weather=forecast_model_uses_weather(config["model"]),
        response_state_cache_dir=data_config.get("response_state_cache_dir"),
        response_state_donor_mode=(
            "strong_sample"
            if args.response_state_intervention == "strong_sample"
            else data_config.get("response_state_donor_mode", "none")
        ),
    )
    permutation_metadata = None
    if args.weather_permutation_seed is not None:
        permutation_metadata = apply_stratified_weather_permutation(
            dataset,
            seed=args.weather_permutation_seed,
            power_bins=args.weather_permutation_power_bins,
        )
        print(f"[weather-permutation] {permutation_metadata}", flush=True)
    loader = create_forecast_loader(
        dataset,
        cycles_per_batch=2 * int(training["cycles_per_batch"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
    )
    weather_node_indices: tuple[int, ...] | None = args.weather_node_indices
    if args.drop_weather_node_index is not None:
        node_count = int(dataset.weather.values.shape[1])
        dropped = int(args.drop_weather_node_index)
        if not 0 <= dropped < node_count:
            raise ValueError(
                f"drop-weather-node-index must be between 0 and {node_count - 1}"
            )
        weather_node_indices = tuple(index for index in range(node_count) if index != dropped)
    if weather_node_indices is not None:
        loader = WeatherNodeSubsetLoader(loader, weather_node_indices)
        print(f"[weather-nodes] retained={list(weather_node_indices)}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("data_manifest_identity") != dataset.manifest["preprocess_fingerprint"]:
        raise ValueError("Checkpoint was trained with a different data cache")
    state_identity = (
        None
        if dataset.response_state_manifest is None
        else dataset.response_state_manifest.get("preprocess_fingerprint")
    )
    if checkpoint.get("response_state_cache_identity") != state_identity:
        raise ValueError("Checkpoint was trained with a different response-state sidecar")
    model = build_forecast_model(
        checkpoint["model_config"],
        dataset.static,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    )
    if args.model_state_key not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain requested state: {args.model_state_key}"
        )
    model.load_state_dict(checkpoint[args.model_state_key])
    if args.local_state_intervention:
        setter = getattr(model, "set_state_intervention", None)
        if not callable(setter):
            raise ValueError("Local-state intervention requires a compatible model")
        setter(args.local_state_intervention)
    if args.response_state_intervention in {"station_permutation", "reference", "zero"}:
        setter = getattr(model, "set_response_state_intervention", None)
        if not callable(setter):
            raise ValueError("Response-state intervention requires a compatible model")
        setter(args.response_state_intervention)
    if args.zero_horizon_gate or args.unit_horizon_gate:
        gate_logits = getattr(model, "horizon_gate_logits", None)
        if gate_logits is None:
            raise ValueError("Horizon-gate ablation requires a horizon-adaptive model")
        with torch.no_grad():
            gate_logits.fill_(-100.0 if args.zero_horizon_gate else 0.0)
    device = resolve_device(args.device)
    model.to(device)
    settings = {
        **training["deterministic"],
        "gradient_clip": training["gradient_clip"],
    }
    for key in ("station_loss_indices", "metric_station_indices"):
        if key in training:
            settings[key] = training[key]
    if args.metric_station_indices is not None:
        settings["metric_station_indices"] = list(args.metric_station_indices)
        # Keep the displayed evaluation loss on the same subset. This does not
        # affect the frozen checkpoint or any model output.
        settings["station_loss_indices"] = list(args.metric_station_indices)
    settings["distillation_weight"] = 0.0
    if weather_node_indices is not None:
        # The hourly auxiliary target still has all native nodes.  Robustness
        # is defined only on the downstream station/cluster predictions.
        settings["auxiliary_weather_loss_weight"] = 0.0
    prediction_batches = [] if args.predictions_output else None
    metrics = evaluate_deterministic(
        model,
        loader,
        device,
        str(training["precision"]),
        settings,
        max_batches=args.max_batches,
        progress_desc=f"Forecast {args.split}",
        prediction_batches=prediction_batches,
        export_station_predictions=args.export_station_predictions,
    )
    metrics["parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    metrics["model_state_key"] = args.model_state_key
    if permutation_metadata is not None:
        metrics["weather_permutation"] = permutation_metadata
    if weather_node_indices is not None:
        metrics["weather_node_perturbation"] = {
            "retained_indices": list(weather_node_indices),
            "dropped_index": args.drop_weather_node_index,
        }
    if args.zero_horizon_gate:
        metrics["horizon_gate_ablation"] = "zero"
    elif args.unit_horizon_gate:
        metrics["horizon_gate_ablation"] = "one"
    if args.local_state_intervention:
        metrics["local_state_intervention"] = args.local_state_intervention
    if args.response_state_intervention:
        metrics["response_state_intervention"] = args.response_state_intervention
    output = (
        Path(args.output).resolve()
        if args.output
        else Path(args.checkpoint).resolve().parent / f"metrics_{args.split}.json"
    )
    save_json(output, metrics)
    if args.predictions_output:
        assert prediction_batches is not None
        save_prediction_archive(args.predictions_output, prediction_batches)
    dataset.close()
    print(f"[{args.split}] {metrics}")
    print(f"[done] Metrics: {output}")


if __name__ == "__main__":
    main()
