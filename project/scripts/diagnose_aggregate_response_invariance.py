from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wind_forcing.aggregate_training import aggregate_inputs, save_json
from wind_forcing.config import load_config
from wind_forcing.data import AggregateResponseForecastDataset, create_aggregate_response_loader
from wind_forcing.models import build_aggregate_model
from wind_forcing.training import autocast_context, move_batch, resolve_device, seed_everything


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that response state cannot create correction when future weather is zero"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    data = config["data"]
    latent = Path(data["response_latent_cache_root"]) / f"seed{seed}"
    dataset = AggregateResponseForecastDataset(
        data["cache_dir"], data["response_cache_dir"], latent, args.split
    )
    loader = create_aggregate_response_loader(
        dataset, cycles_per_batch=2, shuffle=False, seed=seed
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_aggregate_model(
        checkpoint["model_config"], dataset.static,
        total_capacity_mw=dataset.total_capacity_mw,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()
    maximum = 0.0
    control_sum = 0.0
    control_sq_sum = 0.0
    control_count = 0
    samples = 0
    precision = str(config["training"].get("precision", "bf16"))
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        batch = move_batch(batch, device)
        inputs = aggregate_inputs(batch)
        with autocast_context(device, precision):
            weather = model.weather_encoder.forward_from_hourly(
                inputs["future_hourly_weather_latent"],
                inputs["weather_batch_index"],
                inputs["weather_time_offset_hours"],
            )["weather_features"]
            assert weather is not None
            zero = torch.zeros_like(weather)
            native = model(**inputs, future_weather_latent_override=zero)
            shifted = model(
                **inputs,
                response_past_weather_override=torch.roll(
                    inputs["past_hourly_weather_latent"], shifts=1, dims=0
                ),
                response_past_power_override=torch.roll(
                    inputs["past_cluster_power"], shifts=1, dims=0
                ),
                future_weather_latent_override=zero,
            )
        difference = (
            native["weather_correction_factor"] - shifted["weather_correction_factor"]
        ).abs()
        maximum = max(maximum, float(difference.max()))
        control = native["response_control"].float()
        control_sum += float(control.sum())
        control_sq_sum += float(control.square().sum())
        control_count += control.numel()
        samples += len(batch["cluster_history"])
    mean = control_sum / max(control_count, 1)
    variance = max(0.0, control_sq_sum / max(control_count, 1) - mean * mean)
    result = {
        "split": args.split,
        "samples": samples,
        "response_state_input": checkpoint["model_config"]["response_state_input"],
        "zero_weather_state_invariance_max_abs_factor": maximum,
        "zero_weather_state_invariance_pass": maximum <= 1e-6,
        "response_control_mean": mean,
        "response_control_std": variance**0.5,
    }
    save_json(args.output, result)
    dataset.close()
    if not result["zero_weather_state_invariance_pass"]:
        raise RuntimeError(f"Response zero-weather invariant failed: {maximum}")
    print(f"[response-invariance] {result}", flush=True)


if __name__ == "__main__":
    main()
