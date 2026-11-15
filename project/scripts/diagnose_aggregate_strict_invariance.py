from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wind_forcing.aggregate_training import aggregate_inputs, save_json
from wind_forcing.config import load_config
from wind_forcing.data import AggregateForecastDataset, create_aggregate_loader
from wind_forcing.models import build_aggregate_model
from wind_forcing.training import autocast_context, move_batch, resolve_device, seed_everything


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict-modulation zero-weather invariance")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    data = config["data"]
    dataset = AggregateForecastDataset(
        data["cache_dir"], args.split, include_weather=True,
        preload_weather=data.get("preload_weather", "auto"),
        preload_weather_max_gib=float(data.get("preload_weather_max_gib", 16.0)),
    )
    loader = create_aggregate_loader(dataset, cycles_per_batch=2, shuffle=False)
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
    routing_sum = None
    routing_entropy_sum = 0.0
    routing_count = 0
    samples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        batch = move_batch(batch, device)
        inputs = aggregate_inputs(batch)
        with autocast_context(device, str(config["training"].get("precision", "bf16"))):
            weather_output = model.weather_encoder(
                inputs["weather"], inputs["weather_batch_index"], inputs["weather_time_offset_hours"]
            )
            zero = torch.zeros_like(weather_output["weather_features"])
            native = model(**inputs, weather_latent_override=zero)
            shifted = model(
                **inputs,
                correction_state_history=torch.roll(batch["cluster_history"], shifts=1, dims=0),
                weather_latent_override=zero,
            )
            difference = (
                native["weather_correction_factor"] - shifted["weather_correction_factor"]
            ).abs()
            maximum = max(maximum, float(difference.max()))
            control = native["modulation_control"].float()
            control_sum += float(control.sum())
            control_sq_sum += float(control.square().sum())
            control_count += control.numel()
            if model.modulation_mode == "experts":
                routing = torch.softmax(control / model.modulation_amplitude.to(control), dim=-1)
                occupancy = routing.sum(dim=(0, 1)).cpu().double()
                routing_sum = occupancy if routing_sum is None else routing_sum + occupancy
                routing_entropy_sum += float(
                    (-(routing * routing.clamp_min(1e-12).log()).sum(dim=-1)).sum()
                )
                routing_count += routing.shape[0] * routing.shape[1]
            samples += len(batch["cluster_history"])
    mean = control_sum / max(control_count, 1)
    variance = max(0.0, control_sq_sum / max(control_count, 1) - mean * mean)
    result = {
        "split": args.split,
        "samples": samples,
        "zero_weather_state_invariance_max_abs_factor": maximum,
        "zero_weather_state_invariance_pass": maximum <= 1e-6,
        "modulation_control_mean": mean,
        "modulation_control_std": variance ** 0.5,
        "mode": checkpoint["model_config"]["strict_modulation_mode"],
        "state_source": checkpoint["model_config"]["modulation_state_source"],
    }
    if routing_sum is not None:
        result["expert_mean_occupancy"] = (routing_sum / routing_count).tolist()
        result["expert_mean_entropy"] = routing_entropy_sum / routing_count
    save_json(args.output, result)
    dataset.close()
    if not result["zero_weather_state_invariance_pass"]:
        raise RuntimeError(f"Strict zero-weather invariant failed: {maximum}")
    print(f"[strict-invariance] {result}", flush=True)


if __name__ == "__main__":
    main()
