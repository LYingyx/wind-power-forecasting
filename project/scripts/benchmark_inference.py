from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from wind_forcing.config import load_config
from wind_forcing.data import ForecastDataset, create_forecast_loader
from wind_forcing.models import build_forecast_model, forecast_model_uses_weather
from wind_forcing.training import autocast_context, model_inputs, move_batch, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardized forward-pass benchmark on one frozen checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve(); config = load_config(run / "config.yaml")
    seed_everything(int(config["seed"])); device = resolve_device(args.device)
    data = config["data"]; training = config["training"]
    dataset = ForecastDataset(data["cache_dir"], args.split, preload_weather=data["preload_weather"], preload_weather_max_gib=float(data["preload_weather_max_gib"]), include_weather=forecast_model_uses_weather(config["model"]))
    loader = create_forecast_loader(dataset, cycles_per_batch=2 * int(training["cycles_per_batch"]), shuffle=False, num_workers=0)
    batch = move_batch(next(iter(loader)), device); origins = int(batch["station_history"].shape[0])
    checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
    model = build_forecast_model(checkpoint["model_config"], dataset.static, history_steps=int(dataset.manifest["history_power_steps"]), forecast_steps=int(dataset.manifest["forecast_power_steps"]))
    model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
    precision = str(training["precision"])
    prefix = os.environ.get("WIND_PROGRESS_PREFIX", "").strip()
    position_value = os.environ.get("WIND_PROGRESS_POSITION", "").strip()
    position = int(position_value) if position_value else None
    leave = os.environ.get("WIND_PROGRESS_LEAVE", "1").lower() not in {"0", "false", "no"}
    with torch.no_grad():
        for _ in trange(args.warmup, desc=f"{prefix} warmup".strip(), unit="forward", dynamic_ncols=True, position=position, leave=leave, disable=False):
            with autocast_context(device, precision): model(**model_inputs(batch))
        if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in trange(args.iterations, desc=f"{prefix} benchmark".strip(), unit="forward", dynamic_ncols=True, position=position, leave=leave, disable=False):
            with autocast_context(device, precision): model(**model_inputs(batch))
        if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "label": args.label,
        "split": args.split,
        "run_dir": str(run),
        "physical_device_visible_as": str(device),
        "batch_origins": origins,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "milliseconds_per_batch": 1000.0 * elapsed / args.iterations,
        "milliseconds_per_origin": 1000.0 * elapsed / (args.iterations * origins),
        "origins_per_second": args.iterations * origins / elapsed,
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_mib": float((run / "best_model.pt").stat().st_size / 2**20),
        "note": f"Forward-only, one preloaded {args.split} batch; excludes data loading and plotting.",
    }
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    dataset.close(); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
