from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.flop_counter import FlopCounterMode
from tqdm import trange

from wind_forcing.aggregate_training import aggregate_inputs
from wind_forcing.config import load_config
from wind_forcing.data.aggregate import AggregateForecastCollator, AggregateForecastDataset
from wind_forcing.models import build_aggregate_model
from wind_forcing.training import autocast_context, move_batch, resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one aggregate checkpoint on one origin")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    config = load_config(run / "config.yaml")
    seed_everything(int(config["seed"]))
    device = resolve_device(args.device)
    architecture = str(config["model"]["architecture"])
    data = config["data"]
    dataset = AggregateForecastDataset(
        data["cache_dir"], "test", include_weather=not architecture.startswith("hb_"),
        preload_weather=data.get("preload_weather", "auto"),
        preload_weather_max_gib=float(data.get("preload_weather_max_gib", 16.0)),
    )
    batch = move_batch(AggregateForecastCollator(dataset)([0]), device)
    checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
    model = build_aggregate_model(
        checkpoint["model_config"], dataset.static,
        total_capacity_mw=dataset.total_capacity_mw,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    precision = str(config["training"].get("precision", "bf16"))
    prefix = os.environ.get("WIND_PROGRESS_PREFIX", "").strip()
    position_text = os.environ.get("WIND_PROGRESS_POSITION", "").strip()
    position = int(position_text) if position_text else None
    with torch.no_grad():
        for _ in trange(
            args.warmup, desc=f"{prefix} aggregate benchmark warmup".strip(),
            unit="forward", dynamic_ncols=True, position=position,
        ):
            with autocast_context(device, precision):
                model(**aggregate_inputs(batch))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in trange(
            args.iterations, desc=f"{prefix} aggregate benchmark".strip(),
            unit="forward", dynamic_ncols=True, position=position,
        ):
            with autocast_context(device, precision):
                model(**aggregate_inputs(batch))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    # MultiheadAttention's flop hooks require autograd metadata even though no
    # backward pass is executed.
    with torch.enable_grad(), FlopCounterMode(display=False) as flop_counter:
        with autocast_context(device, precision):
            profiled_output = model(**aggregate_inputs(batch))
    profiled_flops = int(flop_counter.get_total_flops())
    del profiled_output
    result = {
        "run_dir": str(run),
        "architecture": architecture,
        "batch_origins": 1,
        "milliseconds_per_origin": 1000.0 * elapsed / args.iterations,
        "origins_per_second": args.iterations / elapsed,
        "peak_allocated_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda" else None
        ),
        "profiled_flops_per_origin": profiled_flops,
        "profiled_gflops_per_origin": profiled_flops / 1e9,
        "parameter_count": int(checkpoint.get("parameter_count", sum(p.numel() for p in model.parameters()))),
        "trainable_parameter_count": int(
            checkpoint.get("trainable_parameter_count", sum(p.numel() for p in model.parameters() if p.requires_grad))
        ),
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "note": "Single-origin forward only; excludes data loading. FLOPs are PyTorch-profiler counted operators.",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    dataset.close()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
