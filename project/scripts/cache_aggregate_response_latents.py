from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
import torch
from tqdm import tqdm

from wind_forcing.data import AggregateForecastDataset
from wind_forcing.models import build_aggregate_model
from wind_forcing.training import resolve_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid(directory: Path, checkpoint: Path, response_fingerprint: str) -> bool:
    marker = directory / "COMPLETE"
    manifest_path = directory / "manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["base_checkpoint_sha256"] != sha256(checkpoint):
            return False
        if manifest["response_fingerprint"] != response_fingerprint:
            return False
        for name in ("future_hourly_latent", "past_hourly_latent"):
            values = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if list(values.shape) != manifest[f"{name}_shape"] or not np.isfinite(values).all():
                return False
        return True
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen same-seed R2 hourly weather latents")
    parser.add_argument("--aggregate-cache", required=True)
    parser.add_argument("--response-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-cycles", type=int, default=24)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    response_root = Path(args.response_cache).resolve()
    response_manifest = json.loads((response_root / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(checkpoint["seed"])
    output = Path(args.output_root).resolve() / f"seed{seed}" / args.split
    output.mkdir(parents=True, exist_ok=True)
    if valid(output, checkpoint_path, str(response_manifest["preprocess_fingerprint"])):
        print(f"[response-latent-cache] audited-reuse seed={seed} split={args.split}", flush=True)
        return
    (output / "COMPLETE").unlink(missing_ok=True)

    dataset = AggregateForecastDataset(
        args.aggregate_cache,
        args.split,
        include_weather=True,
        preload_weather="mmap",
        verbose=False,
    )
    if checkpoint.get("data_manifest_identity") != dataset.manifest["preprocess_fingerprint"]:
        raise ValueError("R2-W checkpoint and aggregate cache differ")
    model_config = checkpoint["model_config"]
    if not (
        model_config.get("architecture") == "aggregate_correction"
        and model_config.get("weather_route") == "r2_latent_interpolation"
        and model_config.get("correction_conditioning") == "weather"
    ):
        raise ValueError("Latent cache requires a complete R2-W checkpoint")
    model = build_aggregate_model(
        model_config,
        dataset.static,
        total_capacity_mw=dataset.total_capacity_mw,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()
    encoder = model.weather_encoder
    future_weather = dataset.weather
    assert future_weather is not None
    past_weather = np.load(
        response_root / args.split / "past_weather.npy", mmap_mode="r", allow_pickle=False
    )
    if len(future_weather) != len(past_weather):
        raise ValueError("Past and future weather cycle counts differ")
    d_model = int(model_config.get("d_model", 64))
    future_path = output / "future_hourly_latent.npy"
    past_path = output / "past_hourly_latent.npy"
    future = open_memmap(
        future_path, mode="w+", dtype=np.float32,
        shape=(len(future_weather), future_weather.shape[1], d_model),
    )
    past = open_memmap(
        past_path, mode="w+", dtype=np.float32,
        shape=(len(past_weather), past_weather.shape[1], d_model),
    )
    chunk = int(args.chunk_cycles)
    precision = "bf16" if device.type == "cuda" else "fp32"
    for start in tqdm(
        range(0, len(future_weather), chunk),
        desc=f"R2 latent seed{seed} {args.split}",
        unit="chunk",
        dynamic_ncols=True,
    ):
        stop = min(start + chunk, len(future_weather))
        future_tensor = torch.from_numpy(
            np.ascontiguousarray(np.asarray(future_weather[start:stop], dtype=np.float32))
        ).to(device)
        past_tensor = torch.from_numpy(
            np.ascontiguousarray(np.asarray(past_weather[start:stop], dtype=np.float32))
        ).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and precision == "bf16",
        ):
            future_values = encoder.encode_hourly(future_tensor)
            past_values = encoder.encode_hourly(past_tensor)
        future[start:stop] = future_values.float().cpu().numpy()
        past[start:stop] = past_values.float().cpu().numpy()
    future.flush()
    past.flush()
    del future, past
    checkpoint_digest = sha256(checkpoint_path)
    payload = {
        "schema_version": 1,
        "split": args.split,
        "seed": seed,
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": checkpoint_digest,
        "aggregate_fingerprint": dataset.manifest["preprocess_fingerprint"],
        "response_fingerprint": response_manifest["preprocess_fingerprint"],
        "future_hourly_latent_shape": list(
            np.load(future_path, mmap_mode="r", allow_pickle=False).shape
        ),
        "past_hourly_latent_shape": list(
            np.load(past_path, mmap_mode="r", allow_pickle=False).shape
        ),
        "dtype": "float32",
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload["preprocess_fingerprint"] = identity
    temporary = output / f".manifest.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    (output / "COMPLETE").touch()
    if not valid(output, checkpoint_path, str(response_manifest["preprocess_fingerprint"])):
        raise RuntimeError("Response latent cache failed its post-write audit")
    dataset.close()
    print(f"[response-latent-cache] complete seed={seed} split={args.split}", flush=True)


if __name__ == "__main__":
    main()
