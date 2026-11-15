from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
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


def valid(output: Path, checkpoint: Path, time_fingerprint: str) -> bool:
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        values = np.load(output / "current_weather_latent.npy", mmap_mode="r", allow_pickle=False)
        return (
            (output / "COMPLETE").is_file()
            and manifest["base_checkpoint_sha256"] == sha256(checkpoint)
            and manifest["time_fingerprint"] == time_fingerprint
            and list(values.shape) == manifest["shape"]
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache R2 latent at aggregate forecast origin")
    parser.add_argument("--aggregate-cache", required=True)
    parser.add_argument("--response-latent-cache", required=True)
    parser.add_argument("--dynamics-time-cache", required=True)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-cycles", type=int, default=64)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    checkpoint_path = Path(args.base_checkpoint).resolve()
    time_root = Path(args.dynamics_time_cache).resolve()
    time_manifest = json.loads((time_root / "manifest.json").read_text(encoding="utf-8"))
    if valid(output, checkpoint_path, time_manifest["preprocess_fingerprint"]):
        print(f"[dynamics-latent] audited-reuse {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETE").unlink(missing_ok=True)

    aggregate = AggregateForecastDataset(args.aggregate_cache, args.split, include_weather=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    model = build_aggregate_model(
        model_config, aggregate.static, total_capacity_mw=aggregate.total_capacity_mw,
        history_steps=int(aggregate.manifest["history_power_steps"]),
        forecast_steps=int(aggregate.manifest["forecast_power_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = resolve_device(args.device)
    model.to(device).eval()
    encoder = model.weather_encoder

    split_root = time_root / args.split
    left_frame = np.load(split_root / "left_frame_index.npy", allow_pickle=False)
    right_frame = np.load(split_root / "right_frame_index.npy", allow_pickle=False)
    fallback = np.load(split_root / "right_cycle_fallback.npy", allow_pickle=False)
    offsets = np.load(split_root / "current_weather_time_offset_hours.npy", allow_pickle=False)
    cycles = np.asarray(aggregate.arrays["weather_cycle_index"], dtype=np.int64)
    response_latent = np.load(
        Path(args.response_latent_cache) / args.split / "past_hourly_latent.npy",
        mmap_mode="r", allow_pickle=False,
    )
    future_latent = np.load(
        Path(args.response_latent_cache) / args.split / "future_hourly_latent.npy",
        mmap_mode="r", allow_pickle=False,
    )
    if len(response_latent) != len(left_frame):
        raise ValueError("Response/current cycle count mismatch")
    d_model = response_latent.shape[-1]
    right_latent = np.empty((len(right_frame), d_model), dtype=np.float32)
    with h5py.File(args.source_h5, "r") as handle:
        store = handle["nwp/values"]
        crop = aggregate.manifest["crop"]
        row = slice(int(crop["row_start"]), int(crop["row_stop"]))
        col = slice(int(crop["column_start"]), int(crop["column_stop"]))
        for start in tqdm(range(0, len(right_frame), args.chunk_cycles), desc="current R2 latent"):
            stop = min(start + args.chunk_cycles, len(right_frame))
            requested = right_frame[start:stop]
            unique, inverse = np.unique(requested, return_inverse=True)
            frames = np.asarray(store[unique], dtype=np.float32)[inverse][:, :, row, col]
            tensor = torch.from_numpy(np.ascontiguousarray(frames[:, None])).to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                encoded = encoder.encode_hourly(tensor)[:, 0]
            right_latent[start:stop] = encoded.float().cpu().numpy()
    # Boundary fallbacks point at the newer issue's first future node.  Reuse
    # its already audited latent to avoid any encoder precision discrepancy.
    right_latent[fallback] = np.asarray(future_latent[fallback, 0], dtype=np.float32)
    left = np.asarray(response_latent[cycles, -1], dtype=np.float32)
    right = right_latent[cycles]
    weight = (-offsets[:, 0] / (offsets[:, 1] - offsets[:, 0])).astype(np.float32)
    if np.any(weight < -1e-6) or np.any(weight > 1.0 + 1e-6):
        raise ValueError("Current-weather interpolation weight lies outside [0,1]")
    current = left + weight[:, None] * (right - left)
    target = open_memmap(
        output / "current_weather_latent.npy", mode="w+", dtype=np.float32, shape=current.shape
    )
    target[:] = current
    target.flush()
    del target
    payload = {
        "schema_version": 1,
        "split": args.split,
        "seed": int(checkpoint["seed"]),
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": sha256(checkpoint_path),
        "time_fingerprint": time_manifest["preprocess_fingerprint"],
        "aggregate_fingerprint": aggregate.manifest["preprocess_fingerprint"],
        "shape": list(current.shape),
        "dtype": "float32",
        "fallback_sample_count": int(fallback[cycles].sum()),
        "interpolation_weight_min": float(weight.min()),
        "interpolation_weight_max": float(weight.max()),
    }
    payload["preprocess_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    temporary = output / f".manifest.json.{os.getpid()}"
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    (output / "COMPLETE").touch()
    if not valid(output, checkpoint_path, time_manifest["preprocess_fingerprint"]):
        raise RuntimeError("Current-weather latent cache post-write audit failed")
    aggregate.close()
    print(f"[dynamics-latent] complete {output}", flush=True)


if __name__ == "__main__":
    main()
