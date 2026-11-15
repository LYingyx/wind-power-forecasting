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

from wind_forcing.data import AggregateWeatherDynamicsDataset
from wind_forcing.models import build_aggregate_model
from wind_forcing.models.aggregate_cluster import linear_interpolate
from wind_forcing.training import resolve_device


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_model(checkpoint_path: Path, dataset, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_config", {}).get("weather_dynamics_mode") != "weather_envelope":
        raise ValueError(f"Not a weather-envelope checkpoint: {checkpoint_path}")
    model = build_aggregate_model(
        checkpoint["model_config"], dataset.static,
        total_capacity_mw=dataset.total_capacity_mw,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache causal past envelope realization factors")
    parser.add_argument("--aggregate-cache", required=True)
    parser.add_argument("--response-cache", required=True)
    parser.add_argument("--response-latent-cache", required=True)
    parser.add_argument("--dynamics-time-cache", required=True)
    parser.add_argument("--dynamics-latent-cache", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--fold-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--maximum", type=float, default=1.2)
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.fold_manifest):
        raise ValueError("Provide exactly one of --checkpoint and --fold-manifest")
    dataset = AggregateWeatherDynamicsDataset(
        args.aggregate_cache, args.response_cache, args.response_latent_cache,
        args.dynamics_time_cache, args.dynamics_latent_cache, args.split, verbose=False,
    )
    origins = np.asarray(dataset.arrays["origin_unix"], dtype=np.int64)
    if args.fold_manifest:
        fold_payload = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
        specifications = [
            (int(item["start_unix"]), int(item["stop_unix"]), Path(item["checkpoint"]).resolve())
            for item in fold_payload["folds"]
        ]
    else:
        specifications = [(int(origins.min()), int(origins.max()), Path(args.checkpoint).resolve())]
    assignment = np.full(len(dataset.base), -1, dtype=np.int64)
    for fold, (start, stop, _) in enumerate(specifications):
        selected = (origins >= start) & (origins <= stop)
        if np.any(assignment[selected] >= 0):
            raise ValueError("Envelope fold ranges overlap")
        assignment[selected] = fold
    if np.any(assignment < 0):
        raise ValueError("Envelope fold ranges do not cover the split")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "past_realization_factor.npy"
    target = open_memmap(target_path, mode="w+", dtype=np.float32, shape=(len(dataset.base), 12))
    device = resolve_device(args.device)
    history_mw = np.asarray(dataset.arrays["cluster_history_mw"], dtype=np.float32)
    past_times_all = dataset.arrays["past_weather_time_offset_hours"]
    cycles_all = np.asarray(dataset.arrays["past_weather_cycle_index"], dtype=np.int64)
    target_hours_template = torch.arange(-11, 1, dtype=torch.float32) / 4.0
    checkpoint_records = []
    for fold, (_, _, checkpoint_path) in enumerate(specifications):
        model, checkpoint = load_model(checkpoint_path, dataset, device)
        indices = np.flatnonzero(assignment == fold)
        checkpoint_records.append({"path": str(checkpoint_path), "sha256": digest(checkpoint_path)})
        for begin in tqdm(range(0, len(indices), args.chunk), desc=f"realization {args.split} fold{fold}"):
            chosen = indices[begin : begin + args.chunk]
            cycles = cycles_all[chosen]
            past = torch.from_numpy(np.ascontiguousarray(
                np.asarray(dataset.base.past_hourly_latent[cycles], dtype=np.float32)
            )).to(device)
            current = torch.from_numpy(np.ascontiguousarray(
                np.asarray(dataset.current_weather_latent[chosen], dtype=np.float32)
            )).to(device)
            past_times = torch.from_numpy(np.ascontiguousarray(
                np.asarray(past_times_all[chosen], dtype=np.float32)
            )).to(device)
            node_values = torch.cat((past, current[:, None]), dim=1)
            node_times = torch.cat((past_times, torch.zeros(len(chosen), 1, device=device)), dim=1)
            targets = target_hours_template.to(device).reshape(1, 12).expand(len(chosen), -1)
            with torch.no_grad():
                values = linear_interpolate(node_values, node_times, targets)
                values = values + model.weather_encoder.budget_adapter(values)
                envelope = torch.sigmoid(
                    model.absolute_head(model.absolute_mapper(values).transpose(1, 2)).squeeze(1)
                )
            power = torch.from_numpy(
                np.ascontiguousarray(history_mw[chosen, -12:] / dataset.total_capacity_mw)
            ).to(device)
            eta = (power / (envelope + float(args.epsilon))).clamp(0.0, float(args.maximum))
            target[chosen] = eta.float().cpu().numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    target.flush()
    del target
    payload = {
        "schema_version": 1,
        "split": args.split,
        "shape": [len(dataset.base), 12],
        "dtype": "float32",
        "epsilon": float(args.epsilon),
        "maximum": float(args.maximum),
        "out_of_fold": bool(args.fold_manifest),
        "envelope_checkpoints": checkpoint_records,
        "aggregate_fingerprint": dataset.manifest["preprocess_fingerprint"],
        "dynamics_latent_fingerprint": dataset.dynamics_latent_manifest["preprocess_fingerprint"],
        "array_sha256": digest(target_path),
    }
    payload["preprocess_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    temporary = output / f".manifest.json.{os.getpid()}"
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    (output / "COMPLETE").touch()
    dataset.close()
    print(f"[realization-cache] complete {output}", flush=True)


if __name__ == "__main__":
    main()
