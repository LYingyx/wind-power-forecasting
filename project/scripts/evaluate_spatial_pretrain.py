from __future__ import annotations

import argparse
from pathlib import Path

import torch

from wind_forcing.config import load_config
from wind_forcing.data import SpatialPretrainDataset, create_spatial_pretrain_loader
from wind_forcing.models import build_spatial_pretrain_model
from wind_forcing.training import (
    evaluate_spatial,
    resolve_device,
    save_json,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one spatial-forcing checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    data = config["data"]
    training = config["training"]
    print(
        f"[eval] stage=spatial split={args.split} checkpoint={args.checkpoint}",
        flush=True,
    )
    print(
        f"[data] loading split={args.split} cache={data['cache_dir']} "
        f"preload_weather={data['preload_weather']}",
        flush=True,
    )
    dataset = SpatialPretrainDataset(
        data["cache_dir"],
        args.split,
        preload_weather=data["preload_weather"],
        preload_weather_max_gib=float(data["preload_weather_max_gib"]),
    )
    settings = {
        **training["spatial_pretrain"],
        "gradient_clip": training["gradient_clip"],
    }
    loader = create_spatial_pretrain_loader(
        dataset,
        batch_size=2 * int(settings["batch_size_cycles"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("data_manifest_identity") != dataset.manifest["preprocess_fingerprint"]:
        raise ValueError("Checkpoint was trained with a different data cache")
    model = build_spatial_pretrain_model(checkpoint["model_config"], dataset.static)
    model.load_state_dict(checkpoint["model_state"])
    device = resolve_device(args.device)
    model.to(device)
    metrics = evaluate_spatial(
        model,
        loader,
        device,
        str(training["precision"]),
        settings,
        args.max_batches,
    )
    save_json(args.output, metrics)
    dataset.close()
    print(f"[{args.split}] {metrics}")
    print(f"[done] Metrics: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
