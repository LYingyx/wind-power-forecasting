from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one aggregate experiment directory")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    config_path = root / "config.yaml"
    checkpoint_path = root / "best_model.pt"
    metrics_path = root / f"metrics_{args.split}.json"
    prediction_path = root / f"predictions_{args.split}.npz"
    for path in (config_path, checkpoint_path, metrics_path, prediction_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("seed") != int(config["seed"]):
        raise ValueError("Checkpoint seed differs from config")
    if config["model"].get("architecture") == "aggregate_response_state":
        response_manifest_path = (
            Path(config["data"]["response_cache_dir"]).resolve() / "manifest.json"
        )
        latent_manifest_path = (
            Path(config["data"]["response_latent_cache_root"]).resolve()
            / f"seed{int(config['seed'])}" / args.split / "manifest.json"
        )
        with response_manifest_path.open(encoding="utf-8") as stream:
            response_manifest = json.load(stream)
        with latent_manifest_path.open(encoding="utf-8") as stream:
            latent_manifest = json.load(stream)
        if (
            checkpoint.get("response_manifest_identity")
            != response_manifest.get("preprocess_fingerprint")
        ):
            raise ValueError("Checkpoint and response cache provenance differ")
        if (
            checkpoint.get("response_latent_identity")
            != latent_manifest.get("base_checkpoint_sha256")
        ):
            raise ValueError("Checkpoint and response latent provenance differ")
    if config["model"].get("architecture") == "aggregate_weather_dynamics":
        seed = int(config["seed"])
        time_manifest_path = Path(config["data"]["dynamics_time_cache_dir"]).resolve() / "manifest.json"
        current_manifest_path = (
            Path(config["data"]["dynamics_latent_cache_root"]).resolve()
            / f"seed{seed}" / args.split / "manifest.json"
        )
        with time_manifest_path.open(encoding="utf-8") as stream:
            time_manifest = json.load(stream)
        with current_manifest_path.open(encoding="utf-8") as stream:
            current_manifest = json.load(stream)
        if checkpoint.get("dynamics_time_identity") != time_manifest.get("preprocess_fingerprint"):
            raise ValueError("Checkpoint and dynamics-time provenance differ")
        if checkpoint.get("dynamics_latent_identity") != current_manifest.get("base_checkpoint_sha256"):
            raise ValueError("Checkpoint and current-weather latent provenance differ")
    with metrics_path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    for key in ("cluster_mae_mw", "cluster_rmse_mw", "cluster_r2", "samples"):
        if key not in metrics or not np.isfinite(float(metrics[key])):
            raise ValueError(f"Invalid metric: {key}")
    expected = 32550 if args.split == "val" else 28782
    if not args.allow_partial and int(metrics["samples"]) != expected:
        raise ValueError(f"Expected {expected} samples, found {metrics['samples']}")
    with np.load(prediction_path, allow_pickle=False) as archive:
        shape = archive["cluster_prediction_mw"].shape
        if shape[1:] != (24,):
            raise ValueError(f"Prediction shape is invalid: {shape}")
        if not args.allow_partial and shape != (expected, 24):
            raise ValueError(f"Prediction archive is incomplete: {shape}")
        if not np.isfinite(archive["cluster_prediction_mw"]).all():
            raise ValueError("Predictions contain non-finite values")
        if not np.array_equal(
            np.sort(archive["sample_index"]),
            np.sort(np.unique(archive["sample_index"])),
        ):
            raise ValueError("Prediction sample indices are not unique")
    print(f"[aggregate-audit] ok split={args.split} run={root}")


if __name__ == "__main__":
    main()
