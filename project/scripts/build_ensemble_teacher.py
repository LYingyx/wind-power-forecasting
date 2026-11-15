from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from wind_forcing.config import load_config
from wind_forcing.data import ForecastDataset, create_forecast_loader
from wind_forcing.models import build_forecast_model, forecast_model_uses_weather
from wind_forcing.training import (
    autocast_context,
    model_inputs,
    move_batch,
    resolve_device,
    seed_everything,
)


def update_welford(
    mean: np.ndarray,
    m2: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
    count: int,
) -> None:
    delta = values - mean[positions]
    mean[positions] += delta / float(count)
    m2[positions] += delta * (values - mean[positions])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aligned train-set mean/std targets from an ensemble"
    )
    parser.add_argument("--group-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-state-key", default="model_state")
    parser.add_argument(
        "--allow-single-member",
        action="store_true",
        help=(
            "Allow a one-checkpoint teacher archive. Its dispersion arrays are "
            "exactly zero; this is intended only for controlled single-teacher KD."
        ),
    )
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one teacher checkpoint is required")
    if len(seeds) < 2 and not args.allow_single_member:
        raise ValueError("An ensemble teacher requires at least two checkpoints")
    group_root = Path(args.group_root).resolve()
    first_config = load_config(group_root / f"seed{seeds[0]}" / "config.yaml")
    seed_everything(seeds[0])
    data_config = first_config["data"]
    dataset = ForecastDataset(
        data_config["cache_dir"],
        "train",
        preload_weather=data_config["preload_weather"],
        preload_weather_max_gib=float(data_config["preload_weather_max_gib"]),
        include_weather=forecast_model_uses_weather(first_config["model"]),
    )
    loader = create_forecast_loader(
        dataset,
        cycles_per_batch=2 * int(first_config["training"]["cycles_per_batch"]),
        shuffle=False,
        num_workers=int(first_config["training"]["num_workers"]),
    )
    sample_ids = np.asarray(dataset.arrays["sample_index"], dtype=np.int64)
    sample_order = np.argsort(sample_ids, kind="stable")
    sorted_ids = sample_ids[sample_order]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise ValueError("Training sample_index values are not unique")
    station_shape = dataset.arrays["station_future"].shape
    cluster_shape = dataset.arrays["cluster_future"].shape
    station_mean = np.zeros(station_shape, dtype=np.float32)
    station_m2 = np.zeros(station_shape, dtype=np.float32)
    cluster_mean = np.zeros(cluster_shape, dtype=np.float32)
    cluster_m2 = np.zeros(cluster_shape, dtype=np.float32)
    device = resolve_device(args.device)

    member_progress = tqdm(seeds, desc="Build ensemble teacher", unit="member")
    for count, seed in enumerate(member_progress, start=1):
        member_progress.set_postfix_str(f"seed={seed}", refresh=False)
        run = group_root / f"seed{seed}"
        config = load_config(run / "config.yaml")
        if config["data"]["cache_dir"] != data_config["cache_dir"]:
            raise ValueError("Teacher checkpoints use different data caches")
        checkpoint = torch.load(
            run / "best_model.pt", map_location="cpu", weights_only=False
        )
        if checkpoint.get("data_manifest_identity") != dataset.manifest["preprocess_fingerprint"]:
            raise ValueError("Teacher checkpoint has a different data fingerprint")
        if args.model_state_key not in checkpoint:
            raise ValueError(
                f"Teacher checkpoint is missing {args.model_state_key}: {run}"
            )
        model = build_forecast_model(
            checkpoint["model_config"],
            dataset.static,
            history_steps=int(dataset.manifest["history_power_steps"]),
            forecast_steps=int(dataset.manifest["forecast_power_steps"]),
        )
        model.load_state_dict(checkpoint[args.model_state_key])
        model.to(device).eval()
        precision = str(config["training"]["precision"])
        seen = 0
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc=f"Teacher seed {seed}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            ):
                global_ids = batch["sample_index"].numpy().astype(np.int64)
                locations = np.searchsorted(sorted_ids, global_ids)
                if np.any(locations >= len(sorted_ids)) or not np.array_equal(
                    sorted_ids[locations], global_ids
                ):
                    raise ValueError("Teacher batch contains unknown sample indices")
                positions = sample_order[locations]
                batch = move_batch(batch, device)
                with autocast_context(device, precision):
                    output = model(**model_inputs(batch))
                station = output.get("station_factor")
                if station is None:
                    raise ValueError("Teacher model does not produce station forecasts")
                update_welford(
                    station_mean,
                    station_m2,
                    positions,
                    station.detach().float().cpu().numpy(),
                    count,
                )
                update_welford(
                    cluster_mean,
                    cluster_m2,
                    positions,
                    output["cluster_factor"].detach().float().cpu().numpy(),
                    count,
                )
                seen += len(positions)
        if seen != len(dataset):
            raise ValueError(f"Teacher seed {seed} produced {seen} predictions")
        member_progress.set_postfix_str(f"completed seed={seed}", refresh=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(seeds) == 1:
        station_std = np.zeros_like(station_mean)
        cluster_std = np.zeros_like(cluster_mean)
    else:
        station_std = np.sqrt(station_m2 / float(len(seeds) - 1), dtype=np.float32)
        cluster_std = np.sqrt(cluster_m2 / float(len(seeds) - 1), dtype=np.float32)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            sample_index=sample_ids,
            origin_unix=np.asarray(dataset.arrays["origin_unix"], dtype=np.int64),
            teacher_station_factor_mean=station_mean,
            teacher_station_factor_std=station_std,
            teacher_cluster_factor_mean=cluster_mean,
            teacher_cluster_factor_std=cluster_std,
            teacher_seeds=np.asarray(seeds, dtype=np.int64),
            teacher_kind=np.asarray(
                "single_checkpoint" if len(seeds) == 1 else "ensemble_mean"
            ),
            data_manifest_identity=np.asarray(
                dataset.manifest["preprocess_fingerprint"]
            ),
            model_state_key=np.asarray(args.model_state_key),
        )
    os.replace(temporary, output)
    dataset.close()
    print(
        f"[done] teacher={output} station_std_median={np.median(station_std):.6g} "
        f"cluster_std_median={np.median(cluster_std):.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
