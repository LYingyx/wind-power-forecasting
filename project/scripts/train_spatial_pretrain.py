from __future__ import annotations

import argparse
from pathlib import Path

from wind_forcing.config import load_config
from wind_forcing.data import SpatialPretrainDataset, create_spatial_pretrain_loader
from wind_forcing.models import build_spatial_pretrain_model
from wind_forcing.training import (
    fit_spatial_pretrain,
    resolve_device,
    save_json,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain grid-weather to station forcing")
    parser.add_argument("--config", default="configs/forced_evolution_pad5.yaml")
    parser.add_argument("--output-dir", default="artifacts/spatial_pretrain_pad5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    data_config = config["data"]
    training = config["training"]
    settings = {
        **training["spatial_pretrain"],
        "gradient_clip": training["gradient_clip"],
        "early_stopping_patience": training["early_stopping_patience"],
    }
    output = Path(args.output_dir).resolve()
    model_config = config["model"]
    query_mode = str(model_config["station_query_mode"])
    cells = int(model_config["station_neighborhood_cells"])
    native_points = (
        1
        if query_mode == "nearest"
        else 4
        if query_mode == "bilinear"
        else (cells + 1) ** 2
    )
    epoch_count = int(settings["epochs"] if args.epochs is None else args.epochs)
    print(
        f"[run] stage=spatial_pretrain "
        f"experiment={config.get('experiment', {}).get('id', 'manual')} "
        f"seed={seed} device={args.device} precision={training['precision']}",
        flush=True,
    )
    print(
        f"[model] query={query_mode} cells={cells} native_points={native_points} "
        f"wind_geometry={model_config['include_wind_geometry']} "
        f"region_context={model_config['include_region_context']} "
        f"forcing_path={model_config['forcing_path_mode']}",
        flush=True,
    )
    print(
        f"[data] loading train/val cache={data_config['cache_dir']} "
        f"preload_weather={data_config['preload_weather']}",
        flush=True,
    )
    dataset_kwargs = {
        "preload_weather": data_config["preload_weather"],
        "preload_weather_max_gib": float(data_config["preload_weather_max_gib"]),
    }
    train = SpatialPretrainDataset(data_config["cache_dir"], "train", **dataset_kwargs)
    val = SpatialPretrainDataset(data_config["cache_dir"], "val", **dataset_kwargs)
    train_loader = create_spatial_pretrain_loader(
        train,
        batch_size=int(settings["batch_size_cycles"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
    )
    val_loader = create_spatial_pretrain_loader(
        val,
        batch_size=2 * int(settings["batch_size_cycles"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
    )
    model = build_spatial_pretrain_model(config["model"], train.static)
    print(
        f"[data] train_cycles={len(train)} train_batches={len(train_loader)} "
        f"val_cycles={len(val)} val_batches={len(val_loader)}",
        flush=True,
    )
    print(
        f"[train] epochs={epoch_count} patience={settings['early_stopping_patience']} "
        f"batch_cycles={settings['batch_size_cycles']} output={output}",
        flush=True,
    )
    history = fit_spatial_pretrain(
        model,
        train_loader,
        val_loader,
        settings,
        resolve_device(args.device),
        output / "best_model.pt",
        {
            "stage": "spatial_pretrain",
            "model_config": config["model"],
            "data_manifest_identity": train.manifest["preprocess_fingerprint"],
            "seed": seed,
        },
        precision=str(training["precision"]),
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    save_json(output / "history.json", history)
    train.close()
    val.close()
    print(f"[done] Spatial checkpoint: {output / 'best_model.pt'}")


if __name__ == "__main__":
    main()
