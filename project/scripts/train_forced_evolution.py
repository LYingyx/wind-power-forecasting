from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wind_forcing.config import load_config
from wind_forcing.data import ForecastDataset, create_forecast_loader
from wind_forcing.models import build_forecast_model, forecast_model_uses_weather
from wind_forcing.training import (
    fit_deterministic,
    resolve_device,
    save_json,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the deterministic forced-evolution model")
    parser.add_argument("--config", default="configs/forced_evolution_pad5.yaml")
    parser.add_argument("--spatial-checkpoint")
    parser.add_argument("--history-checkpoint")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument(
        "--initial-checkpoint-allow-missing-prefix",
        action="append",
        default=[],
        help=(
            "Allow only newly introduced model-state keys with this prefix to "
            "be absent from --initial-checkpoint; may be repeated"
        ),
    )
    parser.add_argument("--teacher-archive")
    parser.add_argument(
        "--spatial-load-mode",
        choices=("complete", "spatial_only"),
        default="spatial_only",
    )
    parser.add_argument("--output-dir", default="artifacts/forced_evolution_pad5")
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
        **training["deterministic"],
        "gradient_clip": training["gradient_clip"],
        "early_stopping_patience": training["early_stopping_patience"],
    }
    for key in ("station_loss_indices", "metric_station_indices"):
        if key in training:
            settings[key] = training[key]
    output = Path(args.output_dir).resolve()
    model_config = config["model"]
    architecture = str(model_config.get("architecture", "forced_evolution"))
    query_mode = str(model_config.get("station_query_mode", "n/a"))
    cells = int(model_config.get("station_neighborhood_cells", 1))
    native_points = (
        1
        if query_mode == "nearest"
        else 4
        if query_mode == "bilinear"
        else (cells + 1) ** 2
    )
    epoch_count = int(settings["epochs"] if args.epochs is None else args.epochs)
    print(
        f"[run] stage=deterministic "
        f"experiment={config.get('experiment', {}).get('id', 'manual')} "
        f"seed={seed} device={args.device} precision={training['precision']}",
        flush=True,
    )
    print(
        f"[model] architecture={architecture} query={query_mode} cells={cells} "
        f"native_points={native_points} "
        f"regional_pool={model_config.get('regional_pool_size', 1)} "
        f"regional_cells={model_config.get('regional_neighborhood_cells', 1)} "
        f"wind_geometry={model_config.get('include_wind_geometry', False)} "
        f"region_context={model_config.get('include_region_context', False)} "
        f"forcing_path={model_config.get('forcing_path_mode', 'n/a')} "
        f"time_query={model_config.get('time_query_mode', 'n/a')} "
        f"fusion={model_config.get('fusion_mode', 'n/a')} "
        f"transition={model_config.get('transition_type', 'n/a')} "
        f"multirate={model_config.get('multirate_enabled', False)} "
        f"ramp_weight={settings.get('ramp_loss_weight', 0.0)} "
        f"output={model_config.get('output_level', 'hierarchical')}",
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
        "include_weather": forecast_model_uses_weather(model_config),
        "response_state_cache_dir": data_config.get("response_state_cache_dir"),
        "response_state_donor_mode": data_config.get("response_state_donor_mode", "none"),
    }
    train = ForecastDataset(data_config["cache_dir"], "train", **dataset_kwargs)
    val = ForecastDataset(data_config["cache_dir"], "val", **dataset_kwargs)
    distillation_weight = float(settings.get("distillation_weight", 0.0))
    teacher_metadata = None
    if distillation_weight > 0.0:
        if not args.teacher_archive:
            raise ValueError("--teacher-archive is required for distillation")
        teacher_path = Path(args.teacher_archive).resolve()
        train.attach_teacher_archive(teacher_path)
        with np.load(teacher_path, allow_pickle=False) as teacher_archive:
            teacher_metadata = {
                "archive": str(teacher_path),
                "seeds": (
                    teacher_archive["teacher_seeds"].astype(np.int64).tolist()
                    if "teacher_seeds" in teacher_archive.files
                    else None
                ),
                "kind": (
                    str(teacher_archive["teacher_kind"].item())
                    if "teacher_kind" in teacher_archive.files
                    else "legacy_unspecified"
                ),
                "model_state_key": (
                    str(teacher_archive["model_state_key"].item())
                    if "model_state_key" in teacher_archive.files
                    else None
                ),
            }
        print(
            f"[distillation] Attached teacher archive: {teacher_path}; "
            f"kind={teacher_metadata['kind']} seeds={teacher_metadata['seeds']}"
        )
        if bool(settings.get("distillation_uncertainty_aware", False)):
            for output_name, array_name in (
                ("distillation_station_scale", "teacher_station_factor_std"),
                ("distillation_cluster_scale", "teacher_cluster_factor_std"),
            ):
                configured = float(settings.get(output_name, 0.0))
                if configured <= 0.0:
                    positive = train.teacher_arrays[array_name]
                    positive = positive[positive > 0.0]
                    if not len(positive):
                        raise ValueError(
                            f"Teacher dispersion has no positive values: {array_name}"
                        )
                    settings[output_name] = float(np.median(positive))
            print(
                "[distillation] uncertainty scales "
                f"station={settings['distillation_station_scale']:.6g} "
                f"cluster={settings['distillation_cluster_scale']:.6g}"
            )
    elif args.teacher_archive:
        raise ValueError("--teacher-archive was supplied with zero distillation weight")
    cycles_per_batch = int(training["cycles_per_batch"])
    train_loader = create_forecast_loader(
        train,
        cycles_per_batch=cycles_per_batch,
        shuffle=True,
        seed=seed,
        num_workers=int(training["num_workers"]),
    )
    val_loader = create_forecast_loader(
        val,
        cycles_per_batch=2 * cycles_per_batch,
        shuffle=False,
        seed=seed,
        num_workers=int(training["num_workers"]),
    )
    model = build_forecast_model(
        config["model"],
        train.static,
        history_steps=int(train.manifest["history_power_steps"]),
        forecast_steps=int(train.manifest["forecast_power_steps"]),
    )
    if args.initial_checkpoint:
        initial_checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False
        )
        if initial_checkpoint.get("data_manifest_identity") != train.manifest["preprocess_fingerprint"]:
            raise ValueError("Initial checkpoint was trained with a different data cache")
        allowed_prefixes = tuple(args.initial_checkpoint_allow_missing_prefix)
        if allowed_prefixes:
            result = model.load_state_dict(initial_checkpoint["model_state"], strict=False)
            invalid_missing = [
                key for key in result.missing_keys
                if not any(key.startswith(prefix) for prefix in allowed_prefixes)
            ]
            if invalid_missing or result.unexpected_keys:
                raise ValueError(
                    "Incompatible initial checkpoint: "
                    f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
                )
            print(
                f"[init] Loaded compatible model state: {args.initial_checkpoint}; "
                f"new_keys={result.missing_keys}"
            )
        else:
            model.load_state_dict(initial_checkpoint["model_state"], strict=True)
            print(f"[init] Loaded complete model state: {args.initial_checkpoint}")
    if architecture == "incremental_correction":
        initialize_from_checkpoint = bool(
            model_config.get("initialize_history_baseline_from_checkpoint", True)
        )
        if initialize_from_checkpoint:
            if not args.history_checkpoint:
                raise ValueError(
                    "--history-checkpoint is required when pretrained baseline "
                    "initialization is enabled"
                )
            history_checkpoint = torch.load(
                args.history_checkpoint, map_location="cpu", weights_only=False
            )
            if history_checkpoint.get("data_manifest_identity") != train.manifest["preprocess_fingerprint"]:
                raise ValueError("History checkpoint was trained with a different data cache")
            model.load_history_checkpoint(history_checkpoint)
            print(f"[init] Loaded history baseline: {args.history_checkpoint}")
        else:
            if bool(model_config.get("baseline_anchor_enabled", False)):
                raise ValueError(
                    "baseline_anchor_enabled requires pretrained baseline initialization"
                )
            print("[init] History baseline uses the seeded random initialization")
    if args.spatial_checkpoint:
        if not hasattr(model, "forcing_encoder"):
            raise ValueError("Spatial checkpoints apply only to forced-evolution models")
        checkpoint = torch.load(args.spatial_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("data_manifest_identity") != train.manifest["preprocess_fingerprint"]:
            raise ValueError("Spatial checkpoint was trained with a different data cache")
        forcing_state = checkpoint["forcing_encoder_state"]
        if args.spatial_load_mode == "complete":
            model.forcing_encoder.load_state_dict(forcing_state)
            skipped: list[str] = []
        else:
            current = model.forcing_encoder.state_dict()
            compatible = {
                key: value
                for key, value in forcing_state.items()
                if not key.startswith("path_encoder.")
                and key in current
                and current[key].shape == value.shape
            }
            result = model.forcing_encoder.load_state_dict(compatible, strict=False)
            skipped = sorted(
                set(forcing_state).difference(compatible)
                | set(result.unexpected_keys)
            )
        print(
            f"[init] Loaded spatial forcing encoder: {args.spatial_checkpoint}; "
            f"mode={args.spatial_load_mode}, skipped={len(skipped)}"
        )
    else:
        print("[init] No spatial checkpoint supplied; training all modules from scratch")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"[model] parameters={total_parameters} trainable={trainable_parameters}",
        flush=True,
    )

    print(
        f"[data] train_samples={len(train)} train_batches={len(train_loader)} "
        f"val_samples={len(val)} val_batches={len(val_loader)}",
        flush=True,
    )
    print(
        f"[train] epochs={epoch_count} patience={settings['early_stopping_patience']} "
        f"cycles_per_batch={cycles_per_batch} output={output}",
        flush=True,
    )
    history = fit_deterministic(
        model,
        train_loader,
        val_loader,
        settings,
        resolve_device(args.device),
        output / "best_model.pt",
        {
            "stage": "deterministic",
            "model_config": config["model"],
            "data_manifest_identity": train.manifest["preprocess_fingerprint"],
            "response_state_cache_identity": (
                None
                if train.response_state_manifest is None
                else train.response_state_manifest.get("preprocess_fingerprint")
            ),
            "seed": seed,
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "distillation_settings": {
                key: settings.get(key)
                for key in (
                    "distillation_weight",
                    "distillation_uncertainty_aware",
                    "distillation_station_scale",
                    "distillation_cluster_scale",
                )
            },
            "distillation_teacher": teacher_metadata,
            "loss_mode": settings.get("loss_mode", "smooth_l1"),
        },
        precision=str(training["precision"]),
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    save_json(output / "history.json", history)
    train.close()
    val.close()
    print(f"[done] Deterministic checkpoint: {output / 'best_model.pt'}")


if __name__ == "__main__":
    main()
