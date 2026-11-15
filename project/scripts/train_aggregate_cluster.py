from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wind_forcing.aggregate_training import fit_aggregate, save_json
from wind_forcing.config import load_config
from wind_forcing.data import (
    AggregateForecastDataset,
    AggregateResponseForecastDataset,
    AggregateWeatherDynamicsDataset,
    create_aggregate_loader,
    create_aggregate_response_loader,
    create_aggregate_dynamics_loader,
)
from wind_forcing.models import build_aggregate_model
from wind_forcing.training import resolve_device, seed_everything


def uses_weather(architecture: str) -> bool:
    return not architecture.startswith("hb_")


@torch.no_grad()
def calibrate_strict_state_reference(model, dataset, device: torch.device) -> None:
    """Compute train-only current and frozen-baseline-latent reference states."""
    history_mw = dataset.arrays["cluster_history_mw"]
    current = float(np.asarray(history_mw[:, -1], dtype=np.float64).mean()) / dataset.total_capacity_mw
    model.to(device)
    model.eval()
    total = torch.zeros(model.d_model, dtype=torch.float64, device=device)
    samples = 0
    for start in range(0, len(history_mw), 4096):
        values = np.asarray(history_mw[start : start + 4096], dtype=np.float32)
        history = torch.from_numpy(values / dataset.total_capacity_mw).to(device)
        latent = model.baseline.encode_latent(history).float()
        total += latent.double().sum(dim=0)
        samples += len(latent)
    model.set_state_reference(current, (total / samples).float())
    print(
        f"[aggregate-state-reference] samples={samples} current={current:.6f} "
        f"latent_norm={float(model.latent_state_reference.norm()):.6f}",
        flush=True,
    )


@torch.no_grad()
def calibrate_response_reference(model, dataset, device: torch.device) -> None:
    """Store mean raw response tokens; the trainable encoder is applied online."""
    model.to(device)
    model.eval()
    total = None
    samples = 0
    cycle = dataset.arrays["past_weather_cycle_index"]
    power_mw = dataset.arrays["past_cluster_power_mw"]
    times = dataset.arrays["past_weather_time_offset_hours"]
    for start in range(0, len(dataset), 2048):
        stop = min(start + 2048, len(dataset))
        index = np.asarray(cycle[start:stop], dtype=np.int64)
        past_weather = torch.from_numpy(
            np.ascontiguousarray(np.asarray(dataset.past_hourly_latent[index], dtype=np.float32))
        ).to(device)
        past_power = torch.from_numpy(
            np.ascontiguousarray(np.asarray(power_mw[start:stop], dtype=np.float32))
        ).to(device) / dataset.total_capacity_mw
        past_time = torch.from_numpy(
            np.ascontiguousarray(np.asarray(times[start:stop], dtype=np.float32))
        ).to(device)
        tokens = model.build_response_tokens(past_weather, past_power, past_time).float()
        batch_sum = tokens.double().sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        samples += len(tokens)
    assert total is not None and samples
    model.set_response_reference_tokens((total / samples).float())
    print(
        f"[response-state-reference] samples={samples} "
        f"token_norm={float(model.response_reference_tokens.norm()):.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one aggregate-only forecast model")
    parser.add_argument("--config", default="configs/aggregate_cluster_v1.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-checkpoint")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--auxiliary-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    architecture = str(config["model"]["architecture"])
    data = config["data"]
    response_architecture = architecture == "aggregate_response_state"
    dynamics_architecture = architecture == "aggregate_weather_dynamics"
    if dynamics_architecture:
        latent_cache = Path(data["response_latent_cache_root"]) / f"seed{seed}"
        dynamics_latent = Path(data["dynamics_latent_cache_root"]) / f"seed{seed}"
        realization = data.get("realization_cache_root")
        if realization:
            realization = str(Path(realization) / f"seed{seed}")
        train = AggregateWeatherDynamicsDataset(
            data["cache_dir"], data["response_cache_dir"], latent_cache,
            data["dynamics_time_cache_dir"], dynamics_latent, "train",
            realization_cache=realization,
            exclude_origin_start_unix=data.get("train_exclude_origin_start_unix"),
            exclude_origin_stop_unix=data.get("train_exclude_origin_stop_unix"),
            exclusion_embargo_seconds=int(data.get("train_exclusion_embargo_seconds", 0)),
        )
        val = AggregateWeatherDynamicsDataset(
            data["cache_dir"], data["response_cache_dir"], latent_cache,
            data["dynamics_time_cache_dir"], dynamics_latent, "val",
            realization_cache=realization,
        )
    elif response_architecture:
        latent_cache = Path(data["response_latent_cache_root"]) / f"seed{seed}"
        train = AggregateResponseForecastDataset(
            data["cache_dir"], data["response_cache_dir"], latent_cache, "train"
        )
        val = AggregateResponseForecastDataset(
            data["cache_dir"], data["response_cache_dir"], latent_cache, "val"
        )
    else:
        kwargs = {
            "include_weather": uses_weather(architecture),
            "preload_weather": data.get("preload_weather", "auto"),
            "preload_weather_max_gib": float(data.get("preload_weather_max_gib", 16.0)),
        }
        train = AggregateForecastDataset(data["cache_dir"], "train", **kwargs)
        val = AggregateForecastDataset(data["cache_dir"], "val", **kwargs)
    training = config["training"]
    cycles = int(training["cycles_per_batch"])
    loader_factory = (
        create_aggregate_dynamics_loader if dynamics_architecture
        else create_aggregate_response_loader if response_architecture
        else create_aggregate_loader
    )
    train_loader = loader_factory(
        train, cycles_per_batch=cycles, shuffle=True, seed=seed,
        num_workers=int(training.get("num_workers", 0)),
    )
    val_loader = loader_factory(
        val, cycles_per_batch=2 * cycles, shuffle=False, seed=seed,
        num_workers=int(training.get("num_workers", 0)),
    )
    model = build_aggregate_model(
        config["model"], train.static, total_capacity_mw=train.total_capacity_mw,
        history_steps=int(train.manifest["history_power_steps"]),
        forecast_steps=int(train.manifest["forecast_power_steps"]),
    )
    base_checkpoint_identity = None
    initial_checkpoint_identity = None
    if architecture in {
        "aggregate_strict_modulation", "aggregate_response_state", "aggregate_weather_dynamics"
    }:
        if not args.base_checkpoint:
            raise ValueError(f"{architecture} requires --base-checkpoint")
        checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("data_manifest_identity") != train.manifest["preprocess_fingerprint"]:
            raise ValueError("R2-W base checkpoint was trained with another aggregate cache")
        if int(checkpoint.get("seed", -1)) != seed:
            raise ValueError("Strict modulation requires a same-seed R2-W base checkpoint")
        base_config = checkpoint.get("model_config", {})
        if not (
            base_config.get("architecture") == "aggregate_correction"
            and base_config.get("weather_route") == "r2_latent_interpolation"
            and base_config.get("correction_conditioning") == "weather"
        ):
            raise ValueError("--base-checkpoint is not a complete R2-W checkpoint")
        model.load_base_state(checkpoint["model_state"])
        base_checkpoint_identity = str(Path(args.base_checkpoint).resolve())
        print(f"[aggregate-init] complete-r2-w={base_checkpoint_identity}", flush=True)
        if args.auxiliary_checkpoint:
            if not dynamics_architecture:
                raise ValueError("--auxiliary-checkpoint is only valid for weather dynamics")
            auxiliary = torch.load(args.auxiliary_checkpoint, map_location="cpu", weights_only=False)
            if int(auxiliary.get("seed", -1)) != seed:
                raise ValueError("Auxiliary weather checkpoint has a different seed")
            auxiliary_mode = auxiliary.get("model_config", {}).get("weather_dynamics_mode")
            expected = (
                "weather_envelope"
                if str(config["model"].get("weather_dynamics_mode")) == "m5_envelope_realization"
                else "weather_mean"
            )
            if auxiliary_mode != expected:
                raise ValueError(f"Dynamics mode requires auxiliary {expected}, got {auxiliary_mode}")
            model.load_auxiliary_state(auxiliary["model_state"])
            auxiliary_checkpoint_identity = str(Path(args.auxiliary_checkpoint).resolve())
            print(f"[aggregate-init] auxiliary={auxiliary_checkpoint_identity}", flush=True)
        else:
            auxiliary_checkpoint_identity = None
        if args.initial_checkpoint:
            initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
            if initial.get("data_manifest_identity") != train.manifest["preprocess_fingerprint"]:
                raise ValueError("Strict initialization checkpoint uses another aggregate cache")
            if int(initial.get("seed", -1)) != seed:
                raise ValueError("Strict initialization checkpoint has a different seed")
            if initial.get("model_config", {}).get("architecture") != architecture:
                raise ValueError("--initial-checkpoint has the wrong aggregate architecture")
            if response_architecture:
                if (
                    initial.get("response_manifest_identity")
                    != train.response_manifest["preprocess_fingerprint"]
                ):
                    raise ValueError("Response initialization checkpoint uses another response cache")
                if (
                    initial.get("response_latent_identity")
                    != train.latent_manifest["base_checkpoint_sha256"]
                ):
                    raise ValueError("Response initialization checkpoint uses another latent cache")
                initial_model = initial.get("model_config", {})
                for key in (
                    "weather_route", "history_visible_steps", "response_state_input",
                    "response_state_hidden", "modulation_low_rank", "modulation_amplitude",
                ):
                    if initial_model.get(key) != config["model"].get(key):
                        raise ValueError(
                            f"Response initialization checkpoint changes frozen setting: {key}"
                        )
                if initial.get("base_checkpoint") != base_checkpoint_identity:
                    raise ValueError("Response initialization checkpoint uses another R2-W base")
            model.load_state_dict(initial["model_state"], strict=True)
            training_stage = (
                str(config["model"]["dynamics_training_stage"])
                if dynamics_architecture
                else str(config["model"]["modulation_training_stage"])
            )
            model.configure_training_stage(training_stage)
            initial_checkpoint_identity = str(Path(args.initial_checkpoint).resolve())
            print(f"[aggregate-init] strict-stage={initial_checkpoint_identity}", flush=True)
        if response_architecture and not bool(model.response_reference_calibrated):
            calibrate_response_reference(model, train, resolve_device(args.device))
        elif not response_architecture and not dynamics_architecture and not bool(model.state_reference_calibrated):
            calibrate_strict_state_reference(model, train, resolve_device(args.device))
    elif hasattr(model, "load_history_state"):
        if args.history_checkpoint:
            checkpoint = torch.load(args.history_checkpoint, map_location="cpu", weights_only=False)
            if checkpoint.get("data_manifest_identity") != train.manifest["preprocess_fingerprint"]:
                raise ValueError("History checkpoint was trained with another aggregate cache")
            model.load_history_state(checkpoint["model_state"])
            print(f"[aggregate-init] history={args.history_checkpoint}", flush=True)
        elif bool(config["model"].get("require_history_checkpoint", True)):
            raise ValueError("aggregate_correction requires --history-checkpoint")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        **training["deterministic"],
        "gradient_clip": training.get("gradient_clip", 1.0),
        "early_stopping_patience": training.get("early_stopping_patience", 10),
        "decoder_lr_multiplier": config["model"].get("decoder_lr_multiplier", 0.1),
        "response_identification_loss_weight": config["model"].get(
            "response_identification_loss_weight", 0.0
        ),
        "weather_quantile": config["model"].get("weather_quantile", 0.0),
        "current_weather_loss_weight": config["model"].get("current_weather_loss_weight", 0.0),
    }
    print(
        f"[aggregate-run] architecture={architecture} seed={seed} device={args.device} "
        f"parameters={total_parameters} trainable={trainable_parameters} "
        f"train_samples={len(train)} val_samples={len(val)}",
        flush=True,
    )
    history = fit_aggregate(
        model, train_loader, val_loader, settings, resolve_device(args.device),
        output / "best_model.pt",
        {
            "stage": "aggregate_forecast",
            "model_config": config["model"],
            "data_manifest_identity": train.manifest["preprocess_fingerprint"],
            "seed": seed,
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "base_checkpoint": base_checkpoint_identity,
            "initial_checkpoint": initial_checkpoint_identity,
            "auxiliary_checkpoint": (
                auxiliary_checkpoint_identity if dynamics_architecture else None
            ),
            "response_manifest_identity": (
                train.response_manifest["preprocess_fingerprint"] if response_architecture else None
            ),
            "response_latent_identity": (
                train.latent_manifest["base_checkpoint_sha256"]
                if response_architecture or dynamics_architecture else None
            ),
            "dynamics_time_identity": (
                train.dynamics_time_manifest["preprocess_fingerprint"]
                if dynamics_architecture else None
            ),
            "dynamics_latent_identity": (
                train.dynamics_latent_manifest["base_checkpoint_sha256"]
                if dynamics_architecture else None
            ),
        },
        precision=str(training.get("precision", "bf16")),
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    save_json(output / "history.json", history)
    train.close()
    val.close()


if __name__ == "__main__":
    main()
