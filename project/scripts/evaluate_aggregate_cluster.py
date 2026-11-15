from __future__ import annotations

import argparse
from pathlib import Path
import copy

import numpy as np
import torch

from wind_forcing.aggregate_training import evaluate_aggregate, save_archive, save_json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an aggregate-only checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--state-mode-override",
        choices=("current", "recent_1h", "recent_3h", "full_6h", "multiscale", "horizon"),
    )
    parser.add_argument(
        "--response-state-intervention",
        choices=("none", "power_shuffle", "weather_shuffle", "pair_shift", "response_mean", "support_only"),
        default="none",
        help="Intervene only on historical response-state identification inputs.",
    )
    parser.add_argument("--state-visible-steps-override", type=int)
    parser.add_argument(
        "--correction-state-intervention",
        choices=("none", "cyclic_shift", "global_permutation", "reference"),
        default="none",
        help=(
            "Intervene only on the history seen by the correction state encoder; "
            "the history baseline always receives the original history."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    architecture = str(config["model"]["architecture"])
    data = config["data"]
    response_architecture = architecture == "aggregate_response_state"
    dynamics_architecture = architecture == "aggregate_weather_dynamics"
    if dynamics_architecture:
        seed = int(config["seed"])
        realization = data.get("realization_cache_root")
        if realization:
            realization = str(Path(realization) / f"seed{seed}")
        dataset = AggregateWeatherDynamicsDataset(
            data["cache_dir"], data["response_cache_dir"],
            Path(data["response_latent_cache_root"]) / f"seed{seed}",
            data["dynamics_time_cache_dir"],
            Path(data["dynamics_latent_cache_root"]) / f"seed{seed}",
            args.split, realization_cache=realization,
        )
    elif response_architecture:
        latent_cache = Path(data["response_latent_cache_root"]) / f"seed{int(config['seed'])}"
        dataset = AggregateResponseForecastDataset(
            data["cache_dir"], data["response_cache_dir"], latent_cache, args.split
        )
    else:
        include_weather = not architecture.startswith("hb_")
        dataset = AggregateForecastDataset(
            data["cache_dir"], args.split, include_weather=include_weather,
            preload_weather=data.get("preload_weather", "auto"),
            preload_weather_max_gib=float(data.get("preload_weather_max_gib", 16.0)),
        )
    loader_factory = (
        create_aggregate_dynamics_loader if dynamics_architecture
        else create_aggregate_response_loader if response_architecture
        else create_aggregate_loader
    )
    loader = loader_factory(
        dataset, cycles_per_batch=2 * int(config["training"]["cycles_per_batch"]),
        shuffle=False, num_workers=int(config["training"].get("num_workers", 0)),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("data_manifest_identity") != dataset.manifest["preprocess_fingerprint"]:
        raise ValueError("Checkpoint and aggregate cache differ")
    if response_architecture:
        if checkpoint.get("response_manifest_identity") != dataset.response_manifest["preprocess_fingerprint"]:
            raise ValueError("Checkpoint and response cache differ")
        if checkpoint.get("response_latent_identity") != dataset.latent_manifest["base_checkpoint_sha256"]:
            raise ValueError("Checkpoint and response latent cache differ")
    if dynamics_architecture:
        if checkpoint.get("response_latent_identity") != dataset.latent_manifest["base_checkpoint_sha256"]:
            raise ValueError("Dynamics checkpoint and R2 latent cache differ")
        if checkpoint.get("dynamics_time_identity") != dataset.dynamics_time_manifest["preprocess_fingerprint"]:
            raise ValueError("Dynamics checkpoint and time cache differ")
        if checkpoint.get("dynamics_latent_identity") != dataset.dynamics_latent_manifest["base_checkpoint_sha256"]:
            raise ValueError("Dynamics checkpoint and current latent cache differ")
    model_config = copy.deepcopy(checkpoint["model_config"])
    if args.state_mode_override:
        model_config["state_mode"] = args.state_mode_override
        model_config["fusion_history_state_mode"] = args.state_mode_override
    model = build_aggregate_model(
        model_config, dataset.static,
        total_capacity_mw=dataset.total_capacity_mw,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    correction_state_lookup = None
    response_power_lookup = None
    response_weather_lookup = None
    if args.correction_state_intervention == "reference":
        if not hasattr(model, "use_reference_state"):
            raise ValueError("reference intervention requires a strict-modulation model")
        model.use_reference_state = True
    elif args.correction_state_intervention == "global_permutation":
        if not hasattr(model, "use_reference_state"):
            raise ValueError("global permutation requires a strict-modulation model")
        # This permutation is global, fixed, and independent of loader batch
        # boundaries. It intervenes on correction state only.
        random = np.random.default_rng(20260317)
        permutation = random.permutation(len(dataset))
        histories = np.asarray(dataset.arrays["cluster_history_mw"], dtype=np.float32)
        histories = histories / dataset.total_capacity_mw
        sample_ids = np.asarray(dataset.arrays["sample_index"], dtype=np.int64)
        correction_state_lookup = {
            int(sample_id): histories[source]
            for sample_id, source in zip(sample_ids, permutation)
        }
    if args.state_visible_steps_override is not None:
        if not hasattr(model, "state_encoder"):
            raise ValueError("State visibility intervention requires an aggregate correction model")
        if not 1 <= args.state_visible_steps_override <= 24:
            raise ValueError("--state-visible-steps-override must be between 1 and 24")
        model.state_encoder.visible_steps_override = args.state_visible_steps_override
    if args.response_state_intervention != "none" and not response_architecture:
        raise ValueError("Response-state interventions require aggregate_response_state")
    if args.response_state_intervention == "response_mean":
        model.response_use_reference_state = True
    elif args.response_state_intervention in {"power_shuffle", "weather_shuffle"}:
        random = np.random.default_rng(20260411)
        permutation = random.permutation(len(dataset))
        sample_ids = np.asarray(dataset.arrays["sample_index"], dtype=np.int64)
        if args.response_state_intervention == "power_shuffle":
            values = np.asarray(dataset.arrays["past_cluster_power_mw"], dtype=np.float32)
            values = values / dataset.total_capacity_mw
            response_power_lookup = {
                int(sample_id): values[source]
                for sample_id, source in zip(sample_ids, permutation)
            }
        else:
            cycles = np.asarray(dataset.arrays["past_weather_cycle_index"], dtype=np.int64)
            source_values = np.asarray(dataset.past_hourly_latent[cycles[permutation]], dtype=np.float32)
            response_weather_lookup = {
                int(sample_id): source_values[index]
                for index, sample_id in enumerate(sample_ids)
            }
    model.to(resolve_device(args.device))
    settings = {
        **config["training"]["deterministic"],
        "baseline_anchor_loss_weight": 0.0,
    }
    metrics, archive = evaluate_aggregate(
        model, loader, resolve_device(args.device), str(config["training"].get("precision", "bf16")),
        settings, max_batches=args.max_batches, description=f"Aggregate {args.split}",
        collect_predictions=bool(args.predictions_output), collect_diagnostics=True,
        correction_state_intervention=args.correction_state_intervention,
        correction_state_lookup=correction_state_lookup,
        response_intervention=args.response_state_intervention,
        response_power_lookup=response_power_lookup,
        response_weather_lookup=response_weather_lookup,
    )
    metrics["parameter_count"] = int(checkpoint.get("parameter_count", sum(p.numel() for p in model.parameters())))
    metrics["trainable_parameter_count"] = int(checkpoint.get("trainable_parameter_count", sum(p.numel() for p in model.parameters() if p.requires_grad)))
    save_json(args.output, metrics)
    if args.predictions_output:
        assert archive is not None
        save_archive(args.predictions_output, archive)
    dataset.close()
    peak = metrics["peak_allocated_mib"]
    peak_text = "n/a" if peak is None else f"{peak:.1f}MiB"
    print(
        f"[{args.split}] samples={metrics['samples']} "
        f"mae={metrics['cluster_mae_mw']:.3f}MW "
        f"rmse={metrics['cluster_rmse_mw']:.3f}MW "
        f"r2={metrics['cluster_r2']:.4f} "
        f"latency={metrics['latency_ms_per_origin']:.3f}ms/origin "
        f"peak={peak_text} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
