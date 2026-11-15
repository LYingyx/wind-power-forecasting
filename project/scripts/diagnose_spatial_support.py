from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wind_forcing.data import ForecastDataset, create_forecast_loader
from wind_forcing.models import build_forecast_model
from wind_forcing.training import autocast_context, model_inputs, move_batch, resolve_device


SEEDS = list(range(42, 52))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p05": float(np.quantile(values, 0.05)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
    }


@torch.inference_mode()
def diagnose_seed(
    checkpoint_path: Path,
    dataset: ForecastDataset,
    loader,
    device: torch.device,
    precision: str,
    max_batches: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_forecast_model(
        checkpoint["model_config"],
        dataset.static,
        history_steps=int(dataset.manifest["history_power_steps"]),
        forecast_steps=int(dataset.manifest["forecast_power_steps"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    query = model.forcing_encoder.station_query
    if not hasattr(query, "relative_geometry"):
        raise TypeError("Spatial diagnosis expects a single-scale station query")
    geometry = query.relative_geometry.detach().float().cpu().numpy()
    patch_index = query.patch_index.detach().cpu()
    station_count, token_count = patch_index.shape
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError("Spatial patch is not square")
    edge = np.zeros((side, side), dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    edge = edge.reshape(-1)
    far = np.zeros((station_count, token_count), dtype=bool)
    far_count = max(1, token_count // 4)
    for station in range(station_count):
        far[station, np.argsort(geometry[station, :, 2])[-far_count:]] = True
    grid_scale_km = 111.32 * abs(
        float(dataset.static["weather_latitude"][1])
        - float(dataset.static["weather_latitude"][0])
    )

    effective_tokens: list[np.ndarray] = []
    edge_ratio: list[np.ndarray] = []
    far_ratio: list[np.ndarray] = []
    radius_km: list[np.ndarray] = []
    centroid_fraction: list[np.ndarray] = []
    along_centroid: list[np.ndarray] = []
    cross_centroid: list[np.ndarray] = []
    station_radius_sum = np.zeros(station_count, dtype=np.float64)
    station_edge_sum = np.zeros(station_count, dtype=np.float64)
    station_far_sum = np.zeros(station_count, dtype=np.float64)
    station_count_values = np.zeros(station_count, dtype=np.int64)
    cycle_count = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            output = model(**model_inputs(batch))
        weights_tensor = output["spatial_attention_unique"]
        if weights_tensor is None:
            raise RuntimeError("Checkpoint did not return spatial attention")
        weights = weights_tensor.detach().float().cpu().numpy()
        unique_cycles = weights.shape[0]
        cycle_count += unique_cycles
        entropy = -(weights * np.log(np.clip(weights, 1e-12, None))).sum(axis=-1)
        effective = np.exp(entropy)
        current_edge = weights[..., edge].sum(axis=-1) / float(edge.mean())
        current_far = (weights * far[None, None]).sum(axis=-1) / float(
            far_count / token_count
        )
        dx = (weights * geometry[None, None, :, :, 0]).sum(axis=-1)
        dy = (weights * geometry[None, None, :, :, 1]).sum(axis=-1)
        current_radius = (
            weights * geometry[None, None, :, :, 2]
        ).sum(axis=-1) * grid_scale_km
        maximum_radius = geometry[:, :, 2].max(axis=-1)
        current_centroid_fraction = np.sqrt(dx**2 + dy**2) / np.maximum(
            maximum_radius[None, None], 1e-6
        )

        weather = batch["weather"].detach().float().cpu()
        flat_index = patch_index.flatten()
        u = weather[:, :, 1].flatten(-2)[:, :, flat_index].reshape(
            unique_cycles, weights.shape[1], station_count, token_count
        )
        v = weather[:, :, 2].flatten(-2)[:, :, flat_index].reshape_as(u)
        u = u * float(query.u10_std) + float(query.u10_mean)
        v = v * float(query.v10_std) + float(query.v10_mean)
        weight_cpu = torch.from_numpy(weights)
        mean_u = (u * weight_cpu).sum(dim=-1).numpy()
        mean_v = (v * weight_cpu).sum(dim=-1).numpy()
        wind_norm = np.sqrt(mean_u**2 + mean_v**2).clip(min=1e-6)
        along = (dx * mean_u + dy * mean_v) / wind_norm
        cross = (dx * mean_v - dy * mean_u) / wind_norm

        effective_tokens.append(effective.reshape(-1))
        edge_ratio.append(current_edge.reshape(-1))
        far_ratio.append(current_far.reshape(-1))
        radius_km.append(current_radius.reshape(-1))
        centroid_fraction.append(current_centroid_fraction.reshape(-1))
        along_centroid.append(along.reshape(-1))
        cross_centroid.append(cross.reshape(-1))
        station_radius_sum += current_radius.sum(axis=(0, 1))
        station_edge_sum += current_edge.sum(axis=(0, 1))
        station_far_sum += current_far.sum(axis=(0, 1))
        station_count_values += unique_cycles * weights.shape[1]

    arrays = {
        "effective_tokens": np.concatenate(effective_tokens),
        "edge_relative_mass": np.concatenate(edge_ratio),
        "far_quartile_relative_mass": np.concatenate(far_ratio),
        "effective_radius_km": np.concatenate(radius_km),
        "centroid_support_fraction": np.concatenate(centroid_fraction),
        "wind_along_centroid_grid": np.concatenate(along_centroid),
        "wind_cross_centroid_grid": np.concatenate(cross_centroid),
    }
    seed = int(checkpoint.get("seed", -1))
    row: dict[str, Any] = {
        "seed": seed,
        "nwp_cycles": cycle_count,
        "attention_values": int(len(arrays["effective_tokens"])),
        "patch_tokens": token_count,
        "uniform_edge_mass": float(edge.mean()),
        "uniform_far_quartile_mass": float(far_count / token_count),
    }
    for name, values in arrays.items():
        row.update(summarize(values, name))
    station_rows = []
    for station in range(station_count):
        station_rows.append(
            {
                "seed": seed,
                "station_index": station,
                "mean_effective_radius_km": float(
                    station_radius_sum[station] / station_count_values[station]
                ),
                "mean_edge_relative_mass": float(
                    station_edge_sum[station] / station_count_values[station]
                ),
                "mean_far_quartile_relative_mass": float(
                    station_far_sum[station] / station_count_values[station]
                ),
                "max_support_radius_km": float(
                    geometry[station, :, 2].max() * grid_scale_km
                ),
            }
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, station_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose learned 4x4 NWP support")
    parser.add_argument(
        "--checkpoint-root",
        default="artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h",
    )
    parser.add_argument("--cache-dir", default="data/model_cache_pad5_f8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--cycles-per-batch", type=int, default=12)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--output-dir", default="artifacts/cross_farm_context_v1/spatial_diagnosis"
    )
    args = parser.parse_args()

    root = Path(args.checkpoint_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = ForecastDataset(
        args.cache_dir,
        "val",
        preload_weather="auto",
        preload_weather_max_gib=16.0,
        include_weather=True,
    )
    loader = create_forecast_loader(
        dataset,
        cycles_per_batch=args.cycles_per_batch,
        shuffle=False,
        num_workers=0,
    )
    device = resolve_device(args.device)
    seed_rows = []
    station_rows = []
    for seed in SEEDS:
        checkpoint = root / f"seed{seed}" / "best_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        row, current_station_rows = diagnose_seed(
            checkpoint,
            dataset,
            loader,
            device,
            args.precision,
            args.max_batches,
        )
        seed_rows.append(row)
        station_rows.extend(current_station_rows)
        print(
            f"[spatial-diagnosis] seed={seed} "
            f"N_eff={row['effective_tokens_median']:.3f} "
            f"edge={row['edge_relative_mass_median']:.3f} "
            f"far={row['far_quartile_relative_mass_median']:.3f}",
            flush=True,
        )

    aggregate = {
        key: float(np.mean([row[key] for row in seed_rows]))
        for key in seed_rows[0]
        if key.endswith(("_mean", "_median", "_p05", "_p95"))
    }
    no_boundary_pressure = bool(
        aggregate["edge_relative_mass_median"] <= 1.05
        and aggregate["far_quartile_relative_mass_median"] <= 1.10
        and aggregate["centroid_support_fraction_p95"] < 0.65
    )
    decision = {
        "no_boundary_pressure": no_boundary_pressure,
        "next_step": (
            "freeze_4x4_and_skip_formal_window_search"
            if no_boundary_pressure
            else "run_formal_4x4_vs_6x6_support_experiment"
        ),
        "diagnostic_thresholds": {
            "edge_relative_mass_median_max": 1.05,
            "far_quartile_relative_mass_median_max": 1.10,
            "centroid_support_fraction_p95_max": 0.65,
        },
        "interpretation_limit": (
            "This is a support-pressure diagnostic, not proof that 4x4 is globally optimal."
        ),
    }
    payload = {
        "protocol": {
            "split": "2023 validation",
            "test_split_used": False,
            "seeds": SEEDS,
            "weighting": "each unique NWP cycle, hour, and station equally",
        },
        "aggregate_across_seed_summaries": aggregate,
        "decision": decision,
        "per_seed": seed_rows,
    }
    write_csv(output / "per_seed.csv", seed_rows)
    write_csv(output / "per_station.csv", station_rows)
    write_json(output / "diagnosis.json", payload)
    dataset.close()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
