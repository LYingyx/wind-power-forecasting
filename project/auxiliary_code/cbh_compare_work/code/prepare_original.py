"""Evaluate frozen original CBH checkpoints and cache their history-only trajectories.

The evaluation uses the exact source rows selected by the temporal-SR pilot.  It
does not retrain or modify the original 56-station model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from paths import CACHE, ORIGINAL, ORIGINAL_RUNS, ROOT, V2
from artifact_utils import dump_json, sha256


sys.path.insert(0, str(ORIGINAL / "snapshot/src"))
from wind_forcing.config import load_config  # noqa: E402
from wind_forcing.data.dataset import ForecastCollator, ForecastDataset  # noqa: E402
from wind_forcing.models import build_forecast_model  # noqa: E402


SEEDS = list(range(42, 52))


def checkpoint_path(seed: int) -> Path:
    return ORIGINAL_RUNS / "results" / f"CBH_seed{seed}" / "best_model.pt"


def config_path(seed: int) -> Path:
    return ORIGINAL / "configs" / f"CBH_seed{seed}.yaml"


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build(seed: int, static: dict[str, np.ndarray], device: torch.device):
    checkpoint = checkpoint_path(seed)
    frozen = json.loads((ORIGINAL / "status/VALIDATION_FROZEN.json").read_text())
    expected = frozen["checkpoints"][f"CBH_seed{seed}"]
    assert sha256(checkpoint) == expected
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = load_config(config_path(seed))
    assert payload["model_config"] == config["model"]
    model = build_forecast_model(
        payload["model_config"], static, history_steps=24, forecast_steps=24
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


def audit_alignment(datasets: dict[str, ForecastDataset], station_indices: list[int]) -> dict:
    checks = {}
    for split, dataset in datasets.items():
        pilot = V2 / "data/pilot" / split
        source = np.load(pilot / "source_indices.npy")
        row = {
            "history_max_abs_error": float(
                np.max(
                    np.abs(
                        dataset.arrays["station_history"][source][:, station_indices]
                        - np.load(pilot / "history.npy")
                    )
                )
            ),
            "target_max_abs_error": float(
                np.max(
                    np.abs(
                        dataset.arrays["station_future"][source][:, station_indices]
                        - np.load(pilot / "target.npy")
                    )
                )
            ),
            "origin_equal": bool(
                np.array_equal(dataset.arrays["origin_unix"][source], np.load(pilot / "origin.npy"))
            ),
            "phase_equal": bool(
                np.array_equal(dataset.arrays["origin_phase"][source], np.load(pilot / "phase.npy"))
            ),
            "sample_index_equal": bool(
                np.array_equal(dataset.arrays["sample_index"][source], np.load(pilot / "sample_id.npy"))
            ),
        }
        assert row["history_max_abs_error"] == 0 and row["target_max_abs_error"] == 0
        assert row["origin_equal"] and row["phase_equal"] and row["sample_index_equal"]
        checks[split] = row
    return checks


@torch.inference_mode()
def baseline_cache(
    model: torch.nn.Module,
    dataset: ForecastDataset,
    source_indices: np.ndarray,
    station_indices: list[int],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    collate = ForecastCollator(dataset)
    result = []
    for start in range(0, len(source_indices), batch_size):
        indices = source_indices[start : start + batch_size].tolist()
        batch = move(collate(indices), device)
        output = model.baseline_model(
            station_history=batch["station_history"],
            cluster_history=batch["cluster_history"],
            origin_phase=batch["origin_phase"],
        )["station_factor"]
        result.append(output[:, station_indices].float().cpu().numpy())
    return np.concatenate(result)


@torch.inference_mode()
def evaluate_original(
    model: torch.nn.Module,
    dataset: ForecastDataset,
    source_indices: np.ndarray,
    station_indices: list[int],
    device: torch.device,
    precision: str,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    collate = ForecastCollator(dataset)
    predictions, baselines, targets = [], [], []
    enabled = precision == "bf16"
    for start in range(0, len(source_indices), batch_size):
        indices = source_indices[start : start + batch_size].tolist()
        batch = move(collate(indices), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
            output = model(**batch)
        predictions.append(output["station_factor"][:, station_indices].float().cpu().numpy())
        baselines.append(output["baseline_station_factor"][:, station_indices].float().cpu().numpy())
        targets.append(batch["station_future"][:, station_indices].float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(baselines), np.concatenate(targets)


def metric_block(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "per_station_mae": np.abs(error).mean(axis=(0, 2)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    farms = json.loads((V2 / "data/selected_farms.json").read_text())
    station_ids = farms["station_ids"]
    station_indices = [int(row["station_index"]) for row in farms["stations"]]
    datasets = {
        "train": ForecastDataset(CACHE, "train", include_weather=False, verbose=False),
        "val": ForecastDataset(CACHE, "val", preload_weather=False, include_weather=True, verbose=False),
    }
    alignment = audit_alignment(datasets, station_indices)
    report_index = np.load(V2 / "data/val_report_indices.npy")
    source_by_split = {
        split: np.load(V2 / "data/pilot" / split / "source_indices.npy")
        for split in ["train", "val"]
    }
    report_source = source_by_split["val"][report_index]
    pilot_report_target = np.load(V2 / "data/pilot/val/target.npy")[report_index]
    original_results = []

    for seed in args.seeds:
        model, checkpoint = build(seed, datasets["val"].static, device)
        cache_files = {}
        for split in ["train", "val"]:
            values = baseline_cache(
                model, datasets[split], source_by_split[split], station_indices, device
            )
            path = ROOT / "cache/baseline" / f"{split}_seed{seed}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, values)
            cache_files[split] = {
                "path": str(path.relative_to(ROOT)),
                "shape": list(values.shape),
                "sha256": sha256(path),
            }

        fp32, fp32_base, target = evaluate_original(
            model, datasets["val"], report_source, station_indices, device, "fp32"
        )
        bf16, bf16_base, target_bf16 = evaluate_original(
            model, datasets["val"], report_source, station_indices, device, "bf16"
        )
        assert np.array_equal(target, target_bf16)
        assert np.array_equal(target, pilot_report_target)
        out = ROOT / "original_eval" / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "predictions_fp32.npy", fp32)
        np.save(out / "predictions_bf16.npy", bf16)
        np.save(out / "baseline_fp32.npy", fp32_base)
        np.save(out / "baseline_bf16.npy", bf16_base)
        metrics = {
            "model": "CBH_ORIGINAL_56_SUBSET",
            "seed": seed,
            "station_ids": station_ids,
            "station_indices": station_indices,
            "report_n": int(len(report_source)),
            "fp32": metric_block(fp32, target),
            "bf16": metric_block(bf16, target),
            "fp32_bf16_prediction_max_abs_error": float(np.max(np.abs(fp32 - bf16))),
            "fp32_bf16_baseline_max_abs_error": float(np.max(np.abs(fp32_base - bf16_base))),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_sha256": sha256(checkpoint_path(seed)),
            "source_config_sha256": sha256(config_path(seed)),
            "baseline_cache": cache_files,
        }
        dump_json(out / "metrics.json", metrics)
        original_results.append(metrics)
        print(seed, metrics["fp32"]["mae"], metrics["bf16"]["mae"], flush=True)
        del model
        torch.cuda.empty_cache()

    np.save(ROOT / "cache/report_target.npy", pilot_report_target)
    dump_json(
        ROOT / "status/PREPARE_ORIGINAL.json",
        {
            "passed": True,
            "station_ids": station_ids,
            "station_indices": station_indices,
            "seeds": args.seeds,
            "alignment": alignment,
            "report_source_indices_sha256": sha256(V2 / "data/pilot/val/source_indices.npy"),
            "report_indices_sha256": sha256(V2 / "data/val_report_indices.npy"),
            "original_checkpoint_results": original_results,
        },
    )
    for dataset in datasets.values():
        dataset.close()


if __name__ == "__main__":
    main()
