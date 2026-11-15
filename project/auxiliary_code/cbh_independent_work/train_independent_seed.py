"""Canonical one-site/one-seed independent baseline and CBH training.

This is the production path.  Each OS process owns one native nn.Module,
optimizer, CUDA RNG stream and data-order RNG at a time.  It therefore avoids
parameter sharing both across sites and across seeds.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from independent_models import (
    BASELINE_CONFIG,
    CACHE,
    CBH_CONFIG,
    OUT,
    SEEDS,
    SOURCE,
    TRAINING_PROTOCOL,
    atomic_json,
    atomic_torch_save,
    build_baseline,
    build_cbh,
    load_static,
    seed_all,
    tensor_state_hash,
)

sys.path.insert(0, str(SOURCE))


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


class SiteSplit:
    def __init__(self, split: str, site: int, include_weather: bool) -> None:
        root = CACHE / split
        self.split = split
        self.site = int(site)
        self.history = np.load(root / "station_history.npy", mmap_mode="r")
        self.future = np.load(root / "station_future.npy", mmap_mode="r")
        self.phase = np.load(root / "origin_phase.npy", mmap_mode="r")
        self.weather_cycle = np.load(root / "weather_cycle_index.npy", mmap_mode="r")
        self.weather_offsets = np.load(root / "weather_time_offset_hours.npy", mmap_mode="r")
        self.local_weather = (
            np.load(OUT / "input_cache" / f"{split}_p0i_local.npy", mmap_mode="r")
            if include_weather
            else None
        )
        self.length = len(self.history)
        if self.history.shape != (self.length, 56, 24) or self.future.shape != self.history.shape:
            raise RuntimeError(f"Unexpected station arrays for {split}: {self.history.shape}")

    def batch(self, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.asarray(indices, dtype=np.int64)
        history = np.asarray(self.history[indices, self.site : self.site + 1], dtype=np.float32)
        future = np.asarray(self.future[indices, self.site : self.site + 1], dtype=np.float32)
        result = {
            "station_history": torch.from_numpy(history).to(device, non_blocking=True),
            "cluster_history": torch.from_numpy(history[:, 0].copy()).to(device, non_blocking=True),
            "station_future": torch.from_numpy(future).to(device, non_blocking=True),
            "origin_phase": torch.from_numpy(np.asarray(self.phase[indices], dtype=np.int64)).to(device, non_blocking=True),
        }
        if self.local_weather is not None:
            cycles = np.asarray(self.weather_cycle[indices], dtype=np.int64)
            unique, inverse = np.unique(cycles, return_inverse=True)
            weather = np.asarray(self.local_weather[unique, self.site], dtype=np.float32)
            result.update(
                weather=torch.from_numpy(weather[..., None, None]).to(device, non_blocking=True),
                weather_batch_index=torch.from_numpy(inverse.astype(np.int64)).to(device, non_blocking=True),
                weather_time_offset_hours=torch.from_numpy(
                    np.asarray(self.weather_offsets[indices], dtype=np.float32)
                ).to(device, non_blocking=True),
            )
        return result


def cycle_batches(cycle_index: np.ndarray, cycles_per_batch: int, shuffle: bool, seed: int, epoch: int) -> list[np.ndarray]:
    values = np.asarray(cycle_index, dtype=np.int64)
    unique = np.unique(values)
    members = {int(cycle): np.flatnonzero(values == cycle).astype(np.int64) for cycle in unique}
    cycles = unique.copy()
    rng = np.random.default_rng(int(seed) + int(epoch))
    if shuffle:
        rng.shuffle(cycles)
    result = []
    for start in range(0, len(cycles), cycles_per_batch):
        selected = cycles[start : start + cycles_per_batch]
        batch = np.concatenate([members[int(cycle)] for cycle in selected])
        if shuffle:
            rng.shuffle(batch)
        result.append(batch)
    return result


def baseline_orders(length: int, seed: int, epochs: int) -> list[list[np.ndarray]]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    result = []
    for _ in range(epochs):
        order = torch.randperm(length, generator=generator).numpy()
        result.append([order[start : start + 48] for start in range(0, length, 48)])
    return result


def model_inputs(batch: Mapping[str, torch.Tensor], include_weather: bool) -> dict[str, torch.Tensor]:
    result = {
        "station_history": batch["station_history"],
        "cluster_history": batch["cluster_history"],
        "origin_phase": batch["origin_phase"],
    }
    if include_weather:
        result.update(
            weather=batch["weather"],
            weather_batch_index=batch["weather_batch_index"],
            weather_time_offset_hours=batch["weather_time_offset_hours"],
        )
    return result


def autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )


def detached_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: SiteSplit,
    device: torch.device,
    include_weather: bool,
    max_batches: int | None = None,
) -> float:
    model.eval()
    batches = (
        cycle_batches(data.weather_cycle, 24, False, 0, 0)
        if include_weather
        else [np.arange(start, min(data.length, start + 96), dtype=np.int64) for start in range(0, data.length, 96)]
    )
    absolute = 0.0
    count = 0
    for number, indices in enumerate(batches):
        if max_batches is not None and number >= max_batches:
            break
        batch = data.batch(indices, device)
        with autocast(device):
            prediction = model(**model_inputs(batch, include_weather))["station_factor"]
        absolute += float((prediction.float() - batch["station_future"].float()).abs().sum())
        count += int(prediction.numel())
    if count == 0:
        raise RuntimeError("Evaluation produced no values")
    return absolute / count


def train_model(
    model: torch.nn.Module,
    train: SiteSplit,
    val: SiteSplit,
    device: torch.device,
    include_weather: bool,
    seed: int,
    epochs: int,
    patience: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], float, int, float]:
    model.to(device)
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(TRAINING_PROTOCOL["learning_rate"]),
        weight_decay=float(TRAINING_PROTOCOL["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    orders = baseline_orders(train.length, seed, epochs) if not include_weather else None
    best = float("inf")
    best_epoch = 0
    stale = 0
    best_state = None
    history = []
    peak_mib = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        if include_weather:
            model.baseline_model.eval()
            batches = cycle_batches(train.weather_cycle, 12, True, seed, epoch)
        else:
            assert orders is not None
            batches = orders[epoch]
        loss_sum = 0.0
        samples = 0
        for number, indices in enumerate(batches):
            if max_train_batches is not None and number >= max_train_batches:
                break
            batch = train.batch(indices, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                prediction = model(**model_inputs(batch, include_weather))["station_factor"]
                loss = (prediction.float() - batch["station_future"].float()).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(TRAINING_PROTOCOL["gradient_clip"]))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            samples += len(indices)
        scheduler.step()
        validation = evaluate(model, val, device, include_weather, max_val_batches)
        improved = validation < best
        if improved:
            best = validation
            best_epoch = epoch + 1
            stale = 0
            best_state = detached_state(model)
        else:
            stale += 1
        if device.type == "cuda":
            peak_mib = max(peak_mib, torch.cuda.max_memory_reserved(device) / 2**20)
        history.append(
            {
                "epoch": epoch + 1,
                "train_mae_cf": loss_sum / samples,
                "val_mae_cf": validation,
                "best_val_mae_cf": best,
                "best_epoch": best_epoch,
                "improved": improved,
                "stale": stale,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            json.dumps(
                {
                    "kind": "cbh" if include_weather else "baseline",
                    "seed": seed,
                    "epoch": epoch + 1,
                    "train": history[-1]["train_mae_cf"],
                    "val": validation,
                    "best": best,
                    "stale": stale,
                    "seconds": history[-1]["seconds"],
                    "peak_mib": peak_mib,
                }
            ),
            flush=True,
        )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No best checkpoint was captured")
    return best_state, history, best, best_epoch, peak_mib


def checkpoint_payload(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    site: int,
    seed: int,
    stage: str,
    epoch: int,
    val_mae: float,
    static: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "model_config": dict(config),
        "data_manifest_identity": json.loads((CACHE / "manifest.json").read_text())["preprocess_fingerprint"],
        "site_index": site,
        "station_id": str(static["station_id"][site]),
        "station_name": str(static["station_name"][site]),
        "seed": seed,
        "epoch": epoch,
        "val_metrics": {"station_mae_cf": val_mae},
        "checkpoint_metric": "station_mae_cf",
        "model_state": dict(state),
        "model_state_sha256": tensor_state_hash(state),
        "protocol": TRAINING_PROTOCOL,
        "independence_contract": {
            "station_count": 1,
            "learned_parameter_sharing_across_sites": False,
            "learned_parameter_sharing_across_seeds": False,
            "cluster_history_source": "same target site history",
            "other_site_power_access": False,
            "global_fixed_static_normalization": True,
        },
    }


def run_job(
    site: int,
    seed: int,
    static: Mapping[str, np.ndarray],
    device: torch.device,
    args,
) -> None:
    output = OUT / "independent_seed" / f"site_{site:02d}" / f"seed_{seed}"
    success = output / "SUCCESS.json"
    if success.exists() and not args.overwrite:
        print(json.dumps({"site": site, "seed": seed, "status": "skip_complete"}), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "START.json",
        {
            "site_index": site,
            "station_id": str(static["station_id"][site]),
            "station_name": str(static["station_name"][site]),
            "seed": seed,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "started": now(),
        },
    )
    seed_all(seed)
    baseline_train = SiteSplit("train", site, False)
    baseline_val = SiteSplit("val", site, False)
    baseline_test = SiteSplit("test", site, False)
    baseline = build_baseline(site, seed, static)
    baseline_state, baseline_history, baseline_best, baseline_epoch, baseline_peak = train_model(
        baseline,
        baseline_train,
        baseline_val,
        device,
        False,
        seed,
        args.epochs,
        args.patience,
        args.max_train_batches,
        args.max_val_batches,
    )
    baseline.load_state_dict(baseline_state, strict=True)
    baseline.to(device)
    baseline_val_mae = evaluate(baseline, baseline_val, device, False, args.max_val_batches)
    baseline_test_mae = evaluate(baseline, baseline_test, device, False, args.max_eval_batches)
    atomic_torch_save(
        output / "baseline_best.pt",
        checkpoint_payload(
            baseline_state, BASELINE_CONFIG, site, seed, "independent_history", baseline_epoch, baseline_best, static
        ),
    )
    atomic_json(output / "baseline_history.json", {"history": baseline_history})
    del baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    correction_train = SiteSplit("train", site, True)
    correction_val = SiteSplit("val", site, True)
    correction_test = SiteSplit("test", site, True)
    cbh = build_cbh(site, seed, static, baseline_state)
    frozen_before = tensor_state_hash(
        {name[len("baseline_model.") :]: value for name, value in cbh.state_dict().items() if name.startswith("baseline_model.")}
    )
    cbh_state, cbh_history, cbh_best, cbh_epoch, cbh_peak = train_model(
        cbh,
        correction_train,
        correction_val,
        device,
        True,
        seed,
        args.epochs,
        args.patience,
        args.max_train_batches,
        args.max_val_batches,
    )
    frozen_after = tensor_state_hash(
        {name[len("baseline_model.") :]: value for name, value in cbh_state.items() if name.startswith("baseline_model.")}
    )
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen history baseline changed")
    cbh.load_state_dict(cbh_state, strict=True)
    cbh.to(device)
    cbh_val_mae = evaluate(cbh, correction_val, device, True, args.max_val_batches)
    cbh_test_mae = evaluate(cbh, correction_test, device, True, args.max_eval_batches)
    cbh_payload = checkpoint_payload(
        cbh_state, CBH_CONFIG, site, seed, "independent_cbh", cbh_epoch, cbh_best, static
    )
    cbh_payload["frozen_baseline_sha256"] = frozen_after
    atomic_torch_save(output / "cbh_best.pt", cbh_payload)
    atomic_json(output / "cbh_history.json", {"history": cbh_history})
    metrics = {
        "site_index": site,
        "station_id": str(static["station_id"][site]),
        "station_name": str(static["station_name"][site]),
        "capacity_mw": float(static["station_capacity_mw"][site]),
        "seed": seed,
        "baseline": {
            "best_epoch": baseline_epoch,
            "val_mae_cf": baseline_val_mae,
            "test_mae_cf": baseline_test_mae,
            "model_state_sha256": tensor_state_hash(baseline_state),
            "peak_reserved_mib": baseline_peak,
        },
        "cbh": {
            "best_epoch": cbh_epoch,
            "val_mae_cf": cbh_val_mae,
            "test_mae_cf": cbh_test_mae,
            "model_state_sha256": tensor_state_hash(cbh_state),
            "frozen_baseline_sha256_before": frozen_before,
            "frozen_baseline_sha256_after": frozen_after,
            "peak_reserved_mib": cbh_peak,
        },
        "completed": now(),
    }
    atomic_json(output / "metrics.json", metrics)
    atomic_json(
        success,
        {
            "passed": True,
            "site_index": site,
            "seed": seed,
            "frozen_baseline_unchanged": frozen_before == frozen_after,
            "completed": now(),
        },
    )
    print(
        json.dumps(
            {
                "site": site,
                "seed": seed,
                "status": "complete",
                "baseline_test": baseline_test_mae,
                "cbh_test": cbh_test_mae,
            }
        ),
        flush=True,
    )


def assigned_jobs(worker_id: int, worker_count: int) -> list[tuple[int, int]]:
    jobs = [(site, seed) for site in range(56) for seed in SEEDS]
    return [job for index, job in enumerate(jobs) if index % worker_count == worker_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--worker-count", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (args.site is None) != (args.seed is None):
        raise ValueError("--site and --seed must be supplied together")
    if args.site is None and args.worker_id is None:
        raise ValueError("Supply a single job or --worker-id")
    if args.site is not None and args.worker_id is not None:
        raise ValueError("Single-job and worker modes are mutually exclusive")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    audit = OUT / "status/LOCAL_WEATHER_AUDIT.json"
    if not audit.exists() or not json.loads(audit.read_text())["passed"]:
        raise RuntimeError("Local weather cache is not audited")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    static = load_static()
    jobs = (
        [(args.site, args.seed)]
        if args.site is not None
        else assigned_jobs(args.worker_id, args.worker_count)
    )
    worker_errors = []
    for site, seed in jobs:
        try:
            run_job(site, seed, static, device, args)
        except Exception as error:
            failure = {
                "site": site,
                "seed": seed,
                "worker_id": args.worker_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "failed": now(),
            }
            atomic_json(
                OUT / "status/failures" / f"site_{site:02d}_seed_{seed}.json",
                failure,
            )
            worker_errors.append(failure)
            print(json.dumps({"status": "failed", **failure}, ensure_ascii=False), flush=True)
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if args.worker_id is not None:
        atomic_json(
            OUT / "status/workers" / f"worker_{args.worker_id:02d}.json",
            {
                "worker_id": args.worker_id,
                "worker_count": args.worker_count,
                "jobs": [{"site": site, "seed": seed} for site, seed in jobs],
                "errors": worker_errors,
                "completed": now(),
            },
        )
    if worker_errors and args.site is not None:
        raise RuntimeError(worker_errors[0]["error"])


if __name__ == "__main__":
    main()
