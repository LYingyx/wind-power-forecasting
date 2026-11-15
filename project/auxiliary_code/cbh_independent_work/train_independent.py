"""Train fully independent single-site history baselines and CBH corrections.

Ten seeds are vectorized with torch.func.  Parameter slices, optimizer moments,
dropout draws, shuffled sample orders and validation checkpoint choices remain
independent.  No learned tensor is shared between sites or seeds.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

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
    FunctionalEnsemble,
    atomic_json,
    atomic_torch_save,
    build_baseline,
    build_cbh,
    ensemble_from_states,
    load_static,
    sha256_file,
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
        self.origin = np.load(root / "origin_unix.npy", mmap_mode="r")
        self.sample_index = np.load(root / "sample_index.npy", mmap_mode="r")
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
        if include_weather and self.local_weather.shape[1:] != (56, 8, 14):
            raise RuntimeError(f"Unexpected local weather: {self.local_weather.shape}")

    def arrays(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        h = np.asarray(self.history[indices, self.site : self.site + 1], dtype=np.float32)
        y = np.asarray(self.future[indices, self.site : self.site + 1], dtype=np.float32)
        return {
            "station_history": h,
            # The independent model's global-history branch receives only the
            # target site's own history.  No other site's power is read.
            "cluster_history": h[:, 0].copy(),
            "station_future": y,
            "origin_phase": np.asarray(self.phase[indices], dtype=np.int64),
            "mask": np.ones(len(indices), dtype=np.bool_),
        }

    def weather_arrays(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        if self.local_weather is None:
            raise RuntimeError("Weather was not enabled")
        cycles = np.asarray(self.weather_cycle[indices], dtype=np.int64)
        unique, inverse = np.unique(cycles, return_inverse=True)
        weather = np.asarray(self.local_weather[unique, self.site], dtype=np.float32)
        return {
            "weather": weather[..., None, None],
            "weather_batch_index": inverse.astype(np.int64),
            "weather_time_offset_hours": np.asarray(self.weather_offsets[indices], dtype=np.float32),
        }


def baseline_orders(length: int, seeds: Sequence[int], epochs: int) -> list[list[np.ndarray]]:
    generators = []
    for seed in seeds:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        generators.append(generator)
    result = []
    for _ in range(epochs):
        epoch = []
        for generator in generators:
            order = torch.randperm(length, generator=generator).numpy()
            epoch.append(order)
        result.append(epoch)
    return result


def cycle_batches(cycle_index: np.ndarray, cycles_per_batch: int, shuffle: bool, seed: int, epoch: int) -> list[np.ndarray]:
    values = np.asarray(cycle_index, dtype=np.int64)
    unique = np.unique(values)
    members = {int(cycle): np.flatnonzero(values == cycle).astype(np.int64) for cycle in unique}
    cycles = unique.copy()
    rng = np.random.default_rng(int(seed) + int(epoch))
    if shuffle:
        rng.shuffle(cycles)
    batches = []
    for start in range(0, len(cycles), cycles_per_batch):
        selected = cycles[start : start + cycles_per_batch]
        batch = np.concatenate([members[int(cycle)] for cycle in selected])
        if shuffle:
            rng.shuffle(batch)
        batches.append(batch)
    return batches


def _pad(values: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        raise ValueError("Cannot pad an empty batch")
    if len(values) > size:
        raise ValueError("Batch exceeds pad size")
    mask = np.zeros(size, dtype=np.bool_)
    mask[: len(values)] = True
    if len(values) == size:
        return values, mask
    return np.concatenate((values, np.repeat(values[-1:], size - len(values)))), mask


def stacked_seed_batch(
    data: SiteSplit,
    indices_by_seed: Sequence[np.ndarray],
    device: torch.device,
    include_weather: bool,
) -> dict[str, torch.Tensor]:
    size = max(map(len, indices_by_seed))
    records = []
    for indices in indices_by_seed:
        padded, mask = _pad(np.asarray(indices, dtype=np.int64), size)
        record = data.arrays(padded)
        record["mask"] = mask
        if include_weather:
            record.update(data.weather_arrays(padded))
        records.append(record)
    result: dict[str, torch.Tensor] = {}
    for name in ("station_history", "cluster_history", "station_future", "origin_phase", "mask"):
        array = np.stack([record[name] for record in records])
        result[name] = torch.from_numpy(array).to(device, non_blocking=True)
    if include_weather:
        max_cycles = max(record["weather"].shape[0] for record in records)
        weather_rows = []
        for record in records:
            weather = record["weather"]
            if len(weather) < max_cycles:
                weather = np.concatenate((weather, np.repeat(weather[-1:], max_cycles - len(weather), axis=0)))
            weather_rows.append(weather)
        result["weather"] = torch.from_numpy(np.stack(weather_rows)).to(device, non_blocking=True)
        result["weather_batch_index"] = torch.from_numpy(
            np.stack([record["weather_batch_index"] for record in records])
        ).to(device, non_blocking=True)
        result["weather_time_offset_hours"] = torch.from_numpy(
            np.stack([record["weather_time_offset_hours"] for record in records])
        ).to(device, non_blocking=True)
    return result


def shared_eval_batch(
    data: SiteSplit,
    indices: np.ndarray,
    ensemble_size: int,
    device: torch.device,
    include_weather: bool,
) -> dict[str, torch.Tensor]:
    record = data.arrays(indices)
    if include_weather:
        record.update(data.weather_arrays(indices))
    result = {}
    for name in ("station_history", "cluster_history", "station_future", "origin_phase", "mask"):
        tensor = torch.from_numpy(record[name]).to(device, non_blocking=True)
        result[name] = tensor.unsqueeze(0).expand((ensemble_size,) + tensor.shape)
    if include_weather:
        for name in ("weather", "weather_batch_index", "weather_time_offset_hours"):
            tensor = torch.from_numpy(record[name]).to(device, non_blocking=True)
            result[name] = tensor.unsqueeze(0).expand((ensemble_size,) + tensor.shape)
    return result


def autocast(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def masked_losses(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # prediction/target [E,B,1,24], mask [E,B]
    values = (prediction.float() - target.float()).abs()
    weights = mask[:, :, None, None].to(values)
    return (values * weights).sum(dim=(1, 2, 3)) / (weights.sum(dim=(1, 2, 3)) * 24.0)


@torch.no_grad()
def evaluate(
    ensemble: FunctionalEnsemble,
    data: SiteSplit,
    include_weather: bool,
    max_batches: int | None = None,
) -> list[float]:
    ensemble.train(False)
    if include_weather:
        batches = cycle_batches(data.weather_cycle, 24, False, 0, 0)
    else:
        batches = [np.arange(start, min(data.length, start + 96), dtype=np.int64) for start in range(0, data.length, 96)]
    total = torch.zeros(ensemble.size, device=ensemble.device, dtype=torch.float64)
    count = torch.zeros_like(total)
    for batch_number, indices in enumerate(batches):
        if max_batches is not None and batch_number >= max_batches:
            break
        batch = shared_eval_batch(data, indices, ensemble.size, ensemble.device, include_weather)
        with autocast(ensemble.device):
            output = ensemble(batch)
        error = (output["station_factor"].float() - batch["station_future"].float()).abs()
        total += error.double().sum(dim=(1, 2, 3))
        count += error.shape[1] * error.shape[2] * error.shape[3]
    if not bool((count > 0).all()):
        raise RuntimeError("Evaluation produced no samples")
    return (total / count).cpu().tolist()


def train_ensemble(
    ensemble: FunctionalEnsemble,
    train: SiteSplit,
    val: SiteSplit,
    include_weather: bool,
    seeds: Sequence[int],
    epochs: int,
    patience: int,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> tuple[list[dict[str, torch.Tensor]], list[dict], list[float], list[int]]:
    optimizer = torch.optim.AdamW(
        ensemble.optimizer_parameters(),
        lr=float(TRAINING_PROTOCOL["learning_rate"]),
        weight_decay=float(TRAINING_PROTOCOL["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best = np.full(len(seeds), np.inf, dtype=np.float64)
    best_epoch = np.zeros(len(seeds), dtype=np.int64)
    stale = np.zeros(len(seeds), dtype=np.int64)
    active = np.ones(len(seeds), dtype=np.bool_)
    best_states: list[dict[str, torch.Tensor] | None] = [None] * len(seeds)
    history: list[dict] = []
    random_orders = baseline_orders(train.length, seeds, epochs) if not include_weather else None
    for epoch in range(epochs):
        started = time.perf_counter()
        ensemble.train(True)
        if include_weather:
            by_seed = [cycle_batches(train.weather_cycle, 12, True, seed, epoch) for seed in seeds]
        else:
            assert random_orders is not None
            by_seed = [
                [order[start : start + 48] for start in range(0, len(order), 48)]
                for order in random_orders[epoch]
            ]
        batch_count = min(map(len, by_seed))
        if len({len(value) for value in by_seed}) != 1:
            raise RuntimeError("Seed batch counts differ")
        sum_loss = np.zeros(len(seeds), dtype=np.float64)
        sum_count = np.zeros(len(seeds), dtype=np.int64)
        max_norm = np.zeros(len(seeds), dtype=np.float64)
        for batch_number in range(batch_count):
            if max_train_batches is not None and batch_number >= max_train_batches:
                break
            indices = [value[batch_number] for value in by_seed]
            batch = stacked_seed_batch(train, indices, ensemble.device, include_weather)
            optimizer.zero_grad(set_to_none=True)
            with autocast(ensemble.device):
                output = ensemble(batch)
                losses = masked_losses(output["station_factor"], batch["station_future"], batch["mask"])
                loss = losses.sum()
            loss.backward()
            norms = ensemble.clip_grad_norm_per_model(float(TRAINING_PROTOCOL["gradient_clip"]))
            optimizer.step()
            counts = batch["mask"].sum(1).cpu().numpy()
            sum_loss += losses.detach().cpu().numpy() * counts
            sum_count += counts
            max_norm = np.maximum(max_norm, norms.detach().cpu().numpy())
        scheduler.step()
        train_mae = sum_loss / sum_count
        val_mae = np.asarray(evaluate(ensemble, val, include_weather, max_val_batches), dtype=np.float64)
        improved = (val_mae < best) & active
        for index in np.flatnonzero(improved):
            best[index] = val_mae[index]
            best_epoch[index] = epoch + 1
            stale[index] = 0
            best_states[index] = ensemble.state_for(int(index))
        for index in np.flatnonzero(active & ~improved):
            stale[index] += 1
        active[(stale >= patience) & active] = False
        history.append(
            {
                "epoch": epoch + 1,
                "train_mae_cf": train_mae.tolist(),
                "val_mae_cf": val_mae.tolist(),
                "best_val_mae_cf": best.tolist(),
                "best_epoch": best_epoch.tolist(),
                "stale": stale.tolist(),
                "active": active.tolist(),
                "improved": improved.tolist(),
                "maximum_preclip_gradient_norm": max_norm.tolist(),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "kind": ensemble.kind,
                    "train_mean": float(train_mae.mean()),
                    "val_mean": float(val_mae.mean()),
                    "best_mean": float(best.mean()),
                    "active": int(active.sum()),
                    "seconds": history[-1]["seconds"],
                }
            ),
            flush=True,
        )
        if not active.any():
            break
    if any(state is None for state in best_states):
        raise RuntimeError("One or more seeds never produced a checkpoint")
    return [state for state in best_states if state is not None], history, best.tolist(), best_epoch.tolist()


def checkpoint_payload(
    state: Mapping[str, torch.Tensor],
    config: Mapping,
    site: int,
    seed: int,
    stage: str,
    epoch: int,
    val_mae: float,
    static: Mapping[str, np.ndarray],
) -> dict:
    return {
        "stage": stage,
        "model_config": dict(config),
        "data_manifest_identity": json.loads((CACHE / "manifest.json").read_text())["preprocess_fingerprint"],
        "site_index": site,
        "station_id": str(static["station_id"][site]),
        "station_name": str(static["station_name"][site]),
        "seed": int(seed),
        "epoch": int(epoch),
        "val_metrics": {"station_mae_cf": float(val_mae)},
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


def run_site(args, site: int, static: Mapping[str, np.ndarray], device: torch.device) -> None:
    site_dir = OUT / "independent" / f"site_{site:02d}"
    success = site_dir / "SUCCESS.json"
    if success.exists() and not args.overwrite:
        print(json.dumps({"site": site, "status": "skip_complete"}), flush=True)
        return
    site_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        site_dir / "START.json",
        {
            "site": site,
            "station_id": str(static["station_id"][site]),
            "station_name": str(static["station_name"][site]),
            "seeds": list(args.seeds),
            "device": str(device),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": now(),
        },
    )
    baseline_train = SiteSplit("train", site, False)
    baseline_val = SiteSplit("val", site, False)
    baseline_test = SiteSplit("test", site, False)
    templates = [build_baseline(site, seed, static) for seed in args.seeds]
    ensemble = FunctionalEnsemble(templates, device, "baseline")
    baseline_states, baseline_history, baseline_best, baseline_epochs = train_ensemble(
        ensemble,
        baseline_train,
        baseline_val,
        False,
        args.seeds,
        args.epochs,
        args.patience,
        args.max_train_batches,
        args.max_val_batches,
    )
    del ensemble
    torch.cuda.empty_cache()
    best_templates = [build_baseline(site, seed, static) for seed in args.seeds]
    best_baseline = ensemble_from_states(best_templates, baseline_states, device, "baseline")
    baseline_val_metrics = evaluate(best_baseline, baseline_val, False, args.max_val_batches)
    baseline_test_metrics = evaluate(best_baseline, baseline_test, False, args.max_eval_batches)
    for index, seed in enumerate(args.seeds):
        payload = checkpoint_payload(
            baseline_states[index], BASELINE_CONFIG, site, seed, "independent_history", baseline_epochs[index], baseline_best[index], static
        )
        atomic_torch_save(site_dir / f"baseline_seed{seed}.pt", payload)
    atomic_json(site_dir / "baseline_history.json", {"seeds": list(args.seeds), "epochs": baseline_history})
    del best_baseline, best_templates, templates
    torch.cuda.empty_cache()

    correction_train = SiteSplit("train", site, True)
    correction_val = SiteSplit("val", site, True)
    correction_test = SiteSplit("test", site, True)
    cbh_templates = [build_cbh(site, seed, static, baseline_states[index]) for index, seed in enumerate(args.seeds)]
    baseline_hash_before = [tensor_state_hash({k[len("baseline_model."):]: v for k, v in model.state_dict().items() if k.startswith("baseline_model.")}) for model in cbh_templates]
    cbh = FunctionalEnsemble(cbh_templates, device, "cbh")
    cbh_states, cbh_history, cbh_best, cbh_epochs = train_ensemble(
        cbh,
        correction_train,
        correction_val,
        True,
        args.seeds,
        args.epochs,
        args.patience,
        args.max_train_batches,
        args.max_val_batches,
    )
    del cbh
    torch.cuda.empty_cache()
    cbh_best_templates = [build_cbh(site, seed, static, baseline_states[index]) for index, seed in enumerate(args.seeds)]
    best_cbh = ensemble_from_states(cbh_best_templates, cbh_states, device, "cbh")
    cbh_val_metrics = evaluate(best_cbh, correction_val, True, args.max_val_batches)
    cbh_test_metrics = evaluate(best_cbh, correction_test, True, args.max_eval_batches)
    baseline_hash_after = []
    for state in cbh_states:
        baseline_hash_after.append(tensor_state_hash({k[len("baseline_model."):]: v for k, v in state.items() if k.startswith("baseline_model.")}))
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("Frozen baseline changed during CBH training")
    for index, seed in enumerate(args.seeds):
        payload = checkpoint_payload(
            cbh_states[index], CBH_CONFIG, site, seed, "independent_cbh", cbh_epochs[index], cbh_best[index], static
        )
        payload["frozen_baseline_sha256"] = baseline_hash_after[index]
        atomic_torch_save(site_dir / f"cbh_seed{seed}.pt", payload)
    atomic_json(site_dir / "cbh_history.json", {"seeds": list(args.seeds), "epochs": cbh_history})
    result = {
        "site_index": site,
        "station_id": str(static["station_id"][site]),
        "station_name": str(static["station_name"][site]),
        "capacity_mw": float(static["station_capacity_mw"][site]),
        "seeds": list(args.seeds),
        "baseline": {
            "best_epoch": baseline_epochs,
            "val_mae_cf": baseline_val_metrics,
            "test_mae_cf": baseline_test_metrics,
            "state_sha256": [tensor_state_hash(state) for state in baseline_states],
        },
        "cbh": {
            "best_epoch": cbh_epochs,
            "val_mae_cf": cbh_val_metrics,
            "test_mae_cf": cbh_test_metrics,
            "state_sha256": [tensor_state_hash(state) for state in cbh_states],
            "frozen_baseline_sha256_before": baseline_hash_before,
            "frozen_baseline_sha256_after": baseline_hash_after,
        },
        "completed": now(),
    }
    atomic_json(site_dir / "metrics.json", result)
    atomic_json(
        success,
        {
            "passed": True,
            "site_index": site,
            "station_id": result["station_id"],
            "seeds": list(args.seeds),
            "frozen_baseline_unchanged": baseline_hash_before == baseline_hash_after,
            "completed": now(),
        },
    )
    print(json.dumps({"site": site, "status": "complete", "cbh_test_mean": float(np.mean(cbh_test_metrics))}), flush=True)


def parse_sites(value: str) -> list[int]:
    if value == "all":
        return list(range(56))
    result = []
    for part in value.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(part))
    if len(set(result)) != len(result) or any(site < 0 or site >= 56 for site in result):
        raise ValueError(f"Invalid sites: {value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default="all")
    parser.add_argument("--seeds", default="42-51")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.sites = parse_sites(args.sites)
    args.seeds = parse_sites(args.seeds.replace("42-51", "42,43,44,45,46,47,48,49,50,51"))
    # parse_sites validates site range; seeds need their own check after reuse.
    if args.seeds != list(SEEDS) and len(args.seeds) < 1:
        raise ValueError("At least one seed is required")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    local_audit = OUT / "status/LOCAL_WEATHER_AUDIT.json"
    if not local_audit.exists() or not json.loads(local_audit.read_text())["passed"]:
        raise RuntimeError("Run prepare_local_weather.py first")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda":
        # Fused SDPA backward in the installed torch build is not stride-safe
        # under a parameter vmap.  The math kernel has identical attention
        # semantics and supports the independent ensemble layout.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    static = load_static()
    for site in args.sites:
        run_site(args, site, static, device)


if __name__ == "__main__":
    main()
