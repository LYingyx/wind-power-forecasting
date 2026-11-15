"""Fine-tune exact selected-station CBH paths with linear or generated proxy weather."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, stack_module_state, vmap

from cbh_adapter import CBHAdapterSR
from paths import ROOT, V2
from artifact_utils import dump_json, sha256


sys.path.insert(0, str(V2 / "code"))
from common import Data, STATIONS  # noqa: E402
from train_sr import proxy_eval  # noqa: E402


METHODS = ["LINEAR", "DET", "ICFM", "VPCFM", "SINE_SI", "VP_EPS", "VP_V", "EDM", "MEANFLOW"]
SEEDS = list(range(42, 52))


class Ensemble:
    def __init__(self, models: list[torch.nn.Module], device: str):
        self.models = models
        self.template = copy.deepcopy(models[0]).to("meta")
        parameters, buffers = stack_module_state(models)
        self.parameters = {
            key: value.detach().to(device).requires_grad_(value.requires_grad)
            for key, value in parameters.items()
        }
        self.buffers = {key: value.to(device) for key, value in buffers.items()}

    def __call__(self, *args: torch.Tensor, train: bool = False):
        self.template.train(train)
        return vmap(
            lambda parameters, buffers, *values: functional_call(
                self.template, (parameters, buffers), values
            ),
            in_dims=(0, 0, *([0] * len(args))),
            randomness="different",
        )(self.parameters, self.buffers, *args)

    def state(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: value[index].detach().cpu().clone()
            for key, value in (self.parameters | self.buffers).items()
        }


class CBHData:
    def __init__(self, split: str, station: int, seeds: list[int], device: str):
        self.raw = Data(split, station, device)
        paths = [ROOT / "cache/baseline" / f"{split}_seed{seed}.npy" for seed in seeds]
        values = np.stack([np.load(path)[:, station] for path in paths])
        self.baseline = torch.tensor(values, device=device)
        self.h, self.y, self.w = self.raw.h, self.raw.y, self.raw.w
        self.cycle, self.times, self.phase = self.raw.cycle, self.raw.times, self.raw.phase
        self.n = self.raw.n

    def batch(self, index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        baseline = torch.stack(
            [self.baseline[row, index[row]] for row in range(len(index))]
        )
        return (*self.raw.batch(index), baseline)


def noise(shape: tuple[int, ...], device: str, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, device=device, generator=generator)


def generator_initialization(method: str, station: int, seeds: list[int]):
    if method == "LINEAR":
        return torch.ones(14), [None] * len(seeds), [None] * len(seeds), [None] * len(seeds)
    scales, states, histories, hashes = [], [], [], []
    for seed in seeds:
        path = V2 / "results" / f"TiDE_SR_{method}" / STATIONS[station] / f"seed{seed}"
        checkpoint_path = path / "pretrained_generator.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["method"] == method and checkpoint["seed"] == seed
        assert checkpoint["station_id"] == STATIONS[station]
        scales.append(checkpoint["scale"])
        states.append(checkpoint["model_state"])
        histories.append(json.loads((path / "pretrain_history.json").read_text()))
        hashes.append(sha256(checkpoint_path))
    assert all(torch.equal(scales[0], value) for value in scales[1:])
    return scales[0], states, histories, hashes


@torch.no_grad()
def evaluate_samples(
    ensemble: Ensemble,
    data: CBHData,
    indices: torch.Tensor,
    method: str,
    samples: int,
    batch_size: int = 128,
) -> torch.Tensor:
    group = len(ensemble.models)
    device = data.h.device
    random = torch.Generator(device="cpu").manual_seed(20260917)
    bank = torch.randn(samples, data.n, 6, 3, 14, generator=random)
    result = []
    ensemble.template.eval()
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        index = selected.expand(group, -1)
        size = index.shape[1]
        zero = torch.zeros(group, size, 3, 3, 14, device=device)
        zero_time = torch.zeros(group, size, 3, device=device)
        rows = []
        for sample in range(samples):
            generated_noise = bank[sample, selected.cpu()].to(device)[None].expand(group, -1, -1, -1, -1)
            prediction, _ = ensemble(
                *data.batch(index), generated_noise, zero, zero_time, zero, train=False
            )
            rows.append(prediction.cpu())
        result.append(torch.stack(rows, 2))
    return torch.cat(result, 1)


@torch.no_grad()
def initial_equivalence(
    ensemble: Ensemble,
    data: CBHData,
    indices: torch.Tensor,
    station: int,
    seeds: list[int],
) -> list[float]:
    predictions = evaluate_samples(ensemble, data, indices, "LINEAR", 1)[:, :, 0]
    errors = []
    for row, seed in enumerate(seeds):
        original = np.load(ROOT / "original_eval" / f"seed{seed}" / "predictions_fp32.npy")[: len(predictions[row]), station]
        errors.append(float(np.max(np.abs(predictions[row].numpy() - original))))
    return errors


def run(method: str, station: int, seeds: list[int], device: str, epochs: int, smoke: bool) -> None:
    started = time.time()
    protocol = json.loads((V2 / "configs/protocol.json").read_text())
    family = f"CBH_ADAPT_{method}"
    result_root = ROOT / ("tmp/smoke" if smoke else "results") / family / STATIONS[station]
    if not smoke and all((result_root / f"seed{seed}" / "metrics.json").exists() for seed in seeds):
        return
    train = CBHData("train", station, seeds, device)
    validation = CBHData("val", station, seeds, device)
    stop = torch.tensor(np.load(V2 / "data/val_stop_indices.npy"), device=device)
    report = torch.tensor(np.load(V2 / "data/val_report_indices.npy"), device=device)
    if smoke:
        train.n = 64
        stop = stop[:32]
        report = report[:32]
    scale, generator_states, pretrain_history, pretrain_hashes = generator_initialization(method, station, seeds)
    models = [CBHAdapterSR(method, scale, seed, station_index=json.loads((V2 / "data/selected_farms.json").read_text())["stations"][station]["station_index"]) for seed in seeds]
    if method != "LINEAR":
        for model, state in zip(models, generator_states):
            assert model.generator is not None
            model.generator.load_state_dict(state)
    ensemble = Ensemble(models, device)
    equivalence = initial_equivalence(ensemble, validation, report, station, seeds) if method == "LINEAR" else [None] * len(seeds)
    if method == "LINEAR" and max(equivalence) > 2e-5:
        raise RuntimeError(f"Initial CBH specialization mismatch: {max(equivalence)}")

    group = len(seeds)
    optimizer = torch.optim.AdamW(
        [value for value in ensemble.parameters.values() if value.requires_grad],
        lr=protocol["lr"],
        weight_decay=protocol["weight_decay"],
    )
    active = np.ones(group, bool)
    best = np.full(group, np.inf)
    patience = np.zeros(group, int)
    best_epoch = np.zeros(group, int)
    states = [None] * group
    histories = [[] for _ in seeds]
    random = [torch.Generator(device=device).manual_seed(seed + 27021) for seed in seeds]

    for epoch in range(1, epochs + 1):
        permutation = torch.stack(
            [torch.randperm(train.n, device=device, generator=item) for item in random]
        )
        total = torch.zeros(group, device=device)
        auxiliary_total = torch.zeros(group, device=device)
        for begin in range(0, train.n, protocol["batch_size"]):
            index = permutation[:, begin : begin + protocol["batch_size"]]
            size = index.shape[1]
            generated_noise = torch.stack(
                [noise((size, 6, 3, 14), device, item) for item in random]
            )
            auxiliary_noise = torch.stack(
                [noise((size, 3, 3, 14), device, item) for item in random]
            )
            auxiliary_extra = torch.stack(
                [noise((size, 3, 3, 14), device, item) for item in random]
            )
            auxiliary_time = torch.stack(
                [torch.rand(size, 3, device=device, generator=item) * 0.998 + 0.001 for item in random]
            )
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary = ensemble(
                *train.batch(index),
                generated_noise,
                auxiliary_noise,
                auxiliary_time,
                auxiliary_extra,
                train=True,
            )
            main = (prediction - train.y[index]).abs().mean((1, 2))
            loss = main + 0.1 * auxiliary
            if not torch.isfinite(loss).all():
                raise RuntimeError(f"Non-finite {family} station={station} epoch={epoch}")
            (loss * torch.tensor(active, device=device)).sum().backward()
            optimizer.step()
            total += main.detach() * size
            auxiliary_total += auxiliary.detach() * size

        stop_samples = evaluate_samples(
            ensemble, validation, stop, method, 1 if method in {"LINEAR", "DET"} else 5
        )
        stop_prediction = stop_samples.median(2).values
        stop_mae = (stop_prediction - validation.y[stop].cpu()).abs().mean((1, 2)).numpy()
        for row, seed in enumerate(seeds):
            if not active[row]:
                continue
            histories[row].append(
                {
                    "epoch": epoch,
                    "train_mae": float(total[row] / train.n),
                    "auxiliary_loss": float(auxiliary_total[row] / train.n),
                    "val_stop_mae": float(stop_mae[row]),
                }
            )
            if stop_mae[row] < best[row]:
                best[row] = stop_mae[row]
                patience[row] = 0
                best_epoch[row] = epoch
                states[row] = ensemble.state(row)
            else:
                patience[row] += 1
            if patience[row] >= protocol["patience"]:
                active[row] = False
        if epoch % 5 == 0 or not active.any():
            print(
                family,
                STATIONS[station],
                "epoch",
                epoch,
                "active",
                int(active.sum()),
                "stop",
                float(best.mean()),
                "seconds",
                round(time.time() - started, 1),
                flush=True,
            )
        if not active.any():
            break

    assert all(state is not None for state in states)
    for key in ensemble.parameters:
        ensemble.parameters[key].data.copy_(torch.stack([state[key] for state in states]).to(device))
    for key in ensemble.buffers:
        ensemble.buffers[key].copy_(torch.stack([state[key] for state in states]).to(device))
    sample_count = 1 if method in {"LINEAR", "DET"} else 5
    samples = evaluate_samples(ensemble, validation, report, method, sample_count)
    prediction = samples.median(2).values
    error = prediction - validation.y[report].cpu()
    target = validation.y[report].cpu()[None, :, None]
    crps = (samples - target).abs().mean((1, 2, 3)) - (
        samples[:, :, :, None] - samples[:, :, None, :]
    ).abs().mean((1, 2, 3, 4)) / 2

    for row, seed in enumerate(seeds):
        out = result_root / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": family,
            "station_id": STATIONS[station],
            "seed": seed,
            "model_state": states[row],
            "best_epoch": int(best_epoch[row]),
            "source_cbh_checkpoint_sha256": models[row].predictor.source_checkpoint_sha256,
            "protocol_sha256": sha256(V2 / "configs/protocol.json"),
        }
        torch.save(checkpoint, out / "best_model.pt")
        np.save(out / "predictions_report.npy", prediction[row].numpy())
        np.save(out / "samples_report.npy", samples[row].numpy())
        dump_json(out / "history.json", histories[row])
        if method == "LINEAR":
            proxy = None
        else:
            model = models[row].to(device)
            model.load_state_dict(states[row])
            model.eval()
            after = proxy_eval(model, validation.w[torch.unique(validation.cycle[report])])
            assert model.generator is not None
            model.generator.load_state_dict(generator_states[row])
            before = proxy_eval(model, validation.w[torch.unique(validation.cycle[report])])
            proxy = {"before_joint": before, "after_joint": after}
            model.cpu()
        dump_json(out / "proxy_metrics.json", proxy)
        baseline_path = ROOT / "cache/baseline" / f"val_seed{seed}.npy"
        metrics = {
            "model": family,
            "station_id": STATIONS[station],
            "seed": seed,
            "report_mae": float(error[row].abs().mean()),
            "report_rmse": float(error[row].square().mean().sqrt()),
            "report_crps": float(crps[row]),
            "sample_std": float(samples[row].std(1, unbiased=False).mean()),
            "val_stop_mae": float(best[row]),
            "best_epoch": int(best_epoch[row]),
            "trained_epochs": len(histories[row]),
            "report_n": len(report),
            "parameters": sum(parameter.numel() for parameter in models[row].parameters()),
            "mapping_initialization": "original frozen CBH checkpoint selected on full 2023 validation",
            "mapping_finetune_data": "same 10% 2022 pilot as temporal SR experiments",
            "source_cbh_checkpoint_sha256": models[row].predictor.source_checkpoint_sha256,
            "source_cbh_checkpoint_epoch": models[row].predictor.source_checkpoint_epoch,
            "baseline_cache_sha256": sha256(baseline_path),
            "generator_pretrain_sha256": pretrain_hashes[row],
            "initial_original_equivalence_max_error": equivalence[row],
            "samples_per_prediction": sample_count,
            "precision": "FP32",
            "protocol_sha256": sha256(V2 / "configs/protocol.json"),
            "checkpoint_sha256": sha256(out / "best_model.pt"),
            "group_wall_seconds": time.time() - started,
        }
        dump_json(out / "metrics.json", metrics)
    print(
        "COMPLETE",
        family,
        STATIONS[station],
        error.abs().mean((1, 2)).tolist(),
        round(time.time() - started, 1),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--stations", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    for method in args.methods:
        for station in args.stations:
            run(method, station, args.seeds, args.device, 2 if args.smoke else args.epochs, args.smoke)


if __name__ == "__main__":
    main()
