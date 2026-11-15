from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Iterable, Sequence

import numpy as np

from launch_aggregate_cluster_experiments import (
    HIGH_MEMORY_MIB,
    HORIZON_KEYS,
    WorkerSlot,
    _looks_like_oom,
    atomic_json,
    build_worker_slots,
    execute_parallel,
    gpu_state,
    parse_slot_counts,
)
from launch_aggregate_strict_modulation import code_snapshot


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_weather_dynamics_v1"
R2_ROOT = PROJECT / "artifacts" / "aggregate_cluster_r2_state_v1" / "mechanism" / "r2_w"
RESPONSE_LATENT_ROOT = PROJECT / "artifacts" / "aggregate_response_state_v1" / "latent_cache"
STRICT_ROOT = PROJECT / "artifacts" / "aggregate_strict_modulation_v1"
TIME_CACHE = PROJECT / "data" / "aggregate_weather_dynamics_v1"
SCREEN_SEEDS = (42, 46, 50)
FORMAL_SEEDS = tuple(range(42, 52))
REMAINING_SEEDS = tuple(seed for seed in FORMAL_SEEDS if seed not in SCREEN_SEEDS)
CORE_GROUPS = ("m1", "m2", "m3", "m4a", "m4b", "m5", "m6a", "m6b")


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    training_stage: str
    initial_checkpoint: str = ""
    auxiliary_checkpoint: str = ""
    exclude_start: int | None = None
    exclude_stop: int | None = None
    partial: bool = False

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"

    @property
    def base_checkpoint(self) -> Path:
        return R2_ROOT / f"seed{self.seed}" / "best_model.pt"


def audited(run_dir: Path, split: str = "val", *, partial: bool = False) -> bool:
    marker = run_dir / ("COMPLETE" if split == "val" else "TEST_COMPLETE")
    if not marker.is_file():
        return False
    command = [
        os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
        str(PROJECT / "scripts" / "audit_aggregate_run.py"),
        "--run-dir", str(run_dir), "--split", split,
    ]
    if partial:
        command.append("--allow-partial")
    return subprocess.run(
        command, cwd=PROJECT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def a0_dir(seed: int) -> Path:
    stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
    return STRICT_ROOT / stage / "a0_ft" / f"seed{seed}"


def final_run(group: str, seed: int) -> Path:
    stage = "screen_joint" if seed in SCREEN_SEEDS else "formal_joint"
    return ROOT / stage / group / f"seed{seed}"


def metric(run: Path, key: str, split: str = "val") -> float:
    with (run / f"metrics_{split}.json").open(encoding="utf-8") as stream:
        return float(json.load(stream)[key])


def initial(stage: str, group: str, seed: int) -> str:
    path = ROOT / stage / group / f"seed{seed}" / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def auxiliary(group: str, seed: int) -> str:
    return initial("screen_aux_joint" if seed in SCREEN_SEEDS else "formal_aux_joint", group, seed)


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not audited(job.run_dir, partial=job.partial)]
    if not pending:
        print(f"[dynamics-scheduler] stage={label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, total_memory: int) -> int:
        cycles, accumulation = ((6, 2) if total_memory >= HIGH_MEMORY_MIB else (3, 4))
        environment = dict(os.environ)
        environment.update(
            WIND_DYNAMICS_ROOT=str(ROOT),
            WIND_AGG_CYCLES_PER_BATCH=str(cycles),
            WIND_AGG_GRAD_ACCUMULATION=str(accumulation),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}]",
        )
        if job.stage == "oof_envelope":
            environment.update(
                WIND_DYNAMICS_HEAD_EPOCHS="20",
                WIND_DYNAMICS_HEAD_PATIENCE="20",
                WIND_CHECKPOINT_SELECTION="last",
                WIND_AGG_MAX_VAL_BATCHES="1",
                WIND_TRAIN_EXCLUDE_START_UNIX=str(job.exclude_start),
                WIND_TRAIN_EXCLUDE_STOP_UNIX=str(job.exclude_stop),
                WIND_TRAIN_EXCLUSION_EMBARGO_SECONDS=str(6 * 3600),
            )
        command = [
            str(PROJECT / "shell" / "run_aggregate_weather_dynamics_job.sh"),
            str(slot.gpu), job.stage, job.group, str(job.seed),
            str(job.base_checkpoint), job.training_stage,
            job.initial_checkpoint, job.auxiliary_checkpoint,
        ]
        return subprocess.run(command, cwd=PROJECT, env=environment).returncode

    execute_parallel(
        label, pending, slots, is_heavy=lambda _job: False, runner=runner,
        item_label=lambda job: f"{job.stage}/{job.group}/seed{job.seed}",
        oom_detector=lambda job, code: _looks_like_oom(job.run_dir / "run.log", code),
        unit="train",
    )


def cache_response_latents(
    seeds: Sequence[int], splits: Sequence[str], slots: Sequence[WorkerSlot]
) -> None:
    items = [(seed, split) for seed in seeds for split in splits]
    cache_slots = tuple(slot for slot in slots if slot.slot == 0) or tuple(slots)

    def runner(item: tuple[int, str], slot: WorkerSlot, _total: int) -> int:
        seed, split = item
        output = RESPONSE_LATENT_ROOT / f"seed{seed}" / split
        output.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES=str(slot.gpu),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:response-latent:s{seed}:{split}]",
        )
        command = [
            "flock", "-x", str(output.parent / f".{split}.dynamics.lock"),
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "cache_aggregate_response_latents.py"),
            "--aggregate-cache", str(PROJECT / "data" / "aggregate_cluster_v1"),
            "--response-cache", str(PROJECT / "data" / "aggregate_response_state_v1"),
            "--checkpoint", str(R2_ROOT / f"seed{seed}" / "best_model.pt"),
            "--split", split, "--output-root", str(RESPONSE_LATENT_ROOT), "--device", "cuda",
        ]
        return subprocess.run(command, cwd=PROJECT, env=environment).returncode

    execute_parallel(
        f"response-latent-{'-'.join(splits)}", items, cache_slots,
        is_heavy=lambda _item: False, runner=runner,
        item_label=lambda item: f"response-latent/seed{item[0]}/{item[1]}", unit="cache",
    )


def cache_current_latents(
    seeds: Sequence[int], splits: Sequence[str], slots: Sequence[WorkerSlot]
) -> None:
    items = [(seed, split) for seed in seeds for split in splits]
    cache_slots = tuple(slot for slot in slots if slot.slot == 0) or tuple(slots)

    def runner(item: tuple[int, str], slot: WorkerSlot, _total: int) -> int:
        seed, split = item
        output = ROOT / "latent_cache" / f"seed{seed}" / split
        output.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=str(slot.gpu))
        return subprocess.run(
            [
                "flock", "-x", str(output.parent / f".{split}.lock"),
                os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                str(PROJECT / "scripts" / "cache_aggregate_dynamics_latents.py"),
                "--aggregate-cache", str(PROJECT / "data" / "aggregate_cluster_v1"),
                "--response-latent-cache", str(RESPONSE_LATENT_ROOT / f"seed{seed}"),
                "--dynamics-time-cache", str(TIME_CACHE),
                "--source-h5", str(PROJECT / "data" / "jilin_56farm_pad8.h5"),
                "--base-checkpoint", str(R2_ROOT / f"seed{seed}" / "best_model.pt"),
                "--split", split, "--output", str(output), "--device", "cuda",
            ],
            cwd=PROJECT, env=environment,
        ).returncode

    execute_parallel(
        f"current-latent-{'-'.join(splits)}", items, cache_slots,
        is_heavy=lambda _item: False, runner=runner,
        item_label=lambda item: f"current-latent/seed{item[0]}/{item[1]}", unit="cache",
    )


def fold_ranges() -> list[tuple[int, int]]:
    origins = np.load(
        PROJECT / "data" / "aggregate_cluster_v1" / "train" / "origin_unix.npy",
        allow_pickle=False,
    )
    ordered = np.sort(np.unique(origins))
    parts = np.array_split(ordered, 5)
    return [(int(part[0]), int(part[-1])) for part in parts]


def train_auxiliary(seeds: Sequence[int], slots: Sequence[WorkerSlot]) -> None:
    groups = ("weather_mean", "weather_envelope")
    head_stage = "screen_aux_head" if tuple(seeds) == SCREEN_SEEDS else "formal_aux_head"
    joint_stage = "screen_aux_joint" if tuple(seeds) == SCREEN_SEEDS else "formal_aux_joint"
    run_jobs(
        f"{head_stage}",
        [Job(head_stage, group, seed, "head") for group in groups for seed in seeds], slots,
    )
    run_jobs(
        f"{joint_stage}",
        [
            Job(joint_stage, group, seed, "joint", initial(head_stage, group, seed))
            for group in groups for seed in seeds
        ], slots,
    )


def train_oof_envelopes(seeds: Sequence[int], slots: Sequence[WorkerSlot]) -> None:
    ranges = fold_ranges()
    jobs = [
        Job(
            "oof_envelope", f"weather_envelope_fold{fold}", seed, "head",
            exclude_start=start, exclude_stop=stop, partial=True,
        )
        for seed in seeds for fold, (start, stop) in enumerate(ranges)
    ]
    run_jobs("oof-envelope", jobs, slots)
    for seed in seeds:
        payload = {
            "seed": seed,
            "purged_embargo_seconds": 6 * 3600,
            "folds": [
                {
                    "fold": fold, "start_unix": start, "stop_unix": stop,
                    "checkpoint": str(
                        ROOT / "oof_envelope" / f"weather_envelope_fold{fold}"
                        / f"seed{seed}" / "best_model.pt"
                    ),
                }
                for fold, (start, stop) in enumerate(ranges)
            ],
        }
        atomic_json(ROOT / "oof_envelope" / f"seed{seed}_fold_manifest.json", payload)


def cache_realization(seeds: Sequence[int], splits: Sequence[str], slots: Sequence[WorkerSlot]) -> None:
    items = [(seed, split) for seed in seeds for split in splits]
    cache_slots = tuple(slot for slot in slots if slot.slot == 0) or tuple(slots)

    def runner(item: tuple[int, str], slot: WorkerSlot, _total: int) -> int:
        seed, split = item
        output = ROOT / "realization_cache" / f"seed{seed}" / split
        marker = output / "COMPLETE"
        if marker.is_file() and (output / "manifest.json").is_file():
            return 0
        output.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(CUDA_VISIBLE_DEVICES=str(slot.gpu))
        command = [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "cache_aggregate_realization_factors.py"),
            "--aggregate-cache", str(PROJECT / "data" / "aggregate_cluster_v1"),
            "--response-cache", str(PROJECT / "data" / "aggregate_response_state_v1"),
            "--response-latent-cache", str(RESPONSE_LATENT_ROOT / f"seed{seed}"),
            "--dynamics-time-cache", str(TIME_CACHE),
            "--dynamics-latent-cache", str(ROOT / "latent_cache" / f"seed{seed}"),
            "--split", split, "--output", str(output), "--device", "cuda",
        ]
        if split == "train":
            command += ["--fold-manifest", str(ROOT / "oof_envelope" / f"seed{seed}_fold_manifest.json")]
        else:
            command += ["--checkpoint", auxiliary("weather_envelope", seed)]
        return subprocess.run(command, cwd=PROJECT, env=environment).returncode

    execute_parallel(
        f"realization-{'-'.join(splits)}", items, cache_slots,
        is_heavy=lambda _item: False, runner=runner,
        item_label=lambda item: f"realization/seed{item[0]}/{item[1]}", unit="cache",
    )


def core_jobs(stage: str, seeds: Sequence[int], groups: Sequence[str], training_stage: str) -> list[Job]:
    prior_stage = stage.replace("joint", "head")
    jobs = []
    for group in groups:
        for seed in seeds:
            aux = ""
            if group == "m5":
                aux = auxiliary("weather_envelope", seed)
            elif group in {"m6a", "m6b"}:
                aux = auxiliary("weather_mean", seed)
            jobs.append(Job(
                stage, group, seed, training_stage,
                initial(prior_stage, group, seed) if training_stage == "joint" else "",
                aux,
            ))
    return jobs


def select_candidates() -> list[str]:
    rows = []
    for group in CORE_GROUPS:
        values = [metric(final_run(group, seed), "cluster_mae_mw") for seed in SCREEN_SEEDS]
        control = [metric(a0_dir(seed), "cluster_mae_mw") for seed in SCREEN_SEEDS]
        ramp = [
            metric(final_run(group, seed), "cluster_ramp_mae_mw")
            - metric(a0_dir(seed), "cluster_ramp_mae_mw") for seed in SCREEN_SEEDS
        ]
        bands = {
            key: statistics.mean(
                metric(final_run(group, seed), key) - metric(a0_dir(seed), key)
                for seed in SCREEN_SEEDS
            ) for key in HORIZON_KEYS
        }
        differences = [a - b for a, b in zip(values, control)]
        delta = statistics.mean(differences)
        direct = delta <= -1.0 and sum(value < 0 for value in differences) == 3
        conditional = (
            delta < 0.0 and statistics.mean(ramp) <= -1.0 and max(bands.values()) <= 1.5
        )
        rows.append({
            "group": group, "mean_mae_mw": statistics.mean(values),
            "mean_delta_vs_a0_ft_mw": delta, "wins": sum(value < 0 for value in differences),
            "mean_ramp_delta_mw": statistics.mean(ramp), "horizon_delta_mw": bands,
            "direct_promotion": direct, "conditional_promotion": conditional,
        })
    rows.sort(key=lambda row: float(row["mean_mae_mw"]))
    selected = [str(rows[0]["group"])]
    for row in rows:
        group = str(row["group"])
        if group not in selected and (row["direct_promotion"] or row["conditional_promotion"]):
            selected.append(group)
        if len(selected) == 3:
            break
    atomic_json(
        ROOT / "selection" / "screening_selection.json",
        {
            "created_unix": int(time.time()), "selection_split": "2023_validation",
            "rule": "always retain numerical best; add direct/conditional qualifiers up to three",
            "ranking": rows, "formal_candidates": selected, "test_results_seen": False,
        },
    )
    return selected


def freeze_validation(groups: Sequence[str]) -> str:
    rows = []
    for group in ("a0_ft", *groups):
        run_for = a0_dir if group == "a0_ft" else lambda seed, g=group: final_run(g, seed)
        values = [metric(run_for(seed), "cluster_mae_mw") for seed in FORMAL_SEEDS]
        rows.append({
            "group": group, "mean_mae_mw": statistics.mean(values),
            "sd_mae_mw": statistics.stdev(values), "seed_mae_mw": values,
        })
    rows.sort(key=lambda row: float(row["mean_mae_mw"]))
    selected = str(rows[0]["group"])
    atomic_json(
        ROOT / "selection" / "validation_freeze.json",
        {
            "created_unix": int(time.time()), "selection_split": "2023_validation",
            "rule": "lowest 10-seed mean MAE; significance is descriptive only",
            "ranking": rows, "selected_final_model": selected, "test_results_seen": False,
        },
    )
    return selected


def run_tests(groups: Sequence[str], slots: Sequence[WorkerSlot]) -> None:
    items = [(group, seed) for group in groups for seed in FORMAL_SEEDS if not audited(final_run(group, seed), "test")]
    if not items:
        return

    def runner(item: tuple[str, int], slot: WorkerSlot, _total: int) -> int:
        group, seed = item
        environment = dict(os.environ)
        environment.update(
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:test:{group}:s{seed}]",
        )
        return subprocess.run(
            [str(PROJECT / "shell" / "run_aggregate_cluster_test.sh"), str(slot.gpu), str(final_run(group, seed))],
            cwd=PROJECT, env=environment,
        ).returncode

    execute_parallel(
        "dynamics-frozen-test", items, slots, is_heavy=lambda _item: False,
        runner=runner, item_label=lambda item: f"{item[0]}/seed{item[1]}/test", unit="eval",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured aggregate weather-dynamics experiments")
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--slots", default="auto")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain distinct physical indices")
    missing = [seed for seed in FORMAL_SEEDS if not audited(R2_ROOT / f"seed{seed}") or not audited(a0_dir(seed))]
    if missing:
        raise RuntimeError(f"Missing audited R2-W/A0-FT controls: {missing}")
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    slots = build_worker_slots(gpus, parse_slot_counts(gpus, args.slots, totals))
    ROOT.mkdir(parents=True, exist_ok=True)
    snapshot = code_snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        previous = json.loads(snapshot_path.read_text(encoding="utf-8"))
        scientific = [path for path in ROOT.glob("screen_joint/*/seed*/COMPLETE")]
        if scientific and previous["sha256"] != snapshot["sha256"]:
            raise RuntimeError("Scientific code changed after dynamics screening began")
    atomic_json(snapshot_path, snapshot)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "objective": "compare structured weather-driven aggregate-power mechanisms",
            "groups": list(CORE_GROUPS), "screen_seeds": list(SCREEN_SEEDS),
            "formal_seeds": list(FORMAL_SEEDS), "a0_ft_reference": str(STRICT_ROOT),
            "weather_frontend": "frozen_same-seed_R2_latent_interpolation",
            "gpus": list(gpus), "slots": [slot.label for slot in slots],
            "code_sha256": snapshot["sha256"], "test_seen_before_freeze": False,
        },
    )
    print(f"[dynamics] gpus={list(gpus)} workers={len(slots)}", flush=True)

    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "prepare_aggregate_dynamics_data.py"),
            "--source-h5", str(PROJECT / "data" / "jilin_56farm_pad8.h5"),
            "--aggregate-cache", str(PROJECT / "data" / "aggregate_cluster_v1"),
            "--output", str(TIME_CACHE),
        ], cwd=PROJECT, check=True,
    )
    cache_response_latents(FORMAL_SEEDS, ("train", "val"), slots)
    cache_current_latents(FORMAL_SEEDS, ("train", "val"), slots)
    train_auxiliary(SCREEN_SEEDS, slots)
    train_oof_envelopes(SCREEN_SEEDS, slots)
    cache_realization(SCREEN_SEEDS, ("train", "val"), slots)
    run_jobs("screen-head", core_jobs("screen_head", SCREEN_SEEDS, CORE_GROUPS, "head"), slots)
    run_jobs("screen-joint", core_jobs("screen_joint", SCREEN_SEEDS, CORE_GROUPS, "joint"), slots)
    formal = select_candidates()

    auxiliary_groups = []
    if "m5" in formal:
        auxiliary_groups.append("weather_envelope")
    if any(group in formal for group in ("m6a", "m6b")):
        auxiliary_groups.append("weather_mean")
    if auxiliary_groups:
        head_stage, joint_stage = "formal_aux_head", "formal_aux_joint"
        run_jobs(
            head_stage,
            [Job(head_stage, group, seed, "head") for group in auxiliary_groups for seed in REMAINING_SEEDS],
            slots,
        )
        run_jobs(
            joint_stage,
            [
                Job(joint_stage, group, seed, "joint", initial(head_stage, group, seed))
                for group in auxiliary_groups for seed in REMAINING_SEEDS
            ], slots,
        )
    if "m5" in formal:
        train_oof_envelopes(REMAINING_SEEDS, slots)
        cache_realization(REMAINING_SEEDS, ("train", "val"), slots)
    run_jobs("formal-head", core_jobs("formal_head", REMAINING_SEEDS, formal, "head"), slots)
    run_jobs("formal-joint", core_jobs("formal_joint", REMAINING_SEEDS, formal, "joint"), slots)
    selected = freeze_validation(formal)

    test_groups = list(formal)
    if selected == "m4b":
        ablation = ("m4_no_recurrence",)
        run_jobs("ablation-screen-head", core_jobs("screen_head", SCREEN_SEEDS, ablation, "head"), slots)
        run_jobs("ablation-screen-joint", core_jobs("screen_joint", SCREEN_SEEDS, ablation, "joint"), slots)
        run_jobs("ablation-formal-head", core_jobs("formal_head", REMAINING_SEEDS, ablation, "head"), slots)
        run_jobs("ablation-formal-joint", core_jobs("formal_joint", REMAINING_SEEDS, ablation, "joint"), slots)
        test_groups.extend(ablation)
    cache_response_latents(FORMAL_SEEDS, ("test",), slots)
    cache_current_latents(FORMAL_SEEDS, ("test",), slots)
    if "m5" in test_groups:
        cache_realization(FORMAL_SEEDS, ("test",), slots)
    run_tests(test_groups, slots)
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "analyze_aggregate_weather_dynamics.py"),
        ], cwd=PROJECT, check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[dynamics] complete selected={selected} formal={formal}", flush=True)


if __name__ == "__main__":
    main()
