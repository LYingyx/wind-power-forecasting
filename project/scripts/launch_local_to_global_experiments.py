from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Iterable, Sequence

from launch_aggregate_cluster_experiments import (
    WorkerSlot,
    _looks_like_oom,
    atomic_json,
    build_worker_slots,
    execute_parallel,
    gpu_state,
    parse_slot_counts,
)


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "local_to_global_raw_bilinear_v1"
CACHE = PROJECT / "data" / "local_bilinear_56farm_v1"
SCREEN = (42, 46, 50)
FORMAL = tuple(range(42, 52))
REMAINING = tuple(seed for seed in FORMAL if seed not in SCREEN)
A0 = PROJECT / "artifacts" / "aggregate_strict_modulation_v1"


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    initial: str = ""
    epochs: int = 80
    split: str = "train"

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"


def complete(job: Job) -> bool:
    marker = job.run_dir / ("COMPLETE" if job.split == "train" else "TEST_COMPLETE")
    metric = job.run_dir / f"metrics_{'val' if job.split == 'train' else 'test'}.json"
    checkpoint = job.run_dir / "best_model.pt"
    return marker.is_file() and metric.is_file() and checkpoint.is_file()


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not complete(job)]
    if not pending:
        print(f"[local-global] {label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, total_memory: int) -> int:
        cycles = 12 if total_memory >= 40000 else 6
        environment = dict(os.environ)
        environment.update(
            WIND_LOCAL_GLOBAL_ROOT=str(ROOT),
            WIND_LOCAL_CYCLES_PER_BATCH=str(cycles),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}]",
        )
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_local_to_global_job.sh"),
                str(slot.gpu), job.stage, job.group, str(job.seed),
                job.initial, str(job.epochs), job.split,
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        label,
        pending,
        slots,
        is_heavy=lambda _job: False,
        runner=runner,
        item_label=lambda job: f"{job.stage}/{job.group}/seed{job.seed}/{job.split}",
        oom_detector=lambda job, code: _looks_like_oom(job.run_dir / "run.log", code),
        unit="run",
    )


def checkpoint(stage: str, group: str, seed: int) -> str:
    path = ROOT / stage / group / f"seed{seed}" / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def value(stage: str, group: str, seed: int, split: str = "val") -> float:
    path = ROOT / stage / group / f"seed{seed}" / f"metrics_{split}.json"
    return float(json.loads(path.read_text(encoding="utf-8"))["cluster_mae_mw"])


def mean(stage: str, group: str, seeds: Sequence[int]) -> float:
    return statistics.mean(value(stage, group, seed) for seed in seeds)


def snapshot() -> dict[str, object]:
    digest = hashlib.sha256()
    files: list[str] = []
    for directory in ("src", "scripts", "shell", "configs", "tests"):
        for path in sorted((PROJECT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}:
                relative = str(path.relative_to(PROJECT))
                files.append(relative)
                digest.update(relative.encode())
                digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "files": files, "created_unix": int(time.time())}


def evaluate_interventions(stage: str, group: str, seeds: Sequence[int], split: str, slots: Sequence[WorkerSlot]) -> None:
    interventions = ("station_roll", "batch_roll", "reference")
    items = [(seed, intervention) for seed in seeds for intervention in interventions]

    def runner(item: tuple[int, str], slot: WorkerSlot, _total: int) -> int:
        seed, intervention = item
        run = ROOT / stage / group / f"seed{seed}"
        output = ROOT / "interventions" / group / intervention / f"seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"metrics_{split}.json"
        if target.is_file():
            return 0
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        command = [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "evaluate_forced_evolution.py"),
            "--config", str(run / "config.yaml"),
            "--checkpoint", str(run / "best_model.pt"),
            "--split", split,
            "--output", str(target),
            "--local-state-intervention", intervention,
            "--device", "cuda",
        ]
        return subprocess.run(command, cwd=PROJECT, env=environment).returncode

    execute_parallel(
        f"interventions-{split}", items, slots,
        is_heavy=lambda _item: False, runner=runner,
        item_label=lambda item: f"{group}/seed{item[0]}/{item[1]}/{split}", unit="eval",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the raw-bilinear local-to-global matrix")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", default="1")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    slots = build_worker_slots(gpus, parse_slot_counts(gpus, args.slots, totals))
    # Fill one process on every requested GPU before assigning the second and
    # third slots.  This avoids concentrating short experiment stages on the
    # first one or two devices while retaining multi-process capacity.
    slots = tuple(sorted(slots, key=lambda slot: (slot.slot, slot.gpu)))
    ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_value = snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        prior = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if prior["sha256"] != snapshot_value["sha256"]:
            raise RuntimeError("Scientific code changed after local-to-global runs began")
    else:
        atomic_json(snapshot_path, snapshot_value)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "screen_seeds": list(SCREEN), "formal_seeds": list(FORMAL),
            "gpus": list(gpus), "slots": args.slots,
            "a0_reference": str(A0), "test_read_before_freeze": False,
        },
    )

    if not (CACHE / "COMPLETE").is_file():
        subprocess.run(
            [
                os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                str(PROJECT / "scripts" / "prepare_local_bilinear_data.py"),
                "--source", str(PROJECT / "data" / "model_cache_pad5_f8"),
                "--output", str(CACHE),
            ], cwd=PROJECT, check=True,
        )

    run_jobs(
        "screen-base",
        [Job("screen_base", group, seed) for group in ("l1_c", "l1_m") for seed in SCREEN],
        slots,
    )
    run_jobs(
        "screen-core",
        [
            Job("screen_core", group, seed, checkpoint("screen_base", "l1_m", seed), 20)
            for group in ("l1_mft", "l2_agg", "l3_local") for seed in SCREEN
        ],
        slots,
    )
    local_gate = (
        mean("screen_core", "l3_local", SCREEN) < mean("screen_core", "l1_mft", SCREEN)
        and mean("screen_core", "l3_local", SCREEN) < mean("screen_core", "l2_agg", SCREEN)
    )
    sharing_groups: tuple[str, ...] = ()
    if local_gate:
        sharing_groups = ("l4_id16", "l5_affine")
        run_jobs(
            "screen-sharing",
            [
                Job("screen_sharing", group, seed, checkpoint("screen_base", "l1_m", seed), 20)
                for group in sharing_groups for seed in SCREEN
            ], slots,
        )
    local_candidates = [("screen_core", "l3_local")]
    local_candidates.extend(("screen_sharing", group) for group in sharing_groups)
    local_stage, local_best = min(
        local_candidates, key=lambda item: mean(item[0], item[1], SCREEN)
    )
    selection = {
        "local_state_gate": local_gate,
        "screen_means": {
            "l1_c": mean("screen_base", "l1_c", SCREEN),
            "l1_m": mean("screen_base", "l1_m", SCREEN),
            "l1_mft": mean("screen_core", "l1_mft", SCREEN),
            "l2_agg": mean("screen_core", "l2_agg", SCREEN),
            "l3_local": mean("screen_core", "l3_local", SCREEN),
            **{group: mean("screen_sharing", group, SCREEN) for group in sharing_groups},
        },
        "formal_groups": ["l1_mft", "l2_agg", local_best],
        "local_best": local_best,
        "test_seen": False,
    }
    atomic_json(ROOT / "selection" / "screening.json", selection)

    run_jobs(
        "formal-base",
        [Job("formal_base", "l1_m", seed) for seed in REMAINING], slots,
    )
    formal_groups = tuple(selection["formal_groups"])
    run_jobs(
        "formal-core",
        [
            Job("formal", group, seed, checkpoint("formal_base", "l1_m", seed), 20)
            for group in formal_groups for seed in REMAINING
        ], slots,
    )
    # Register screening checkpoints into the formal view without copying model data.
    formal_sources = {
        "l1_mft": "screen_core", "l2_agg": "screen_core",
        "l3_local": "screen_core", "l4_id16": "screen_sharing", "l5_affine": "screen_sharing",
    }
    for group in formal_groups:
        for seed in SCREEN:
            target = ROOT / "formal" / group / f"seed{seed}"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(ROOT / formal_sources[group] / group / f"seed{seed}", target_is_directory=True)

    final_means = {group: mean("formal", group, FORMAL) for group in formal_groups}
    selected = min(formal_groups, key=final_means.__getitem__)
    atomic_json(
        ROOT / "selection" / "validation_freeze.json",
        {"formal_means": final_means, "selected": selected, "seeds": list(FORMAL), "test_seen": False},
    )
    evaluate_interventions("formal", local_best, FORMAL, "val", slots)
    run_jobs(
        "frozen-test",
        [Job("formal", group, seed, split="test") for group in formal_groups for seed in FORMAL],
        slots,
    )
    evaluate_interventions("formal", local_best, FORMAL, "test", slots)
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "analyze_local_to_global.py"),
        ], cwd=PROJECT, check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[local-global] complete selected={selected}", flush=True)


if __name__ == "__main__":
    main()
