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
from typing import Any, Iterable, Sequence

import numpy as np

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
ROOT = PROJECT / "artifacts" / "local_state_identifiability_v2"
PRIOR = PROJECT / "artifacts" / "local_to_global_raw_bilinear_v1"
BASE_CACHE = PROJECT / "data" / "local_bilinear_56farm_v1"
STATE_CACHE = PROJECT / "data" / "local_state_identifiability_v2"
SOURCE_H5 = PROJECT / "data" / "jilin_56farm_pad8.h5"
SCREEN = (42, 46, 50)
FORMAL = tuple(range(42, 52))
REMAINING = tuple(seed for seed in FORMAL if seed not in SCREEN)
HORIZON = ("horizon_0_1h_mae_mw", "horizon_1_3h_mae_mw", "horizon_3_6h_mae_mw")
PHASE_A = (
    "b0", "b_static", "id_only", "p_noid", "p_static", "p_direct", "g_static", "gl_static"
)
PHASE_A_STATE = ("p_noid", "p_static", "p_direct", "g_static", "gl_static")
PHASE_B = ("p12_static", "physical_static", "hierarchical_static", "io_pair_static", "io_pair_id_static")
LOCAL_GROUPS = {
    "p_noid", "p_static", "p_direct", "gl_static", "p12_static", "physical_static",
    "hierarchical_static", "io_pair_static", "io_pair_id_static", "p_id", "p_affine",
}


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    split: str = "train"
    epochs: int = 40

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"


def base_checkpoint(seed: int) -> str:
    stage = "screen_base" if seed in SCREEN else "formal_base"
    path = PRIOR / stage / "l1_m" / f"seed{seed}" / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def complete(job: Job) -> bool:
    marker = job.run_dir / ("COMPLETE" if job.split == "train" else "TEST_COMPLETE")
    suffix = "val" if job.split == "train" else "test"
    return all(
        path.is_file()
        for path in (
            marker,
            job.run_dir / "best_model.pt",
            job.run_dir / f"metrics_{suffix}.json",
            job.run_dir / f"predictions_{suffix}.npz",
        )
    )


def run_jobs(label: str, jobs: Iterable[Job], slots: Sequence[WorkerSlot]) -> None:
    pending = [job for job in jobs if not complete(job)]
    if not pending:
        print(f"[state-id] {label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, total_memory: int) -> int:
        environment = dict(os.environ)
        environment.update(
            WIND_STATE_ID_ROOT=str(ROOT),
            WIND_STATE_CYCLES_PER_BATCH="8" if total_memory >= 40000 else "5",
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.stage}:{job.group}:s{job.seed}]",
        )
        return subprocess.run(
            [
                str(PROJECT / "shell" / "run_local_state_identifiability_job.sh"),
                str(slot.gpu), job.stage, job.group, str(job.seed),
                base_checkpoint(job.seed), str(job.epochs), job.split,
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


def metric(stage: str, group: str, seed: int, split: str = "val") -> dict[str, Any]:
    return json.loads(
        (ROOT / stage / group / f"seed{seed}" / f"metrics_{split}.json").read_text(
            encoding="utf-8"
        )
    )


def values(stage: str, group: str, key: str, seeds: Sequence[int]) -> np.ndarray:
    return np.asarray([float(metric(stage, group, seed)[key]) for seed in seeds])


def farm_macro_cf(stage: str, group: str, seed: int, split: str = "val") -> float:
    archive = np.load(
        ROOT / stage / group / f"seed{seed}" / f"predictions_{split}.npz",
        allow_pickle=False,
    )
    capacity = np.load(BASE_CACHE / "static" / "station_capacity_mw.npy")
    error = np.abs(archive["station_prediction_mw"] - archive["station_target_mw"])
    return float((error / capacity[None, :, None]).mean(axis=(0, 2)).mean())


def reference(group: str) -> str:
    if group in {"p_noid"}:
        return "b0"
    if group in {"p_id"}:
        return "id_only"
    if group in {"p_affine"}:
        return "affine_only"
    return "b_static"


def screen(group: str) -> dict[str, Any]:
    ref = reference(group)
    mae = values("screen", group, "cluster_mae_mw", SCREEN)
    ref_mae = values("screen", ref, "cluster_mae_mw", SCREEN)
    delta = mae - ref_mae
    shuffled = values("screen", f"{group}_shuf", "cluster_mae_mw", SCREEN)
    shuffle_degradation = shuffled - mae
    rmse_delta = values("screen", group, "cluster_rmse_mw", SCREEN) - values(
        "screen", ref, "cluster_rmse_mw", SCREEN
    )
    horizon = {
        key: float(np.mean(values("screen", group, key, SCREEN) - values("screen", ref, key, SCREEN)))
        for key in HORIZON
    }
    farm_delta = np.asarray(
        [farm_macro_cf("screen", group, seed) - farm_macro_cf("screen", ref, seed) for seed in SCREEN]
    )
    local_ok = group not in LOCAL_GROUPS or int(np.sum(farm_delta < 0.0)) >= 2
    passed = bool(
        delta.mean() <= -0.5
        and np.sum(delta < 0.0) >= 2
        and shuffle_degradation.mean() >= 0.5
        and np.sum(shuffle_degradation > 0.0) >= 2
        and rmse_delta.mean() <= 3.0
        and max(horizon.values()) <= 3.0
        and local_ok
    )
    return {
        "group": group,
        "reference": ref,
        "mean_delta_mae_mw": float(delta.mean()),
        "wins": int(np.sum(delta < 0.0)),
        "delta_mae_mw": delta.tolist(),
        "mean_train_shuffle_degradation_mw": float(shuffle_degradation.mean()),
        "shuffle_positive": int(np.sum(shuffle_degradation > 0.0)),
        "train_shuffle_degradation_mw": shuffle_degradation.tolist(),
        "mean_rmse_delta_mw": float(rmse_delta.mean()),
        "horizon_delta_mw": horizon,
        "farm_macro_cf_delta": farm_delta.tolist(),
        "farm_wins": int(np.sum(farm_delta < 0.0)),
        "passed": passed,
    }


def nonstate_gain(group: str, ref: str) -> dict[str, Any]:
    difference = values("screen", group, "cluster_mae_mw", SCREEN) - values(
        "screen", ref, "cluster_mae_mw", SCREEN
    )
    return {
        "mean_delta_mae_mw": float(difference.mean()),
        "wins": int(np.sum(difference < 0.0)),
        "passed": bool(difference.mean() <= -0.5 and np.sum(difference < 0.0) >= 2),
    }


def link_screen_to_formal(group: str) -> None:
    for seed in SCREEN:
        target = ROOT / "formal" / group / f"seed{seed}"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(ROOT / "screen" / group / f"seed{seed}", target_is_directory=True)


def intervention_names(group: str) -> tuple[str, ...]:
    result = ["strong_sample", "reference", "zero"]
    if group in LOCAL_GROUPS:
        result.append("station_permutation")
    return tuple(result)


def evaluate_interventions(
    groups: Sequence[str], seeds: Sequence[int], split: str, slots: Sequence[WorkerSlot]
) -> None:
    items = [
        (group, seed, name)
        for group in groups
        for seed in seeds
        for name in intervention_names(group)
        if not (
            ROOT / "interventions" / group / name / f"seed{seed}" / f"metrics_{split}.json"
        ).is_file()
    ]
    if not items:
        return

    def runner(item: tuple[str, int, str], slot: WorkerSlot, _total: int) -> int:
        group, seed, name = item
        run = ROOT / "formal" / group / f"seed{seed}"
        target = ROOT / "interventions" / group / name / f"seed{seed}"
        target.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        command = [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "evaluate_forced_evolution.py"),
            "--config", str(run / "config.yaml"),
            "--checkpoint", str(run / "best_model.pt"),
            "--split", split,
            "--output", str(target / f"metrics_{split}.json"),
            "--predictions-output", str(target / f"predictions_{split}.npz"),
            "--export-station-predictions",
            "--response-state-intervention", name,
            "--device", "cuda",
        ]
        return subprocess.run(command, cwd=PROJECT, env=environment).returncode

    execute_parallel(
        f"interventions-{split}", items, slots,
        is_heavy=lambda _item: False, runner=runner,
        item_label=lambda item: f"{item[0]}/s{item[1]}/{item[2]}/{split}", unit="eval",
    )


def snapshot() -> dict[str, Any]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict local response-state identifiability study")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", default="auto")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    slots = tuple(
        sorted(
            build_worker_slots(gpus, parse_slot_counts(gpus, args.slots, totals)),
            key=lambda slot: (slot.slot, slot.gpu),
        )
    )
    if not (BASE_CACHE / "COMPLETE").is_file():
        raise FileNotFoundError(BASE_CACHE / "COMPLETE")
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "prepare_local_state_identifiability_data.py"),
            "--source-h5", str(SOURCE_H5), "--base-cache", str(BASE_CACHE),
            "--output", str(STATE_CACHE),
        ], cwd=PROJECT, check=True,
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    current = snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        prior = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if prior["sha256"] != current["sha256"]:
            raise RuntimeError("Scientific code changed after identifiability runs began")
    else:
        atomic_json(snapshot_path, current)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "screen_seeds": list(SCREEN), "formal_seeds": list(FORMAL),
            "gpus": list(gpus), "slots": args.slots,
            "primary_population": "all 56 farms / 6827.01 MW",
            "farm_selection": "forbidden; train-only quality audit only",
            "validation_year": 2023, "frozen_check_year": 2024,
            "test_is_pristine_blind": False,
            "screen_gate": {
                "cluster_mae_improvement_mw": 0.5, "minimum_wins": "2/3",
                "matched_train_shuffle_degradation_mw": 0.5,
                "rmse_and_horizon_max_worsening_mw": 3.0,
                "local_cells_require_farm_macro_wins": "2/3",
            },
        },
    )

    phase_a_jobs = [Job("screen", group, seed) for group in PHASE_A for seed in SCREEN]
    phase_a_jobs += [
        Job("screen", f"{group}_shuf", seed) for group in PHASE_A_STATE for seed in SCREEN
    ]
    run_jobs("phase-a", phase_a_jobs, slots)
    phase_a_result = {group: screen(group) for group in PHASE_A_STATE}
    static_result = nonstate_gain("b_static", "b0")

    identity_result = nonstate_gain("id_only", "b0")
    identity_groups: tuple[str, ...] = ()
    if identity_result["passed"]:
        identity_groups = ("p_id", "p_id_shuf", "affine_only", "p_affine", "p_affine_shuf")
        run_jobs(
            "conditional-station-identity",
            [Job("screen", group, seed) for group in identity_groups for seed in SCREEN],
            slots,
        )

    phase_b_jobs = [Job("screen", group, seed) for group in PHASE_B for seed in SCREEN]
    phase_b_jobs += [
        Job("screen", f"{group}_shuf", seed) for group in PHASE_B for seed in SCREEN
    ]
    run_jobs("phase-b-state-source", phase_b_jobs, slots)
    all_screen = {**phase_a_result, **{group: screen(group) for group in PHASE_B}}
    if identity_groups:
        all_screen.update({group: screen(group) for group in ("p_id", "p_affine")})
    passed_strict = [
        group for group, row in all_screen.items()
        if row["passed"] and group != "p_direct"
    ]
    passed_strict.sort(key=lambda group: all_screen[group]["mean_delta_mae_mw"])
    promoted = passed_strict[:2]
    if all_screen["p_direct"]["passed"]:
        promoted.append("p_direct")
    promoted = list(dict.fromkeys(promoted))
    atomic_json(
        ROOT / "selection" / "screening.json",
        {
            "static_context": static_result,
            "learned_identity": identity_result,
            "identity_cells_run": list(identity_groups),
            "candidates": all_screen,
            "promoted": promoted,
            "promotion_limit": "two strict candidates plus direct diagnostic",
            "test_seen": False,
        },
    )

    formal_controls = {"b0", "b_static"}
    if any(reference(group) == "id_only" for group in promoted):
        formal_controls.add("id_only")
    if any(reference(group) == "affine_only" for group in promoted):
        formal_controls.add("affine_only")
    formal_groups = sorted(formal_controls | set(promoted) | {f"{group}_shuf" for group in promoted})
    run_jobs(
        "formal-remaining",
        [Job("formal", group, seed) for group in formal_groups for seed in REMAINING],
        slots,
    )
    for group in formal_groups:
        # Controls/candidates were all screened before any 2024 evaluation.
        if all((ROOT / "screen" / group / f"seed{seed}").is_dir() for seed in SCREEN):
            link_screen_to_formal(group)
        else:
            run_jobs("formal-missing-screen", [Job("formal", group, seed) for seed in SCREEN], slots)
    means = {
        group: statistics.mean(float(metric("formal", group, seed)["cluster_mae_mw"]) for seed in FORMAL)
        for group in formal_groups
    }
    selected = min(promoted, key=means.__getitem__) if promoted else None
    atomic_json(
        ROOT / "selection" / "validation_freeze.json",
        {
            "formal_groups": formal_groups,
            "formal_mean_cluster_mae_mw": means,
            "selected": selected,
            "promoted": promoted,
            "selection_split": "2023 validation",
            "test_seen": False,
        },
    )

    run_jobs(
        "frozen-2024",
        [Job("formal", group, seed, split="test") for group in formal_groups for seed in FORMAL],
        slots,
    )
    evaluate_interventions(promoted, FORMAL, "val", slots)
    evaluate_interventions(promoted, FORMAL, "test", slots)
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "analyze_local_state_identifiability_v2.py"),
        ], cwd=PROJECT, check=True,
    )
    (ROOT / "COMPLETE").touch()
    print(f"[state-id] complete selected={selected}", flush=True)


if __name__ == "__main__":
    main()
