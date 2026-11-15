from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
from tqdm import tqdm


PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "artifacts" / "aggregate_cluster_v1"
SEEDS_SCREEN = (42, 46, 50)
SEEDS_FORMAL = tuple(range(42, 52))
REMAINING = tuple(seed for seed in SEEDS_FORMAL if seed not in SEEDS_SCREEN)
HORIZON_KEYS = (
    "horizon_0_1h_mae_mw",
    "horizon_1_3h_mae_mw",
    "horizon_3_6h_mae_mw",
)
HEAVY_GROUPS = {
    "r1_raw_interpolation", "r7_patch_regional", "r8_joint_spacetime",
    "r8_factorized", "f_tft", "f_timexer",
}
HIGH_MEMORY_MIB = 40000
LIGHT_MIN_FREE_MIB = 6 * 1024
HEAVY_HIGH_MIN_FREE_MIB = 30 * 1024
HEAVY_LOW_MIN_FREE_MIB = 20 * 1024
STARTUP_RESERVATION_SECONDS = 10.0
T = TypeVar("T")


@dataclass(frozen=True)
class Job:
    stage: str
    group: str
    seed: int
    history_checkpoint: str = ""

    @property
    def run_dir(self) -> Path:
        return ROOT / self.stage / self.group / f"seed{self.seed}"

    @property
    def heavy(self) -> bool:
        return self.group in HEAVY_GROUPS

    @property
    def history_only(self) -> bool:
        return self.group.startswith("hb_")


@dataclass(frozen=True)
class WorkerSlot:
    gpu: int
    slot: int
    position: int

    @property
    def label(self) -> str:
        return f"gpu{self.gpu}.slot{self.slot}"


@dataclass
class _GpuRuntime:
    active: int = 0
    heavy_active: int = 0
    disabled_reason: str = ""
    reservations: dict[str, tuple[int, float]] | None = None

    def __post_init__(self) -> None:
        if self.reservations is None:
            self.reservations = {}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def metric(stage: str, group: str, seed: int, split: str = "val") -> dict[str, Any]:
    path = ROOT / stage / group / f"seed{seed}" / f"metrics_{split}.json"
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def values(stage: str, group: str, seeds: Sequence[int], key: str = "cluster_mae_mw") -> np.ndarray:
    return np.asarray([float(metric(stage, group, seed)[key]) for seed in seeds], dtype=np.float64)


def summary(stage: str, group: str, seeds: Sequence[int], split: str = "val") -> dict[str, Any]:
    rows = [metric(stage, group, seed, split) for seed in seeds]
    result: dict[str, Any] = {"stage": stage, "group": group, "seeds": list(seeds), "split": split}
    for key in (
        "cluster_mae_mw", "cluster_rmse_mw", "cluster_nmae", "cluster_r2",
        "cluster_energy_mae_mwh", "cluster_ramp_mae_mw", "persistence_skill_mae",
        *HORIZON_KEYS, "parameter_count", "trainable_parameter_count",
        "latency_ms_per_origin", "samples_per_second", "peak_allocated_mib",
    ):
        available = [float(row[key]) for row in rows if row.get(key) is not None]
        if available:
            result[f"{key}_mean"] = float(np.mean(available))
            result[f"{key}_sd"] = float(np.std(available, ddof=1)) if len(available) > 1 else 0.0
    return result


def paired(
    stage_a: str,
    group_a: str,
    stage_b: str,
    group_b: str,
    seeds: Sequence[int],
    *,
    split: str = "val",
) -> dict[str, Any]:
    difference = np.asarray(
        [
            float(metric(stage_a, group_a, seed, split)["cluster_mae_mw"])
            - float(metric(stage_b, group_b, seed, split)["cluster_mae_mw"])
            for seed in seeds
        ],
        dtype=np.float64,
    )
    critical = 4.303 if len(seeds) == 3 else 2.262
    mean = float(difference.mean())
    half = critical * float(difference.std(ddof=1)) / math.sqrt(len(difference))
    band_delta = {
        key: float(
            np.mean(
                [metric(stage_a, group_a, seed, split)[key] - metric(stage_b, group_b, seed, split)[key] for seed in seeds]
            )
        )
        for key in HORIZON_KEYS
    }
    return {
        "mean_delta_mw": mean,
        "ci_low_mw": mean - half,
        "ci_high_mw": mean + half,
        "wins": int(np.sum(difference < 0)),
        "differences_mw": difference.tolist(),
        "horizon_delta_mw": band_delta,
        "split": split,
    }


def gpu_state(gpu: int) -> tuple[int, int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    total, free, utilization = (int(item.strip()) for item in output.split(","))
    return total, free, utilization


def parse_slot_counts(
    gpus: Sequence[int],
    specification: str,
    total_memory_by_gpu: dict[int, int] | None = None,
) -> tuple[int, ...]:
    """Resolve auto, uniform, or per-GPU worker slot specifications."""
    value = specification.strip().lower()
    if value == "auto":
        totals = total_memory_by_gpu or {gpu: gpu_state(gpu)[0] for gpu in gpus}
        return tuple(3 if totals[gpu] >= HIGH_MEMORY_MIB else 2 for gpu in gpus)
    fields = [field.strip() for field in value.split(",") if field.strip()]
    try:
        counts = tuple(int(field) for field in fields)
    except ValueError as error:
        raise ValueError("--slots must be auto, a positive integer, or a per-GPU CSV") from error
    if len(counts) == 1:
        counts = counts * len(gpus)
    if len(counts) != len(gpus) or any(count <= 0 for count in counts):
        raise ValueError("--slots CSV must match --gpus and contain only positive integers")
    return counts


def build_worker_slots(gpus: Sequence[int], counts: Sequence[int]) -> tuple[WorkerSlot, ...]:
    if len(gpus) != len(counts):
        raise ValueError("GPU and slot-count lengths differ")
    result: list[WorkerSlot] = []
    position = 1
    for gpu, count in zip(gpus, counts):
        for local_slot in range(count):
            result.append(WorkerSlot(gpu=gpu, slot=local_slot, position=position))
            position += 1
    return tuple(result)


def minimum_free_mib(total_memory: int, heavy: bool) -> int:
    if not heavy:
        return LIGHT_MIN_FREE_MIB
    return HEAVY_HIGH_MIN_FREE_MIB if total_memory >= HIGH_MEMORY_MIB else HEAVY_LOW_MIN_FREE_MIB


def maximum_heavy_jobs(total_memory: int) -> int:
    return 2 if total_memory >= HIGH_MEMORY_MIB else 1


class GpuAdmissionController:
    """Atomically gates starts by free memory and per-GPU heavy-job limits.

    Utilization is intentionally observed only for logging. A short-lived
    logical memory reservation closes the race between admission and CUDA
    allocation becoming visible to nvidia-smi.
    """

    def __init__(
        self,
        slot_counts: dict[int, int],
        *,
        state_provider: Callable[[int], tuple[int, int, int]] = gpu_state,
        reservation_seconds: float = STARTUP_RESERVATION_SECONDS,
    ) -> None:
        self.slot_counts = dict(slot_counts)
        self.state_provider = state_provider
        self.reservation_seconds = reservation_seconds
        self._lock = threading.Lock()
        self._runtime = {gpu: _GpuRuntime() for gpu in slot_counts}

    def _expire_reservations(self, runtime: _GpuRuntime, now: float) -> None:
        assert runtime.reservations is not None
        for token, (_, deadline) in list(runtime.reservations.items()):
            if deadline <= now:
                del runtime.reservations[token]

    def try_acquire(self, gpu: int, heavy: bool, token: str) -> int | None:
        with self._lock:
            runtime = self._runtime[gpu]
            if runtime.disabled_reason or runtime.active >= self.slot_counts[gpu]:
                return None
            total, free, _utilization = self.state_provider(gpu)
            if heavy and runtime.heavy_active >= maximum_heavy_jobs(total):
                return None
            now = time.monotonic()
            self._expire_reservations(runtime, now)
            assert runtime.reservations is not None
            reserved = sum(amount for amount, _ in runtime.reservations.values())
            minimum = minimum_free_mib(total, heavy)
            if free - reserved < minimum:
                return None
            runtime.active += 1
            runtime.heavy_active += int(heavy)
            runtime.reservations[token] = (minimum, now + self.reservation_seconds)
            return total

    def release(self, gpu: int, heavy: bool, token: str) -> None:
        with self._lock:
            runtime = self._runtime[gpu]
            runtime.active = max(0, runtime.active - 1)
            runtime.heavy_active = max(0, runtime.heavy_active - int(heavy))
            assert runtime.reservations is not None
            runtime.reservations.pop(token, None)

    def disable(self, gpu: int, reason: str) -> None:
        with self._lock:
            self._runtime[gpu].disabled_reason = reason

    def disabled_reason(self, gpu: int) -> str:
        with self._lock:
            return self._runtime[gpu].disabled_reason

    def snapshot(self, gpu: int) -> dict[str, int | str]:
        with self._lock:
            runtime = self._runtime[gpu]
            return {
                "active": runtime.active,
                "heavy_active": runtime.heavy_active,
                "disabled_reason": runtime.disabled_reason,
            }


def _looks_like_oom(path: Path, returncode: int) -> bool:
    if returncode in {137, -9}:
        return True
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 2 * 1024 * 1024))
        text = stream.read().decode("utf-8", errors="ignore").lower()
    return any(
        marker in text
        for marker in (
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "cublas_status_alloc_failed",
        )
    )


def execute_parallel(
    label: str,
    items: Iterable[T],
    slots: Sequence[WorkerSlot],
    *,
    is_heavy: Callable[[T], bool],
    runner: Callable[[T, WorkerSlot, int], int],
    item_label: Callable[[T], str],
    oom_detector: Callable[[T, int], bool] | None = None,
    unit: str = "job",
    state_provider: Callable[[int], tuple[int, int, int]] = gpu_state,
    reservation_seconds: float = STARTUP_RESERVATION_SECONDS,
) -> None:
    pending = list(items)
    if not pending:
        return
    if not slots:
        raise ValueError("At least one aggregate worker slot is required")
    per_gpu_slots: dict[int, int] = {}
    for slot in slots:
        per_gpu_slots[slot.gpu] = per_gpu_slots.get(slot.gpu, 0) + 1
    totals = {gpu: state_provider(gpu)[0] for gpu in per_gpu_slots}
    admission = GpuAdmissionController(
        per_gpu_slots,
        state_provider=state_provider,
        reservation_seconds=reservation_seconds,
    )
    condition = threading.Condition()
    failures: list[dict[str, Any]] = []
    progress = tqdm(total=len(pending), desc=f"aggregate {label}", unit=unit, position=0, dynamic_ncols=True)
    claim_counter = 0

    def claim(slot: WorkerSlot) -> tuple[T, int, str] | None:
        nonlocal claim_counter
        while True:
            with condition:
                if admission.disabled_reason(slot.gpu) or not pending:
                    return None
                prefer_heavy = totals[slot.gpu] >= HIGH_MEMORY_MIB
                indices = sorted(
                    range(len(pending)),
                    key=lambda index: int(is_heavy(pending[index]) != prefer_heavy),
                )
                for index in indices:
                    item = pending[index]
                    heavy = is_heavy(item)
                    token = f"{slot.label}:{claim_counter}:{item_label(item)}"
                    total_memory = admission.try_acquire(slot.gpu, heavy, token)
                    if total_memory is None:
                        continue
                    claim_counter += 1
                    return pending.pop(index), total_memory, token
                condition.wait(timeout=5.0)

    def worker(slot: WorkerSlot) -> None:
        while True:
            claimed = claim(slot)
            if claimed is None:
                return
            item, total_memory, token = claimed
            heavy = is_heavy(item)
            print(
                f"[aggregate-scheduler] dispatch worker={slot.label} "
                f"item={item_label(item)} heavy={heavy}",
                flush=True,
            )
            detail = ""
            try:
                returncode = runner(item, slot, total_memory)
            except Exception as error:  # Preserve other workers and report centrally.
                returncode = -1
                detail = repr(error)
            oom = bool(returncode and oom_detector is not None and oom_detector(item, returncode))
            if oom:
                reason = f"OOM in {item_label(item)}; no new work will be assigned to {slot.label.split('.')[0]}"
                admission.disable(slot.gpu, reason)
                print(f"[aggregate-scheduler] ERROR {reason}", flush=True)
            admission.release(slot.gpu, heavy, token)
            with condition:
                progress.update(1)
                if returncode:
                    failures.append(
                        {
                            "item": item_label(item),
                            "gpu": slot.gpu,
                            "slot": slot.slot,
                            "returncode": returncode,
                            "oom": oom,
                            "detail": detail,
                        }
                    )
                condition.notify_all()
            if returncode:
                return

    threads = [
        threading.Thread(target=worker, args=(slot,), name=slot.label, daemon=False)
        for slot in slots
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    progress.close()
    if failures:
        raise RuntimeError(f"Aggregate stage {label} failed: {failures}")


def completed_run_is_valid(job: Job) -> bool:
    if not (job.run_dir / "COMPLETE").is_file():
        return False
    result = subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "audit_aggregate_run.py"),
            "--run-dir", str(job.run_dir),
        ],
        cwd=PROJECT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def completed_test_is_valid(stage: str, group: str, seed: int) -> bool:
    run_dir = ROOT / stage / group / f"seed{seed}"
    if not (run_dir / "TEST_COMPLETE").is_file():
        return False
    result = subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "audit_aggregate_run.py"),
            "--run-dir", str(run_dir), "--split", "test",
        ],
        cwd=PROJECT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def run_jobs(
    label: str,
    jobs: Iterable[Job],
    slots: Sequence[WorkerSlot],
    *,
    history_visible: int,
    state_mode: str,
) -> None:
    pending = [job for job in jobs if not completed_run_is_valid(job)]
    if not pending:
        print(f"[aggregate-scheduler] stage={label} already complete", flush=True)
        return

    def runner(job: Job, slot: WorkerSlot, total_memory: int) -> int:
        if job.heavy:
            cycles, accumulation = 4, 3
        elif total_memory < HIGH_MEMORY_MIB:
            cycles, accumulation = 6, 2
        else:
            cycles, accumulation = 12, 1
        environment = dict(os.environ)
        environment.update(
            WIND_AGG_ROOT=str(ROOT),
            WIND_AGG_HISTORY_VISIBLE=str(history_visible),
            WIND_AGG_STATE_MODE=state_mode,
            WIND_AGG_CYCLES_PER_BATCH=str(cycles),
            WIND_AGG_GRAD_ACCUMULATION=str(accumulation),
            WIND_PROGRESS_POSITION=str(slot.position),
            WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.group}:s{job.seed}]",
        )
        command = [
            str(PROJECT / "shell" / "run_aggregate_cluster_job.sh"),
            str(slot.gpu), job.stage, job.group, str(job.seed),
        ]
        if job.history_checkpoint:
            command.append(job.history_checkpoint)
        return subprocess.run(command, cwd=PROJECT, env=environment).returncode

    execute_parallel(
        label,
        pending,
        slots,
        is_heavy=lambda job: job.heavy,
        runner=runner,
        item_label=lambda job: f"{job.stage}/{job.group}/seed{job.seed}",
        oom_detector=lambda job, returncode: _looks_like_oom(job.run_dir / "run.log", returncode),
    )


def checkpoint(stage: str, group: str, seed: int) -> str:
    path = ROOT / stage / group / f"seed{seed}" / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def code_snapshot() -> dict[str, Any]:
    digest = hashlib.sha256()
    files: list[str] = []
    for directory in ("src", "scripts", "shell", "configs", "tests"):
        for path in sorted((PROJECT / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".sh", ".yaml"}:
                continue
            relative = str(path.relative_to(PROJECT))
            files.append(relative)
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "files": files, "created_unix": int(time.time())}


def history_selection(slots: Sequence[WorkerSlot]) -> tuple[int, str, list[tuple[str, str]]]:
    groups = ("hb_persistence", "hb_dlinear", "hb_tcn_1h", "hb_tcn_3h", "hb_tcn_6h", "hb_patchtst")
    run_jobs(
        "history-screen",
        (Job("history", group, seed) for group in groups for seed in SEEDS_SCREEN),
        slots,
        history_visible=24,
        state_mode="recent_1h",
    )
    short = min(("hb_tcn_1h", "hb_tcn_3h"), key=lambda group: values("history", group, SEEDS_SCREEN).mean())
    tcn_best = min((short, "hb_tcn_6h"), key=lambda group: values("history", group, SEEDS_SCREEN).mean())
    external = min(("hb_dlinear", "hb_patchtst"), key=lambda group: values("history", group, SEEDS_SCREEN).mean())
    external_delta = values("history", external, SEEDS_SCREEN) - values("history", tcn_best, SEEDS_SCREEN)
    formal_groups = ["hb_tcn_6h", short]
    if external_delta.mean() <= -1.0 and np.all(external_delta < 0):
        formal_groups.append(external)
    run_jobs(
        "history-formal",
        (Job("history", group, seed) for group in formal_groups for seed in REMAINING),
        slots,
        history_visible=24,
        state_mode="recent_1h",
    )
    candidates = list(dict.fromkeys((short, "hb_tcn_6h")))
    best_mean = min(values("history", group, SEEDS_FORMAL).mean() for group in candidates)
    eligible = [group for group in candidates if values("history", group, SEEDS_FORMAL).mean() <= best_mean + 0.5]
    selected = min(eligible, key=lambda group: {"hb_tcn_1h": 4, "hb_tcn_3h": 12, "hb_tcn_6h": 24}[group])
    comparison = paired("history", "hb_tcn_6h", "history", short, SEEDS_FORMAL)
    if comparison["mean_delta_mw"] <= -1.0 and comparison["wins"] >= 7 and comparison["ci_high_mw"] < 0:
        selected = "hb_tcn_6h"
    visible = {"hb_tcn_1h": 4, "hb_tcn_3h": 12, "hb_tcn_6h": 24}[selected]
    result = {
        "selected_group": selected,
        "history_visible_steps": visible,
        "screening": [summary("history", group, SEEDS_SCREEN) for group in groups],
        "formal": [summary("history", group, SEEDS_FORMAL) for group in formal_groups],
        "full_vs_short": comparison,
    }
    atomic_json(ROOT / "selection" / "history.json", result)
    return visible, selected, [("history", group) for group in formal_groups]


def state_selection(slots: Sequence[WorkerSlot], visible: int, history_group: str) -> tuple[str, list[str]]:
    groups = ("sc_current", "sc_1h", "sc_3h", "sc_6h", "sc_multi", "sc_horizon")
    jobs = (
        Job("state", group, seed, checkpoint("history", history_group, seed))
        for group in groups for seed in SEEDS_SCREEN
    )
    run_jobs("state-screen", jobs, slots, history_visible=visible, state_mode="recent_1h")
    fixed_other = min(
        ("sc_current", "sc_3h", "sc_6h"),
        key=lambda group: values("state", group, SEEDS_SCREEN).mean(),
    )
    structured = min(("sc_multi", "sc_horizon"), key=lambda group: values("state", group, SEEDS_SCREEN).mean())
    formal = ["sc_1h", fixed_other]
    delta = values("state", structured, SEEDS_SCREEN) - values("state", "sc_1h", SEEDS_SCREEN)
    if delta.mean() <= -0.75 and np.all(delta < 0):
        formal.append(structured)
    formal = list(dict.fromkeys(formal))
    run_jobs(
        "state-formal",
        (
            Job("state", group, seed, checkpoint("history", history_group, seed))
            for group in formal for seed in REMAINING
        ),
        slots,
        history_visible=visible,
        state_mode="recent_1h",
    )
    passed: list[str] = []
    comparisons: dict[str, Any] = {}
    for group in formal:
        if group == "sc_1h":
            continue
        comparison = paired("state", group, "state", "sc_1h", SEEDS_FORMAL)
        comparisons[group] = comparison
        if (
            comparison["mean_delta_mw"] <= -1.0
            and comparison["wins"] >= 7
            and comparison["ci_high_mw"] < 0
            and max(comparison["horizon_delta_mw"].values()) <= 1.0
        ):
            passed.append(group)
    if passed:
        selected = min(passed, key=lambda group: values("state", group, SEEDS_FORMAL).mean())
    else:
        best_mean = min(values("state", group, SEEDS_FORMAL).mean() for group in formal)
        eligible = [group for group in formal if values("state", group, SEEDS_FORMAL).mean() <= best_mean + 0.5]
        complexity = {"sc_current": 0, "sc_1h": 1, "sc_3h": 2, "sc_6h": 3, "sc_multi": 4, "sc_horizon": 5}
        selected = min(eligible, key=lambda group: complexity[group])
    mode = {
        "sc_current": "current", "sc_1h": "recent_1h", "sc_3h": "recent_3h",
        "sc_6h": "full_6h", "sc_multi": "multiscale", "sc_horizon": "horizon",
    }[selected]
    atomic_json(
        ROOT / "selection" / "state.json",
        {
            "selected_group": selected,
            "state_mode": mode,
            "screening": [summary("state", group, SEEDS_SCREEN) for group in groups],
            "formal": [summary("state", group, SEEDS_FORMAL) for group in formal],
            "comparisons_vs_sc_1h": comparisons,
        },
    )
    return mode, formal


def screen_pass(stage: str, group: str, reference_stage: str, reference: str) -> bool:
    result = paired(stage, group, reference_stage, reference, SEEDS_SCREEN)
    return (
        result["mean_delta_mw"] <= -1.0
        and result["wins"] == 3
        and max(result["horizon_delta_mw"].values()) <= 1.0
    )


def model_matrix(
    slots: Sequence[WorkerSlot], visible: int, history_group: str, state_mode: str
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    fusion = (
        "f_early_concat", "f_target_concat", "f_lead_gate", "f_prediction_blend",
        "f_cross_attention", "f_bidirectional", "f_tft", "f_astafn", "f_timexer",
    )
    routes = (
        "r1_raw_interpolation", "r2_latent_interpolation", "r3_full_path",
        "r4_target_value", "r5_dynamic_pool", "r6_regional_tokens",
        "r7_patch_regional", "r8_joint_spacetime",
    )
    mechanism = ("m0", "ms", "mw", "madd")
    jobs: list[Job] = [Job("baselines", "w_only", seed) for seed in SEEDS_SCREEN]
    jobs += [Job("fusion", group, seed) for group in fusion for seed in SEEDS_SCREEN]
    jobs += [
        Job("weather", group, seed, checkpoint("history", history_group, seed))
        for group in routes for seed in SEEDS_SCREEN
    ]
    jobs += [
        Job("mechanism", group, seed, checkpoint("history", history_group, seed))
        for group in mechanism for seed in SEEDS_SCREEN
    ]
    run_jobs("model-screen", jobs, slots, history_visible=visible, state_mode=state_mode)

    controls: list[str] = []
    if screen_pass("weather", "r4_target_value", "weather", "r3_full_path"):
        controls.append("r4_no_time")
    if screen_pass("weather", "r8_joint_spacetime", "weather", "r3_full_path"):
        controls.append("r8_factorized")
    run_jobs(
        "route-controls-screen",
        (
            Job("weather", group, seed, checkpoint("history", history_group, seed))
            for group in controls for seed in SEEDS_SCREEN
        ),
        slots,
        history_visible=visible,
        state_mode=state_mode,
    )
    candidate_routes: list[str] = []
    for group in routes:
        if group == "r3_full_path" or not screen_pass("weather", group, "weather", "r3_full_path"):
            continue
        if group == "r4_target_value" and not screen_pass("weather", group, "weather", "r4_no_time"):
            continue
        if group == "r8_joint_spacetime" and not screen_pass("weather", group, "weather", "r8_factorized"):
            continue
        candidate_routes.append(group)
    candidate_routes.sort(key=lambda group: values("weather", group, SEEDS_SCREEN).mean())
    candidate_routes = candidate_routes[:2]
    top_fusion = sorted(fusion, key=lambda group: values("fusion", group, SEEDS_SCREEN).mean())[:3]

    formal_jobs: list[Job] = [Job("baselines", "w_only", seed) for seed in REMAINING]
    formal_jobs += [Job("fusion", group, seed) for group in top_fusion for seed in REMAINING]
    formal_routes = ["r3_full_path", *candidate_routes]
    formal_jobs += [
        Job("weather", group, seed, checkpoint("history", history_group, seed))
        for group in formal_routes for seed in REMAINING
    ]
    selected_controls = []
    if "r4_target_value" in candidate_routes:
        selected_controls.append("r4_no_time")
    if "r8_joint_spacetime" in candidate_routes:
        selected_controls.append("r8_factorized")
    formal_jobs += [
        Job("weather", group, seed, checkpoint("history", history_group, seed))
        for group in selected_controls for seed in REMAINING
    ]
    formal_jobs += [
        Job("mechanism", group, seed, checkpoint("history", history_group, seed))
        for group in mechanism for seed in REMAINING
    ]
    run_jobs("model-formal", formal_jobs, slots, history_visible=visible, state_mode=state_mode)

    comparisons = {
        group: paired("weather", group, "weather", "r3_full_path", SEEDS_FORMAL)
        for group in candidate_routes
    }
    passed_routes = [
        group for group in candidate_routes
        if comparisons[group]["mean_delta_mw"] <= -1.0
        and comparisons[group]["wins"] >= 7
        and comparisons[group]["ci_high_mw"] < 0
        and max(comparisons[group]["horizon_delta_mw"].values()) <= 1.0
    ]
    final_route = min(
        ["r3_full_path", *passed_routes],
        key=lambda group: values("weather", group, SEEDS_FORMAL).mean(),
    )
    atomic_json(
        ROOT / "selection" / "model.json",
        {
            "selected_route": final_route,
            "candidate_routes": candidate_routes,
            "passed_routes": passed_routes,
            "top_fusion": top_fusion,
            "route_comparisons": comparisons,
            "screening": {
                "baselines": [summary("baselines", "w_only", SEEDS_SCREEN)],
                "weather": [summary("weather", group, SEEDS_SCREEN) for group in routes],
                "fusion": [summary("fusion", group, SEEDS_SCREEN) for group in fusion],
                "mechanism": [summary("mechanism", group, SEEDS_SCREEN) for group in mechanism],
                "controls": [summary("weather", group, SEEDS_SCREEN) for group in controls],
            },
        },
    )
    formal_methods = [("baselines", "w_only")]
    formal_methods += [("fusion", group) for group in top_fusion]
    formal_methods += [("weather", group) for group in formal_routes]
    formal_methods += [("weather", group) for group in selected_controls]
    formal_methods += [("mechanism", group) for group in mechanism]
    return candidate_routes, top_fusion, formal_methods


def run_tests(slots: Sequence[WorkerSlot], methods: Sequence[tuple[str, str]]) -> None:
    jobs = [(stage, group, seed) for stage, group in methods for seed in SEEDS_FORMAL]
    pending = [
        (stage, group, seed)
        for stage, group, seed in jobs
        if not completed_test_is_valid(stage, group, seed)
    ]
    if not pending:
        return

    def runner(item: tuple[str, str, int], slot: WorkerSlot, _total_memory: int) -> int:
        stage, group, seed = item
        environment = dict(os.environ)
        environment["WIND_PROGRESS_POSITION"] = str(slot.position)
        environment["WIND_PROGRESS_PREFIX"] = f"[{slot.label}:test:{group}:s{seed}]"
        run_dir = ROOT / stage / group / f"seed{seed}"
        return subprocess.run(
            [str(PROJECT / "shell" / "run_aggregate_cluster_test.sh"), str(slot.gpu), str(run_dir)],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        "frozen-test",
        pending,
        slots,
        is_heavy=lambda item: item[1] in HEAVY_GROUPS,
        runner=runner,
        item_label=lambda item: f"{item[0]}/{item[1]}/seed{item[2]}/test",
        oom_detector=lambda item, returncode: _looks_like_oom(
            ROOT / item[0] / item[1] / f"seed{item[2]}" / "test.log", returncode
        ),
    )


def run_benchmarks(slots: Sequence[WorkerSlot], methods: Sequence[tuple[str, str]]) -> None:
    pending = [
        (stage, group)
        for stage, group in methods
        if not (ROOT / "analysis" / "benchmark" / f"{stage}__{group}.json").is_file()
    ]
    if not pending:
        return

    def runner(item: tuple[str, str], slot: WorkerSlot, _total_memory: int) -> int:
        stage, group = item
        output = ROOT / "analysis" / "benchmark" / f"{stage}__{group}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        environment["WIND_PROGRESS_POSITION"] = str(slot.position)
        environment["WIND_PROGRESS_PREFIX"] = f"[{slot.label}:benchmark:{group}]"
        return subprocess.run(
            [
                os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                str(PROJECT / "scripts" / "benchmark_aggregate_cluster.py"),
                "--run-dir", str(ROOT / stage / group / "seed42"),
                "--output", str(output), "--device", "cuda",
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        "benchmark",
        pending,
        slots,
        is_heavy=lambda item: item[1] in HEAVY_GROUPS,
        runner=runner,
        item_label=lambda item: f"{item[0]}/{item[1]}/benchmark",
        oom_detector=lambda _item, returncode: _looks_like_oom(Path("/nonexistent"), returncode),
        unit="model",
    )


def run_state_intervention(slots: Sequence[WorkerSlot], route: str, state_mode: str) -> None:
    if state_mode in {"current", "recent_1h"}:
        return
    pending = [
        (split, seed)
        for split in ("val", "test")
        for seed in SEEDS_FORMAL
        if not (
            ROOT / "weather" / route / f"seed{seed}" / f"metrics_{split}_state_recent1h.json"
        ).is_file()
    ]
    def runner(item: tuple[str, int], slot: WorkerSlot, _total_memory: int) -> int:
        split, seed = item
        run_dir = ROOT / "weather" / route / f"seed{seed}"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        environment["WIND_PROGRESS_POSITION"] = str(slot.position)
        environment["WIND_PROGRESS_PREFIX"] = f"[{slot.label}:state-off:{route}:s{seed}]"
        return subprocess.run(
            [
                os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
                str(PROJECT / "scripts" / "evaluate_aggregate_cluster.py"),
                "--config", str(run_dir / "config.yaml"),
                "--checkpoint", str(run_dir / "best_model.pt"),
                "--split", split,
                "--output", str(run_dir / f"metrics_{split}_state_recent1h.json"),
                "--device", "cuda", "--state-visible-steps-override", "4",
            ],
            cwd=PROJECT,
            env=environment,
        ).returncode

    execute_parallel(
        "state-intervention",
        pending,
        slots,
        is_heavy=lambda _item: route in HEAVY_GROUPS,
        runner=runner,
        item_label=lambda item: f"weather/{route}/seed{item[1]}/{item[0]}-state-off",
        unit="eval",
        oom_detector=lambda _item, returncode: _looks_like_oom(Path("/nonexistent"), returncode),
    )


def run_regime_analysis() -> None:
    subprocess.run(
        [
            os.environ.get("CHRONOS_PYTHON", "/data/liying/anaconda3/envs/chronos/bin/python"),
            str(PROJECT / "scripts" / "analyze_aggregate_cluster_results.py"),
            "--root", str(ROOT),
            "--cache", str(PROJECT / "data" / "aggregate_cluster_v1"),
            "--manifest", str(ROOT / "frozen_test_manifest.json"),
            "--output", str(ROOT / "analysis" / "regimes"),
        ],
        cwd=PROJECT,
        check=True,
    )


def write_results(
    history_methods: Sequence[tuple[str, str]],
    state_groups: Sequence[str],
    formal_methods: Sequence[tuple[str, str]],
) -> None:
    history_selection_data = json.load((ROOT / "selection" / "history.json").open())
    state_selection_data = json.load((ROOT / "selection" / "state.json").open())
    model_selection_data = json.load((ROOT / "selection" / "model.json").open())
    methods = list(dict.fromkeys([*history_methods, *(('state', group) for group in state_groups), *formal_methods]))
    val_rows = [summary(stage, group, SEEDS_FORMAL, "val") for stage, group in methods]
    test_rows = [summary(stage, group, SEEDS_FORMAL, "test") for stage, group in methods]
    screening_rows = [
        *history_selection_data["screening"],
        *state_selection_data["screening"],
        *[
            row
            for family in model_selection_data["screening"].values()
            for row in family
        ],
    ]
    output = ROOT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "summary.json",
        {"screening_3seed": screening_rows, "validation_10seed": val_rows, "test_10seed": test_rows},
    )
    fields = sorted({key for row in (*val_rows, *test_rows) for key in row})
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*val_rows, *test_rows])
    screen_fields = sorted({key for row in screening_rows for key in row})
    with (output / "screening_3seed.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=screen_fields)
        writer.writeheader()
        writer.writerows(screening_rows)
    best_fusion = min(
        model_selection_data["top_fusion"],
        key=lambda group: values("fusion", group, SEEDS_FORMAL).mean(),
    )
    final_route = model_selection_data["selected_route"]
    final_comparison = {
        split: paired(
            "weather", final_route, "fusion", best_fusion, SEEDS_FORMAL, split=split
        )
        for split in ("val", "test")
    }
    mechanism = {
        split: {
            group: paired(
                "weather", "r3_full_path", "mechanism", group,
                SEEDS_FORMAL, split=split,
            )
            for group in ("m0", "ms", "mw", "madd")
        }
        for split in ("val", "test")
    }
    atomic_json(output / "formal_comparisons.json", {"final_vs_fusion": final_comparison, "mechanism": mechanism})

    state_intervention: dict[str, Any] = {}
    for split in ("val", "test"):
        original = np.asarray(
            [metric("weather", final_route, seed, split)["cluster_mae_mw"] for seed in SEEDS_FORMAL],
            dtype=np.float64,
        )
        paths = [
            ROOT / "weather" / final_route / f"seed{seed}" / f"metrics_{split}_state_recent1h.json"
            for seed in SEEDS_FORMAL
        ]
        if all(path.is_file() for path in paths):
            restricted = np.asarray(
                [float(json.load(path.open())["cluster_mae_mw"]) for path in paths], dtype=np.float64
            )
            delta = restricted - original
            critical = 2.262
            half = critical * float(delta.std(ddof=1)) / math.sqrt(len(delta))
            state_intervention[split] = {
                "restricted_minus_native_mean_mw": float(delta.mean()),
                "sd_mw": float(delta.std(ddof=1)),
                "ci_low_mw": float(delta.mean() - half),
                "ci_high_mw": float(delta.mean() + half),
                "native_wins": int(np.sum(delta > 0)),
            }
    if state_intervention:
        atomic_json(output / "state_recent1h_intervention.json", state_intervention)

    attention_rows: list[dict[str, Any]] = []
    attention_values = []
    for seed in SEEDS_FORMAL:
        value = metric("weather", final_route, seed, "val").get("mean_state_attention")
        if value is not None:
            attention_values.append(np.asarray(value, dtype=np.float64))
    if attention_values:
        mean_attention = np.mean(attention_values, axis=0)
        lag_attention = mean_attention.mean(axis=0)
        for index, weight in enumerate(lag_attention):
            attention_rows.append(
                {"lag_quarter_index": index - 23, "lag_hours": (index - 23) / 4.0, "mean_attention": float(weight)}
            )
        with (output / "state_attention_by_lag.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(attention_rows[0]))
            writer.writeheader()
            writer.writerows(attention_rows)
        atomic_json(output / "state_attention_lead_by_lag.json", mean_attention.tolist())

    benchmark: dict[tuple[str, str], dict[str, Any]] = {}
    for stage, group in methods:
        path = output / "benchmark" / f"{stage}__{group}.json"
        if path.is_file():
            benchmark[(stage, group)] = json.load(path.open())

    regime_summary_path = output / "regimes" / "summary.csv"
    regime_delta_rows: list[dict[str, Any]] = []
    if regime_summary_path.is_file():
        with regime_summary_path.open("r", encoding="utf-8", newline="") as stream:
            regime_rows = list(csv.DictReader(stream))
        lookup = {
            (row["split"], row["stage"], row["group"], row["regime"], row["level"]): float(row["cluster_mae_mw_mean"])
            for row in regime_rows
        }
        for split in ("val", "test"):
            for regime in ("current_power", "recent_1h_change", "history_variability", "future_wind_variation"):
                for level in ("low", "medium", "high"):
                    candidate = lookup[(split, "weather", final_route, regime, level)]
                    reference = lookup[(split, "fusion", best_fusion, regime, level)]
                    regime_delta_rows.append(
                        {"split": split, "regime": regime, "level": level, "final_minus_fusion_mae_mw": candidate - reference}
                    )
        with (output / "regimes" / "final_vs_fusion_delta.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(regime_delta_rows[0]))
            writer.writeheader()
            writer.writerows(regime_delta_rows)

    def format_formal_table(rows: Sequence[dict[str, Any]], split_name: str) -> list[str]:
        table = [
            f"### {split_name}", "",
            "| family | method | MAE/MW | RMSE/MW | R² | Ramp MAE/MW |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            table.append(
                f"| {row['stage']} | {row['group']} | {row['cluster_mae_mw_mean']:.3f} ± {row['cluster_mae_mw_sd']:.3f} "
                f"| {row['cluster_rmse_mw_mean']:.3f} ± {row['cluster_rmse_mw_sd']:.3f} "
                f"| {row['cluster_r2_mean']:.4f} ± {row['cluster_r2_sd']:.4f} "
                f"| {row['cluster_ramp_mae_mw_mean']:.3f} ± {row['cluster_ramp_mae_mw_sd']:.3f} |"
            )
        return table

    lines = [
        "# Aggregate-only 集群功率预测实验结果",
        "",
        "本实验只比较新 aggregate-only 任务内重新训练的模型；所有数值均为独立 checkpoint 的多随机种子统计，不包含预测平均、KD或旧任务结果。",
        "",
        "## 冻结选择",
        "",
        f"- 历史基线范围：`{history_selection_data['selected_group']}`（{history_selection_data['history_visible_steps']}个15 min点）。",
        f"- correction状态：`{state_selection_data['selected_group']}`。",
        f"- 最终天气路线：`{final_route}`。",
        f"- 当前最强aggregate fusion baseline：`{best_fusion}`。",
        "",
        "## 3-seed 完整筛选",
        "",
        "| family | method | validation MAE/MW |",
        "|---|---|---:|",
    ]
    for row in screening_rows:
        lines.append(
            f"| {row['stage']} | {row['group']} | {row['cluster_mae_mw_mean']:.3f} ± {row['cluster_mae_mw_sd']:.3f} |"
        )
    lines += ["", "## 10-seed 正式结果", ""]
    lines += format_formal_table(val_rows, "2023 validation")
    lines += [""]
    lines += format_formal_table(test_rows, "2024 内部时间留出")
    val_compare = final_comparison["val"]
    test_compare = final_comparison["test"]
    lines += [
        "",
        f"最终路线相对最强 fusion baseline：validation ΔMAE={val_compare['mean_delta_mw']:+.3f} MW "
        f"（95% CI [{val_compare['ci_low_mw']:+.3f}, {val_compare['ci_high_mw']:+.3f}]，{val_compare['wins']}/10 seeds获胜）；"
        f"2024内部时间留出 ΔMAE={test_compare['mean_delta_mw']:+.3f} MW "
        f"（95% CI [{test_compare['ci_low_mw']:+.3f}, {test_compare['ci_high_mw']:+.3f}]）。负值表示最终路线更好。",
        "",
        "## 状态—天气机制",
        "",
        f"- validation MFull−M0：{mechanism['val']['m0']['mean_delta_mw']:+.3f} MW。",
        f"- validation MFull−MS：{mechanism['val']['ms']['mean_delta_mw']:+.3f} MW。",
        f"- validation MFull−MW：{mechanism['val']['mw']['mean_delta_mw']:+.3f} MW。",
        f"- validation MFull−MAdd：{mechanism['val']['madd']['mean_delta_mw']:+.3f} MW。",
        "",
        "只有 MFull−MAdd 也稳定为负，才把结果解释为状态调制天气响应；否则只解释为信息互补。",
    ]
    if state_intervention:
        lines += ["", "## 长历史状态干预", ""]
        for split, result in state_intervention.items():
            lines.append(
                f"- {split} 关闭 1 h 以前的 correction 状态后，ΔMAE={result['restricted_minus_native_mean_mw']:+.3f} ± "
                f"{result['sd_mw']:.3f} MW（95% CI [{result['ci_low_mw']:+.3f}, {result['ci_high_mw']:+.3f}]）。"
            )
    key_methods = list(dict.fromkeys([
        ("history", history_selection_data["selected_group"]),
        ("baselines", "w_only"), ("fusion", best_fusion), ("weather", final_route),
    ]))
    if benchmark:
        lines += [
            "", "## 单模型计算代价", "",
            "| method | Trainable params | Profiled GFLOPs/origin | Latency ms/origin | Peak MiB |",
            "|---|---:|---:|---:|---:|",
        ]
        for method in key_methods:
            result = benchmark[method]
            lines.append(
                f"| {method[1]} | {result['trainable_parameter_count']} | {result['profiled_gflops_per_origin']:.3f} "
                f"| {result['milliseconds_per_origin']:.3f} | {result['peak_allocated_mib']:.1f} |"
            )
    if regime_delta_rows:
        validation_regimes = [row for row in regime_delta_rows if row["split"] == "val"]
        best_regime = min(validation_regimes, key=lambda row: row["final_minus_fusion_mae_mw"])
        worst_regime = max(validation_regimes, key=lambda row: row["final_minus_fusion_mae_mw"])
        lines += [
            "", "## 分层稳健性", "",
            f"- 相对最强 fusion，validation 最大优势出现在 `{best_regime['regime']}/{best_regime['level']}`："
            f"ΔMAE={best_regime['final_minus_fusion_mae_mw']:+.3f} MW。",
            f"- 最弱分层为 `{worst_regime['regime']}/{worst_regime['level']}`："
            f"ΔMAE={worst_regime['final_minus_fusion_mae_mw']:+.3f} MW。",
        ]
    lines += [
        "", "## 产物索引", "",
        "完整逐 lead、3-seed/10-seed 表、paired CI、状态 attention、效率与分层结果位于 `artifacts/aggregate_cluster_v1/analysis/`。",
    ]
    document = PROJECT / "exp_docs" / "direct_cluster_forecasting_experiment_results.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the staged aggregate-only experiment matrix")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU indices")
    parser.add_argument(
        "--slots",
        default="1",
        help="Worker slots: auto, one positive integer, or a CSV matching --gpus (default: 1)",
    )
    args = parser.parse_args()
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain distinct physical GPU indices")
    totals = {gpu: gpu_state(gpu)[0] for gpu in gpus}
    slot_counts = parse_slot_counts(gpus, args.slots, totals)
    slots = build_worker_slots(gpus, slot_counts)
    print(
        f"[aggregate-scheduler] gpus={list(gpus)} slot_counts={list(slot_counts)} "
        f"workers={len(slots)} utilization_gate=disabled",
        flush=True,
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        ROOT / "run_manifest.json",
        {
            "gpus": list(gpus),
            "slots_specification": args.slots,
            "slot_counts": list(slot_counts),
            "worker_count": len(slots),
            "screen_seeds": list(SEEDS_SCREEN),
            "formal_seeds": list(SEEDS_FORMAL),
        },
    )
    snapshot = code_snapshot()
    snapshot_path = ROOT / "code_snapshot.json"
    if snapshot_path.is_file():
        previous = json.load(snapshot_path.open())
        if previous["sha256"] != snapshot["sha256"] and any(ROOT.glob("**/COMPLETE")):
            raise RuntimeError("Code changed after aggregate formal runs began")
        if previous["sha256"] != snapshot["sha256"]:
            atomic_json(snapshot_path, snapshot)
    else:
        atomic_json(snapshot_path, snapshot)
    visible, history_group, history_methods = history_selection(slots)
    state_mode, state_groups = state_selection(slots, visible, history_group)
    _, _, formal_methods = model_matrix(slots, visible, history_group, state_mode)
    test_methods = list(dict.fromkeys([*history_methods, *(('state', group) for group in state_groups), *formal_methods]))
    atomic_json(
        ROOT / "frozen_test_manifest.json",
        {"code_sha256": snapshot["sha256"], "methods": test_methods, "seeds": list(SEEDS_FORMAL)},
    )
    run_tests(slots, test_methods)
    selected_route = json.load((ROOT / "selection" / "model.json").open())["selected_route"]
    run_state_intervention(slots, selected_route, state_mode)
    run_benchmarks(slots, test_methods)
    run_regime_analysis()
    write_results(history_methods, state_groups, formal_methods)
    (ROOT / "COMPLETE").touch()
    print(f"[aggregate] all stages complete: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
