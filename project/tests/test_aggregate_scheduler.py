from __future__ import annotations

import threading
import time

from scripts.launch_aggregate_cluster_experiments import (
    GpuAdmissionController,
    WorkerSlot,
    build_worker_slots,
    execute_parallel,
    parse_slot_counts,
)


def test_auto_slots_resolve_49gb_to_three_and_24gb_to_two() -> None:
    gpus = (0, 1, 4, 5)
    totals = {0: 49140, 1: 49140, 4: 24564, 5: 24564}
    assert parse_slot_counts(gpus, "auto", totals) == (3, 3, 2, 2)
    assert parse_slot_counts(gpus, "2", totals) == (2, 2, 2, 2)
    assert parse_slot_counts(gpus, "3,2,1,1", totals) == (3, 2, 1, 1)


def test_worker_slot_labels_and_fixed_tqdm_positions() -> None:
    slots = build_worker_slots((0, 4), (3, 2))
    assert [slot.label for slot in slots] == [
        "gpu0.slot0", "gpu0.slot1", "gpu0.slot2", "gpu4.slot0", "gpu4.slot1",
    ]
    assert [slot.position for slot in slots] == [1, 2, 3, 4, 5]


def test_high_utilization_does_not_block_when_memory_is_sufficient() -> None:
    controller = GpuAdmissionController(
        {0: 1},
        state_provider=lambda _gpu: (49140, 40000, 100),
        reservation_seconds=10,
    )
    assert controller.try_acquire(0, False, "light") == 49140
    controller.release(0, False, "light")


def test_memory_thresholds_and_heavy_concurrency_limits() -> None:
    high_too_low = GpuAdmissionController(
        {0: 1}, state_provider=lambda _gpu: (49140, 30 * 1024 - 1, 0)
    )
    assert high_too_low.try_acquire(0, True, "heavy") is None
    low_too_low = GpuAdmissionController(
        {4: 1}, state_provider=lambda _gpu: (24564, 20 * 1024 - 1, 0)
    )
    assert low_too_low.try_acquire(4, True, "heavy") is None
    light_too_low = GpuAdmissionController(
        {4: 1}, state_provider=lambda _gpu: (24564, 6 * 1024 - 1, 0)
    )
    assert light_too_low.try_acquire(4, False, "light") is None

    high = GpuAdmissionController(
        {0: 3},
        state_provider=lambda _gpu: (49140, 49140, 100),
        reservation_seconds=0,
    )
    assert high.try_acquire(0, True, "h1") == 49140
    assert high.try_acquire(0, True, "h2") == 49140
    assert high.try_acquire(0, True, "h3") is None
    assert high.snapshot(0)["heavy_active"] == 2

    low = GpuAdmissionController(
        {4: 2},
        state_provider=lambda _gpu: (24564, 24564, 100),
        reservation_seconds=0,
    )
    assert low.try_acquire(4, True, "l1") == 24564
    assert low.try_acquire(4, True, "l2") is None
    assert low.try_acquire(4, False, "light") == 24564
    assert low.snapshot(4) == {"active": 2, "heavy_active": 1, "disabled_reason": ""}


def test_parallel_workers_share_one_gpu_without_duplicate_jobs_and_exit() -> None:
    slots = (WorkerSlot(0, 0, 1), WorkerSlot(0, 1, 2), WorkerSlot(0, 2, 3))
    jobs = [("light", seed) for seed in (42, 43, 44, 45, 46)]
    seen: list[tuple[str, int]] = []
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def runner(item: tuple[str, int], _slot: WorkerSlot, _total: int) -> int:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            seen.append(item)
            active -= 1
        return 0

    execute_parallel(
        "unit-test",
        jobs,
        slots,
        is_heavy=lambda _item: False,
        runner=runner,
        item_label=lambda item: f"{item[0]}/seed{item[1]}",
        state_provider=lambda _gpu: (49140, 49140, 99),
        reservation_seconds=0,
    )
    assert sorted(seen) == sorted(jobs)
    assert len(seen) == len(set(seen))
    assert maximum_active >= 2


def test_old_uniform_one_slot_mode_remains_available() -> None:
    assert parse_slot_counts((0, 1, 2), "1", {0: 49140, 1: 49140, 2: 49140}) == (1, 1, 1)
