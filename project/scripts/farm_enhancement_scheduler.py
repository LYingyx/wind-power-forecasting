"""Single-owner scheduler with live memory reservations and durable progress."""
from __future__ import annotations
import json
import fcntl
import os
from pathlib import Path
import subprocess
import time

from launch_aggregate_cluster_experiments import atomic_json, gpu_state, _looks_like_oom


def admission_free(live_free: int, initial_free: int, reserved: int) -> int:
    # A reservation lasts for the entire job, including startup and validation.
    # This never assumes that a process has allocated its peak within 10 s.
    return min(live_free, initial_free - reserved)


def gpu_process_memory() -> dict[int, int]:
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                    "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in text.splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            result[int(fields[0])] = result.get(int(fields[0]), 0) + int(fields[1])
    return result


def child_gpu_memory(root_pid: int, usage: dict[int, int]) -> int:
    total = 0
    for pid, amount in usage.items():
        current = pid
        for _ in range(64):
            if current == root_pid:
                total += amount
                break
            if current <= 1:
                break
            try:
                current = int(Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()[1])
            except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
                break
    return total


def unallocated_reservation(budget: int, observed: int) -> int:
    return max(0, budget - observed)


def job_is_locked(job) -> bool:
    path = job.run_dir / "job.lock"
    if not path.exists():
        return False
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def execute(label, jobs, slots, root: Path, command, complete, *, memory_mib: int) -> None:
    # A launcher restart may leave its workers alive. Let their run locks drain
    # before admitting new work; then discover the newly completed artifacts.
    last_recovery_report = 0.
    while True:
        recovering = [job.key for job in jobs if job_is_locked(job)]
        if not recovering:
            break
        if time.time() - last_recovery_report >= 30:
            atomic_json(root / "status.json", {"state":"recovering-existing-workers", "label":label,
                        "launcher_pid":os.getpid(),"updated_unix":time.time(),"jobs":recovering})
            print(f"[recovery] waiting for {len(recovering)} existing workers",flush=True)
            last_recovery_report = time.time()
        time.sleep(5)
    pending = [job for job in jobs if not complete(job)]
    done = len(jobs) - len(pending)
    if not pending:
        return
    initial = {gpu: gpu_state(gpu)[1] for gpu in dict.fromkeys(slot.gpu for slot in slots)}
    active = {}
    disabled = set()
    failures = []
    attempts = {}
    last_report = 0.0
    started = time.time()
    while pending or active:
        for position, record in list(active.items()):
            process, job, slot, log, launched = record
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[position]
            valid = code == 0 and complete(job)
            if valid:
                done += 1
                print(f"[done] {job.stage}/{job.group}/seed{job.seed}/{job.split} seconds={time.time()-launched:.1f}", flush=True)
            else:
                worker_path = job.run_dir / "worker.json"
                attempt = Path(json.loads(worker_path.read_text())["attempt"]) if worker_path.exists() else job.run_dir
                oom = _looks_like_oom(attempt / "run.log", code or -1)
                if oom:
                    disabled.add(slot.gpu)
                failures.append({"job": job.key, "gpu": slot.gpu, "code": code, "oom": oom, "attempt": str(attempt)})
                if attempts[job.key] < 2:
                    pending.append(job)
                else:
                    disabled.update(initial)  # Drain active runs before failing the stage.
        usage = gpu_process_memory()
        for slot in slots:
            if not pending or slot.position in active or slot.gpu in disabled:
                continue
            reserved = sum(unallocated_reservation(memory_mib, child_gpu_memory(record[0].pid, usage))
                           for record in active.values() if record[2].gpu == slot.gpu)
            free = gpu_state(slot.gpu)[1] - reserved
            if free < memory_mib + 1024:
                continue
            job = pending.pop(0)
            attempts[job.key] = attempts.get(job.key, 0) + 1
            job.run_dir.mkdir(parents=True, exist_ok=True)
            log = (job.run_dir / f"dispatch_{job.split}_{time.time_ns()}.log").open("w")
            environment = dict(os.environ, WIND_FARM_INTERACTION_ROOT=str(root),
                               WIND_PROGRESS_PREFIX=f"[{slot.label}:{job.group}:s{job.seed}]",
                               WIND_PROGRESS_POSITION="0", PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command(job, slot), env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[slot.position] = (process, job, slot, log, time.time())
            print(f"[dispatch] {slot.label} pid={process.pid} job={job.key} reserve_mib={memory_mib}", flush=True)
        now = time.time()
        if now - last_report >= 30 or not active:
            status = {"label": label, "launcher_pid": os.getpid(), "updated_unix": now,
                      "stage_total": len(jobs), "completed": done, "active": len(active),
                      "queued": len(pending), "failed_attempts": len(failures), "failures": failures,
                      "memory_budget_mib": memory_mib, "disabled_gpus": sorted(disabled),
                      "stage_elapsed_seconds": now - started,
                      "workers": [{"gpu": slot.gpu, "slot": slot.slot, "pid": process.pid,
                                   "job": job.key, "started_unix": launched}
                                  for process, job, slot, _, launched in active.values()]}
            atomic_json(root / "status.json", status)
            print(f"[status] stage={label} done={done}/{len(jobs)} active={len(active)} queued={len(pending)} failed_attempts={len(failures)}", flush=True)
            last_report = now
        if not active and pending and len(disabled) == len(initial):
            raise RuntimeError(f"No usable GPUs remain: {failures}")
        if pending or active:
            time.sleep(5)
    if done != len(jobs):
        raise RuntimeError(f"Stage incomplete: {done}/{len(jobs)}; {failures}")
