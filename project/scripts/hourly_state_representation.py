"""Hourly supervision of the frozen v2 run; scientific gates stay in coordinator.py."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

RUN_NAME = "daily_state_representation_20261116_v2"
PROJECTS = {
    "local4090": Path("/shared_data_ssd2/liying/wind_cluster_project_20261018/project"),
    "v1006001": Path("/home/liying/wind_physics_project/project"),
}


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def age(path):
    return round(time.time() - path.stat().st_mtime, 1) if path.exists() else None


def processes(run):
    """Match our own user and exact absolute script argument, never just a PID."""
    paths = {str(run / "operations" / (name + ".py")): name
             for name in ("scheduler", "coordinator")}
    paths.update({str(run / "code" / name): "workers"
                  for name in ("train.py", "reference_evaluation.py")})
    result = {name: [] for name in ("scheduler", "coordinator", "workers")}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / "cmdline").read_bytes().decode().split("\0")
            for path, name in paths.items():
                if path in args:
                    result[name].append(int(proc.name))
        except (OSError, UnicodeError):
            continue
    return result


def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def verify_freeze(run):
    if not read(run / "operations/CHECKS.json", {}).get("passed"):
        raise RuntimeError("Startup checks have not passed")
    for name in ("SCIENCE_FREEZE.json", "OPERATIONS_FREEZE.json"):
        for rel, expected in read(run / name)["files"].items():
            digest = hashlib.sha256()
            with (run / rel).open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(f"Frozen file changed: {rel}")


def inspect_host(run, host, python, recover=True):
    directory = run / "monitoring"
    directory.mkdir(exist_ok=True)
    with (directory / "host.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = read(run / "schedulers/status.json", {})
        assigned = read(run / "operations/ASSIGNMENTS.json", {})
        failures = sorted({*status.get("failed", []), *[
            p.parent.name for p in (run / "jobs").glob("*/FAILURE.json")
            if assigned.get(p.parent.name) == host]})
        stopped = (run / "operations/STOP.json").exists()
        complete = (run / "COMPLETE.json").exists()
        recent = read(directory / "restart_history.json", [])
        report = dict(host=host, timestamp=now(), stopped=stopped, complete=complete,
                      scheduler_age_seconds=age(run / "schedulers/status.json"),
                      failures=failures, actions=[], warnings=[])
        for name in (["coordinator", "scheduler"] if host == "local4090" else ["scheduler"]):
            found = processes(run)
            lockpath = run / ("operations/coordinator.lock" if name == "coordinator"
                              else "schedulers/scheduler.lock")
            held = locked(lockpath)
            if found[name] or held:
                if len(found[name]) > 1:
                    report["warnings"].append(f"Multiple {name} processes: {found[name]}")
                continue
            reason = None
            if complete or stopped or status.get("stopping"):
                reason = "completed_or_explicitly_stopped"
            elif name == "scheduler" and failures:
                reason = "failure_requires_inspection; circuit breaker preserved"
            elif name == "scheduler" and found["workers"]:
                # The frozen scheduler cannot adopt orphaned workers. Let them finish;
                # their inherited job locks remain authoritative until they exit.
                reason = "orphan_workers_still_running; waiting_for_safe_recovery"
            elif sum(x["component"] == name and time.time() - x["epoch"] < 21600
                     for x in recent) >= 2:
                reason = "restart_limit_2_per_6_hours"
            elif not recover:
                reason = "read_only_check"
            if reason:
                report["actions"].append(dict(component=name, action="not_restarted", reason=reason))
                continue
            try:
                verify_freeze(run)
                # Save the attempt before spawning, so failed launches are bounded too.
                record = dict(component=name, epoch=time.time(), timestamp=now())
                recent.append(record)
                write(directory / "restart_history.json", recent)
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1")
                args = [str(python), "-u", str(run / "operations" / (name + ".py"))]
                if name == "scheduler":
                    args += ["--host", host]
                with (run / "operations" / (name + ".log")).open("a") as log:
                    process = subprocess.Popen(args, cwd=run, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                time.sleep(1)
                record.update(pid=process.pid, survived_initial_second=process.poll() is None)
                write(directory / "restart_history.json", recent)
                report["actions"].append(dict(action="restart_attempt", **record))
            except Exception as exc:
                report["warnings"].append(f"{name} recovery refused: {exc!r}")
        report["processes"] = processes(run)
        if not complete and not stopped:
            if report["scheduler_age_seconds"] is None or report["scheduler_age_seconds"] > 300:
                report["warnings"].append("Scheduler heartbeat missing or older than 5 minutes")
            if status.get("error"):
                report["warnings"].append(str(status["error"]))
            if failures:
                report["warnings"].append("Failed jobs are preserved for inspection")
        return report


def hourly():
    project = PROJECTS["local4090"]
    run = project / "runs" / RUN_NAME
    directory = run / "monitoring"
    directory.mkdir(exist_ok=True)
    with (directory / "hourly.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another hourly check is running; skipped.")
            return
        previous = read(run / "HOURLY_STATUS.json", {})
        report = dict(timestamp=now(), run=RUN_NAME, interval_seconds=3600, hosts={}, warnings=[])
        complete = read(run / "COMPLETE.json")
        report["complete"] = complete
        if complete is None:
            for host, base in PROJECTS.items():
                try:
                    if host == "local4090":
                        report["hosts"][host] = inspect_host(run, host, base / ".runtime/chronos/bin/python")
                    else:
                        command = "python3 - --host " + shlex.quote(host)
                        result = subprocess.run(
                            ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                             "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", host, command],
                            input=Path(__file__).read_text(), text=True, capture_output=True,
                            check=True, timeout=90)
                        report["hosts"][host] = json.loads(result.stdout)
                except Exception as exc:
                    report["warnings"].append(f"{host}: {exc!r}")
        status = read(run / "STATUS.json", {})
        gates = read(run / "analysis/GATES.json", {})
        report["progress"] = {k: status.get(k) for k in ["timestamp", "currently_enabled_training_jobs",
                              "completed_training", "diagnostic_evaluations_completed", "failures", "errors"]}
        report["active_by_host"] = dict(Counter(x["host"] for x in status.get("active", [])))
        report["gates"] = gates
        report["newly_enabled_stages"] = [k for k in ["stage2_enabled", "cross_backbone_enabled", "loss_enabled"]
                                           if gates.get(k) and not previous.get("gates", {}).get(k)]
        report["coordinator_age_seconds"] = age(run / "STATUS.json")
        if complete is None and (report["coordinator_age_seconds"] is None or report["coordinator_age_seconds"] > 300):
            report["warnings"].append("Coordinator heartbeat missing or older than 5 minutes")
        for host, state in report["hosts"].items():
            report["warnings"] += [f"{host}: {warning}" for warning in state["warnings"]]
            report["warnings"] += [f"{host}: {action['component']}: {action.get('reason', 'restart_failed')}"
                                   for action in state["actions"]
                                   if action["action"] == "not_restarted" and not state["stopped"]
                                   or action["action"] == "restart_attempt" and not action["survived_initial_second"]]
        report["warnings"] += [str(x) for x in status.get("errors", [])]
        # Do not call collect() here: the single locked coordinator owns every gate mutation.
        write(run / "HOURLY_STATUS.json", report)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        write(directory / (stamp + ".json"), report)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=list(PROJECTS))
    args = parser.parse_args()
    if args.host:
        base = PROJECTS[args.host]
        print(json.dumps(inspect_host(base / "runs" / RUN_NAME, args.host,
                                      base / ".runtime/chronos/bin/python"), ensure_ascii=False))
    else:
        hourly()
