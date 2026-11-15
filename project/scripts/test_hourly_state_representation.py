"""Exercise recovery safeguards with temporary processes, never real training jobs."""
import hashlib
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from hourly_state_representation import inspect_host, processes, write


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run = Path(self.temp.name)
        for name in ["operations", "schedulers", "code", "jobs"]:
            (self.run / name).mkdir()
        for name, lock in [("scheduler", "schedulers/scheduler.lock"),
                           ("coordinator", "operations/coordinator.lock")]:
            (self.run / "operations" / (name + ".py")).write_text(
                "import fcntl,time\nfrom pathlib import Path\n"
                f"f=(Path(__file__).resolve().parents[1]/{lock!r}).open('a+')\n"
                "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\ntime.sleep(60)\n")
        (self.run / "code/train.py").write_text("import time\ntime.sleep(60)\n")
        write(self.run / "operations/CHECKS.json", {"passed": True})
        write(self.run / "operations/ASSIGNMENTS.json", {"failed_job": "local4090"})
        write(self.run / "SCIENCE_FREEZE.json", {"files": {}})
        write(self.run / "OPERATIONS_FREEZE.json", {"files": {
            str(p.relative_to(self.run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (self.run / "operations").glob("*.py")}})

    def tearDown(self):
        for pids in processes(self.run).values():
            for pid in pids:
                try:
                    import os
                    os.kill(pid, 15)
                    os.waitpid(pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
        self.temp.cleanup()

    def check(self, recover=True):
        return inspect_host(self.run, "local4090", Path(sys.executable), recover)

    def test_missing_components_recover_once(self):
        report = self.check()
        self.assertEqual(len(report["actions"]), 2)
        self.assertTrue(all(x["survived_initial_second"] for x in report["actions"]))
        self.assertEqual(self.check()["actions"], [])
        self.assertEqual(processes(self.run), report["processes"])

    def test_stop_is_respected(self):
        write(self.run / "operations/STOP.json", {})
        self.assertTrue(all(x["reason"] == "completed_or_explicitly_stopped"
                            for x in self.check()["actions"]))
        self.assertFalse(any(processes(self.run).values()))

    def test_complete_is_respected(self):
        write(self.run / "COMPLETE.json", {"training_completed": 35})
        self.assertTrue(all(x["reason"] == "completed_or_explicitly_stopped"
                            for x in self.check()["actions"]))
        self.assertFalse(any(processes(self.run).values()))

    def test_held_scheduler_lock_prevents_duplicate_launch(self):
        with (self.run / "schedulers/scheduler.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = self.check(recover=False)
            self.assertFalse(any(x["component"] == "scheduler" for x in report["actions"]))

    def test_failure_circuit_is_not_bypassed(self):
        write(self.run / "jobs/failed_job/FAILURE.json", {})
        report = self.check(recover=False)
        scheduler = next(x for x in report["actions"] if x["component"] == "scheduler")
        self.assertIn("failure_requires_inspection", scheduler["reason"])

    def test_orphans_finish_before_scheduler_recovery(self):
        worker = subprocess.Popen([sys.executable, str(self.run / "code/train.py")])
        try:
            time.sleep(.1)
            report = self.check(recover=False)
            scheduler = next(x for x in report["actions"] if x["component"] == "scheduler")
            self.assertIn("orphan_workers_still_running", scheduler["reason"])
            self.assertIsNone(worker.poll())
        finally:
            worker.terminate()
            worker.wait()

    def test_frozen_code_change_prevents_recovery(self):
        (self.run / "operations/scheduler.py").write_text("raise RuntimeError('changed')\n")
        report = self.check()
        self.assertEqual(len([x for x in report["warnings"] if "Frozen file changed" in x]), 2)
        self.assertFalse(any(processes(self.run).values()))

    def test_repeated_restarts_are_bounded(self):
        write(self.run / "monitoring/restart_history.json", [
            {"component": name, "epoch": time.time()} for name in ["scheduler", "coordinator"] for _ in range(2)])
        self.assertTrue(all(x["reason"] == "restart_limit_2_per_6_hours" for x in self.check()["actions"]))
        self.assertFalse(any(processes(self.run).values()))


if __name__ == "__main__":
    unittest.main()
