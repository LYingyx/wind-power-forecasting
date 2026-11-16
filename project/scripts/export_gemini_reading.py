"""Export auditable training records and uncompressed UTF-8 review files.

No model training, checkpoint loading, or source-code modification is performed.
Run with --refresh-training on the research server to refresh job provenance.
Run with --check-only --revision HEAD to check exactly the committed snapshot.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["daily_protocol_restart_20261114_v1", "daily_ablation_restart_20261114_v1",
        "daily_site_finetune_20261114_v1"]
ACTIVE_RUN = "daily_state_representation_20261116_v2"
REVIEW = "project/analysis/daily_protocol_review_20261115/"
EXPORT = "docs/gemini_plain_text/"
FINGERPRINT = "bb838b5c77790c527bea81bacbffd12fd9f626d199b502c2cfb435dcc5f1c036"
CONTEXT = ["README.md", "docs/GEMINI_READING_GUIDE_CN.md",
           "docs/RESULTS_TO_CODE_CN.md",
           "project/MANUSCRIPT_PROTOCOL_ALIGNMENT_20261114_CN.md",
           "project/DAILY_PROTOCOL_RESULTS_REVIEW_20261115_CN.md",
           "project/ENCODING_ROLE_CLARIFICATION_20261115_CN.md",
           "project/STATE_CONDITIONED_REPRESENTATION_PLAN_20261115_CN.md",
           REVIEW + "COMPLETION_AUDIT.json", REVIEW + "COMPARISONS.json",
           REVIEW + "group_summary.csv", REVIEW + "comparison_summary.csv"]
PIPELINE = ["project/pyproject.toml", "project/src/wind_forcing/data/operational.py",
            "project/src/wind_forcing/data/operational_farm.py",
            "project/scripts/prepare_operational_pangu.py",
            "project/scripts/merge_verified_pangu_crops.py",
            "project/scripts/audit_operational_pangu.py",
            "project/scripts/review_daily_results_20261115.py",
            "project/scripts/render_daily_review_20261115.py",
            "project/scripts/hourly_state_representation.py",
            "project/scripts/test_hourly_state_representation.py",
            "project/scripts/audit_stateless_noise_repair_20261116.py",
            "project/configs/operational_pangu_d1_08_lag15h.json"]
DETAILS = [REVIEW + name for name in ["FARM_AND_SEED_DETAILS_CN.md",
           "per_farm_comparisons.csv", "per_seed_comparisons.csv",
           "per_seed_summary.csv", "horizon_comparisons.csv"]]


def git(*args, data=None):
    return subprocess.check_output(["git", *args], cwd=ROOT, input=data)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def snapshot(revision=None):
    if revision is None:
        names = set(git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
                    .decode().strip("\0").split("\0"))
        result = {}
        for name in sorted(names):
            path = ROOT / name
            if path.is_symlink():
                raise ValueError(f"Review snapshot must not contain symlinks: {name}")
            result[name] = path.read_bytes()
        return result
    objects = []
    for row in git("ls-tree", "-r", "-z", revision).decode().split("\0"):
        if not row:
            continue
        metadata, name = row.split("\t", 1)
        mode, kind, oid = metadata.split()
        if kind != "blob" or mode == "120000":
            raise ValueError(f"Unsupported review entry: {name}")
        objects.append((name, oid))
    raw = git("cat-file", "--batch", data=("\n".join(x[1] for x in objects) + "\n").encode())
    cursor = 0
    result = {}
    for name, oid in objects:
        end = raw.index(b"\n", cursor)
        actual_oid, kind, size = raw[cursor:end].decode().split()
        assert actual_oid == oid and kind == "blob"
        cursor = end + 1
        result[name] = raw[cursor:cursor + int(size)]
        cursor += int(size) + 1
    assert cursor == len(raw)
    return result


def refresh_training():
    """Publish all jobs' selection evidence without copying weights or predictions."""
    for run in RUNS:
        base = ROOT / "project/runs" / run
        planned = json.loads((base / "manifest.json").read_text())
        records = []
        for spec in planned:
            job = base / "jobs" / spec["id"]
            originals = {}
            def read(name):
                path = job / name
                raw = path.read_bytes()
                originals[name] = {"source": path.relative_to(ROOT).as_posix(),
                                   "sha256": digest(raw)}
                return json.loads(raw)
            metrics, history, success = read("metrics.json"), read("history.json"), read("SUCCESS.json")
            assert success["passed"] and not metrics["smoke_only"]
            assert metrics["requested_spec"] == spec
            assert spec["preprocess_fingerprint"] == FINGERPRINT
            def summary(value):
                return {k: value[k] for k in ["mae", "rmse", "high_power_rmse", "horizons"] if k in value}
            record = {k: metrics[k] for k in ["requested_spec", "spec", "shared_spec",
                      "evaluation_year", "validation_selection", "checkpoint_epoch",
                      "checkpoint_sha256", "source_sha256", "parameter_count",
                      "trainable_parameter_count", "trained_epochs"] if k in metrics}
            record.update(job_id=spec["id"], metrics_summary=summary(metrics["metrics"]),
                          history=history, source_files=originals)
            choices = [(row["epoch"], row["validation_mae"]) for row in history]
            if "baseline_metrics" in metrics:
                record["baseline_metrics_summary"] = summary(metrics["baseline_metrics"])
                choices.insert(0, (0, metrics["baseline_metrics"]["mae"]))
            assert choices and min(choices, key=lambda x: x[1])[0] == metrics["checkpoint_epoch"], spec["id"]
            for epoch in [1, 6]:
                if (job / f"epoch{epoch}.json").exists():
                    fixed = read(f"epoch{epoch}.json")
                    record[f"epoch{epoch}"] = {"epoch": epoch, "metrics_summary": summary(fixed["metrics"]),
                                              "checkpoint_sha256": fixed["checkpoint_sha256"]}
            if (job / "Z1.json").exists():
                z1 = read("Z1.json")
                record["Z1_bin30_summary"] = {k: summary(v["bin30"]) for k, v in z1["analyses"].items()}
            records.append(record)
        write_json(ROOT / REVIEW / "provenance" / (run + ".json"),
                   {"schema_version": 1, "run": run, "job_count": len(records),
                    "note": "Exact specs and epoch histories; metric summaries are projections of original JSON. Full weights, predictions and original JSON bodies are not included. Source hashes identify the server originals.",
                    "jobs": records})
        print(f"Exported training provenance: {run}: {len(records)} jobs")


def groups(files):
    code, pipeline = [], list(PIPELINE)
    active_prefix = f"project/runs/{ACTIVE_RUN}/"
    extra = [ACTIVE_RUN] if active_prefix + "manifest.json" in files else []
    for run in RUNS + extra:
        prefix = f"project/runs/{run}/"
        code += sorted(p for p in files if p.startswith(prefix + "code/") and p.endswith(".py"))
        pipeline += [prefix + "PROTOCOL.json", prefix + "SCIENCE_FREEZE.json"]
        pipeline += sorted(p for p in files if p.startswith(prefix + "assets/selected/") and p.endswith(".json"))
        pipeline += sorted(p for p in files if p in [prefix + "operations/collect.py", prefix + "operations/pipeline.py"])
    if extra:
        pipeline += [active_prefix + name for name in ["README_CN.md", "HOURLY_MONITOR_CN.md", "STATUS_REVIEW_20261116_CN.md", "NOISE_ENDPOINT_REPAIR.json", "NOISE_REPAIR_CHECKS_LOCAL.json", "NOISE_REPAIR_CHECKS_REMOTE.json", "manifest.json", "diagnostic_manifest.json", "OPERATIONS_FREEZE.json"]]
        pipeline += [active_prefix + "MIGRATION_4090_20261116.json", active_prefix + "operations/ASSIGNMENTS.json", active_prefix + "operations/RESOURCES.json"]
        pipeline += [active_prefix + "LOSS_TF_PRECOMPUTE_20261116.json"]
        pipeline += [active_prefix + "analysis/" + name for name in ["LEARNINGS_20261116_CN.md", "LEARNINGS_EVIDENCE_20261116.json"]]
        pipeline += sorted(p for p in files if p.startswith(active_prefix + "operations/") and p.endswith(".py")
                           and p != active_prefix + "operations/collect.py")
    pipeline += sorted(p for p in files if any(p.startswith(f"project/runs/{name}/") for name in
                       ["rmse_gap_diagnostic_20261115", "ramp_information_diagnostic_20261115"])
                       and p.endswith((".py", ".md")))
    return [("01_context_and_results.txt", CONTEXT), ("02_models_and_training.txt", code),
            ("03_farm_seed_horizon_metrics.txt", DETAILS), ("04_data_selection_evaluation.txt", pipeline)]


def check_core(files):
    assert len(files) <= 5000 and sum(map(len, files.values())) <= 100_000_000, "Gemini import limits exceeded"
    for name, raw in files.items():
        assert not raw.startswith(b"PK\x03\x04"), f"ZIP content in {name}"
        if name.endswith(".png"):
            assert raw.startswith(b"\x89PNG\r\n\x1a\n"), name
            continue
        text = raw.decode("utf-8")
        assert "\x00" not in text, f"Binary NUL in {name}"
    for _, names in groups(files):
        for name in names:
            assert name in files, f"Required source missing: {name}"
    active_prefix = f"project/runs/{ACTIVE_RUN}/"
    if active_prefix + "SCIENCE_FREEZE.json" in files:
        for rel, sha in json.loads(files[active_prefix + "SCIENCE_FREEZE.json"])["files"].items():
            if rel.startswith("code/") and rel.endswith(".py"):
                assert digest(files[active_prefix + rel]) == sha, (ACTIVE_RUN, rel)
        for rel, sha in json.loads(files[active_prefix + "OPERATIONS_FREEZE.json"])["files"].items():
            assert digest(files[active_prefix + rel]) == sha, (ACTIVE_RUN, rel)
    total_jobs = 0
    reported = {}
    for run in RUNS:
        specs = json.loads(files[f"project/runs/{run}/manifest.json"])
        freeze = json.loads(files[f"project/runs/{run}/SCIENCE_FREEZE.json"])
        for rel, sha in freeze["files"].items():
            if rel.startswith("code/") and rel.endswith(".py"):
                assert digest(files[f"project/runs/{run}/" + rel]) == sha, (run, rel)
        records = json.loads(files[REVIEW + "provenance/" + run + ".json"])
        assert records["job_count"] == len(specs) == len(records["jobs"])
        assert {x["id"] for x in specs} == {x["job_id"] for x in records["jobs"]}
        by_id = {x["id"]: x for x in specs}
        for row in records["jobs"]:
            assert row["requested_spec"] == by_id[row["job_id"]]
            spec = row["requested_spec"]
            assert spec["preprocess_fingerprint"] == FINGERPRINT
            choices = [(x["epoch"], x["validation_mae"]) for x in row["history"]]
            if "baseline_metrics_summary" in row:
                choices.insert(0, (0, row["baseline_metrics_summary"]["mae"]))
            assert min(choices, key=lambda x: x[1])[0] == row["checkpoint_epoch"], row["job_id"]
            source = f"runs/{run}/jobs/{row['job_id']}/"
            if spec["stage"] not in ["tune", "reference"]:
                reported[(spec["group"], source + "metrics.json")] = (row["metrics_summary"], row["checkpoint_epoch"])
            if spec["stage"] == "personalization":
                for epoch in [1, 6]:
                    reported[(spec["group"] + f"_E{epoch}", source + f"epoch{epoch}.json")] = (row[f"epoch{epoch}"]["metrics_summary"], epoch)
                if spec["mode"] == "P1_FULL":
                    for key, mode, epoch in [("p0", "P0_SHARED_Z1", 0), ("e1", "P1_FULL_E1_Z1", 1)]:
                        reported[(spec["family"] + "_" + mode, source + "Z1.json")] = (row["Z1_bin30_summary"][key], epoch)
        total_jobs += len(specs)
    assert total_jobs == 2571
    def rows(name):
        return list(csv.DictReader(io.StringIO(files[REVIEW + name].decode())))
    raw_groups = collections.defaultdict(list)
    source_groups = collections.defaultdict(list)
    for row in rows("all_farm_seed_metrics.csv"):
        raw_groups[row["group"]].append(row)
        source_groups[(row["group"], row["source"])].append(row)
    assert set(reported) == set(source_groups), "Training provenance / metric table sources differ"
    for key, values in source_groups.items():
        summary, epoch = reported[key]
        assert all(int(x["selected_epoch"]) == epoch for x in values), key
        for metric in ["mae", "rmse", "high_power_rmse"]:
            selected = [float(x[metric]) for x in values if metric != "high_power_rmse" or int(x["high_count"]) > 0]
            assert abs(sum(selected) / len(selected) - summary[metric]) < 1e-12, (key, metric)
    summaries = {x["group"]: x for x in rows("group_summary.csv")}
    assert set(summaries) == set(raw_groups)
    for name, values in raw_groups.items():
        assert len(values) == 275
        assert {(int(x["seed"]), int(x["station"])) for x in values} == {(seed, station) for seed in range(42, 47) for station in range(55)}
        for key in ["mae", "rmse", "high_power_rmse"]:
            selected = [float(x[key]) for x in values if key != "high_power_rmse" or int(x["high_count"]) > 0]
            assert abs(sum(selected) / len(selected) - float(summaries[name][key])) < 1e-12, (name, key)
    for row in rows("comparison_summary.csv"):
        for key in ["mae", "rmse", "high_power_rmse"]:
            value = 100 * (float(summaries[row["candidate"]][key]) / float(summaries[row["reference"]][key]) - 1)
            assert abs(value - float(row[key + "_change_percent"])) < 1e-9


def build(files):
    manifest = {"format": "uncompressed UTF-8; upload individual TXT files, not ZIP",
                "source_base_commit": git("rev-parse", "HEAD").decode().strip(), "bundles": []}
    for filename, names in groups(files):
        by_hash = {}
        for name in names:
            by_hash.setdefault(digest(files[name]), []).append(name)
        parts = [f"# {filename}\n\nUTF-8 纯文本，无需解压。来源：https://github.com/LYingyx/wind-power-forecasting\n"
                 "按完全相同的源文件内容去重；各原路径和 SHA-256 均列出，运行时仍使用各自原路径。\n"
                 "这是阅读副本；训练记录见 project/analysis/daily_protocol_review_20261115/provenance/。\n\n"]
        for sha, aliases in by_hash.items():
            text = files[aliases[0]].decode()
            fence = "`" * max(3, 1 + max(map(len, re.findall(r"`+", text)), default=0))
            parts += ["## SOURCE: " + aliases[0] + "\n\n" + "\n".join("- " + p for p in aliases)
                      + f"\n\nSHA-256: {sha}\n\n{fence}\n", text,
                      ("" if text.endswith("\n") else "\n") + fence + "\n\n"]
        content = "".join(parts).encode()
        path = ROOT / EXPORT / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["bundles"].append({"path": EXPORT + filename, "bytes": len(content),
                                   "sha256": digest(content), "sources": {p: digest(files[p]) for p in names}})
    write_json(ROOT / EXPORT / "MANIFEST.json", manifest)


def check_exports(files):
    manifest = json.loads(files[EXPORT + "MANIFEST.json"])
    expected = {EXPORT + name: set(paths) for name, paths in groups(files)}
    assert {x["path"] for x in manifest["bundles"]} == set(expected)
    for item in manifest["bundles"]:
        raw = files[item["path"]]
        assert digest(raw) == item["sha256"] and len(raw) == item["bytes"]
        assert set(item["sources"]) == expected[item["path"]]
        for name, sha in item["sources"].items():
            assert digest(files[name]) == sha, f"Stale export; regenerate after changing {name}"
            assert files[name] in raw, f"Source body absent from export: {name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-training", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--revision", help="Read committed Git objects, e.g. HEAD, instead of working files")
    args = parser.parse_args()
    if args.revision and (not args.check_only or args.refresh_training):
        parser.error("--revision requires --check-only and cannot refresh training")
    if args.refresh_training:
        refresh_training()
    files = snapshot(args.revision)
    check_core(files)
    if not args.check_only:
        build(files)
        files = snapshot()
        check_core(files)
    check_exports(files)
    print(json.dumps({"passed": True, "files": len(files), "bytes": sum(map(len, files.values())),
                      "training_jobs": 2571, "utf8_and_source_body_checks": True,
                      "selection_and_summary_checks": True, "gemini_client_import_tested": False}))


if __name__ == "__main__":
    main()
