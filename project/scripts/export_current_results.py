"""Publish a read-only, audited snapshot of the active experiment's results."""
from __future__ import annotations

import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "project/runs/daily_state_representation_20261116_v2"
OUT = RUN / "analysis"
SEEDS = set(range(42, 47))
METRICS = ("mae", "rmse", "high_power_rmse")
ARRAYS = ("per_farm_mae", "per_farm_rmse", "high_power_per_farm_rmse")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path, sources):
    raw = path.read_bytes()
    sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def selected_epoch(history, record):
    chosen = min(history, key=lambda x: x["validation_mae"])
    assert chosen["epoch"] == record["checkpoint_epoch"]
    assert abs(chosen["validation_mae"] - record["metrics"]["mae"]) < 1e-9


def comparison(groups, a, b, seeds):
    result = dict(candidate=a, reference=b, seeds=sorted(seeds),
                  complete_five_seeds=seeds == SEEDS)
    for metric, array in zip(METRICS, ARRAYS):
        x = [groups[a][s]["metrics"][metric] for s in sorted(seeds)]
        y = [groups[b][s]["metrics"][metric] for s in sorted(seeds)]
        result[metric + "_change_percent"] = 100 * (statistics.mean(x) / statistics.mean(y) - 1)
        result[metric + "_per_seed_change_percent"] = [100 * (u / v - 1) for u, v in zip(x, y)]
        result[metric + "_farm_wins"] = sum(
            statistics.mean(groups[a][s]["metrics"][array][i] for s in seeds)
            < statistics.mean(groups[b][s]["metrics"][array][i] for s in seeds)
            for i in range(55))
    result["both_metrics_seed_wins"] = sum(
        all(groups[a][s]["metrics"][m] < groups[b][s]["metrics"][m] for m in METRICS[:2])
        for s in seeds)
    return result


def main():
    sources = {}
    specs = read(RUN / "manifest.json", sources)
    diagnostic_specs = read(RUN / "diagnostic_manifest.json", sources)
    cohort = read(RUN / "assets/COHORT.json", sources)
    for name in ("SCIENCE_FREEZE.json", "OPERATIONS_FREEZE.json"):
        freeze = read(RUN / name, sources)
        for rel, expected in freeze["files"].items():
            assert sha(RUN / rel) == expected, rel
    groups = collections.defaultdict(dict)
    records, diagnostics, references, pending = [], [], [], []
    for spec in specs + diagnostic_specs:
        job = RUN / "jobs" / spec["id"]
        if not (job / "SUCCESS.json").exists():
            if spec in specs:
                item = dict(id=spec["id"], group=spec["group"], seed=spec["seed"],
                            status="pending", final_metrics_available=False)
                for name in ("progress.json", "history.json", "FAILURE.json"):
                    if (job / name).exists():
                        item[name.removesuffix(".json")] = read(job / name, sources)
                if "FAILURE" in item:
                    item["status"] = "failed"
                elif "progress" in item:
                    item["status"] = "in_progress_not_final"
                elif not all((RUN / d).exists() for d in spec["dependencies"]):
                    item["status"] = "waiting_for_dependency"
                pending.append(item)
            continue
        local_sources = {}
        success = read(job / "SUCCESS.json", local_sources)
        value = read(job / "metrics.json", local_sources)
        assert success["passed"] and not value["smoke_only"]
        assert value["requested_spec"] == spec and value["evaluation_year"] == 2023
        is_diagnostic = spec.get("task") == "reference_evaluation"
        weight = RUN / spec["checkpoint"] if is_diagnostic else job / "best.pt"
        assert sha(weight) == value["checkpoint_sha256"], spec["id"]
        item = dict(id=spec["id"], group=spec["group"], seed=spec["seed"],
                    record=value, success=success, checkpoint_hash_verified=True)
        if is_diagnostic:
            assert value["no_training"]
            diagnostics.append(item)
        else:
            item["history"] = read(job / "history.json", local_sources)
            selected_epoch(item["history"], value)
            records.append(item)
            groups[spec["group"]][spec["seed"]] = value
        item["source_files"] = local_sources
    for label in ("R0", "TF_R3"):
        for seed in sorted(SEEDS):
            local_sources = {}
            value = read(RUN / "reference_metrics" / label / str(seed) / "metrics.json", local_sources)
            jid = value["requested_spec"]["id"]
            candidates = [ROOT / "project/runs" / name / "jobs" / jid for name in
                          ("daily_protocol_restart_20261114_v1", "daily_ablation_restart_20261114_v1")]
            original = next(p for p in candidates if (p / "history.json").exists())
            history = read(original / "history.json", local_sources)
            selected_epoch(history, value)
            assert sha(RUN / "assets/reference_weights" / f"{label}_r{seed}.pt") == value["checkpoint_sha256"]
            references.append(dict(id=jid, group=label, seed=seed, record=value,
                                   history=history, checkpoint_hash_verified=True, source_files=local_sources,
                                   note="Reused prior training; not a new v2 training run"))
            groups[label][seed] = value
    summaries = []
    for group, values in sorted(groups.items()):
        summaries.append(dict(group=group, seeds=sorted(values), complete_five_seeds=set(values) == SEEDS,
                              **{m: statistics.mean(v["metrics"][m] for v in values.values()) for m in METRICS}))
    pairs = [(a, b) for a in groups for b in ("R0", "TF_R3", "TF_E2L") if a != b and b in groups]
    pairs += [("R3", "R2"), ("R3", "R0+H"), ("R4", "R3"),
              ("R4", "R4-noS"), ("R4", "R4-wrongS"), ("LOSS_TF", "TF_E2L")]
    comparisons = []
    for a, b in dict.fromkeys(pairs):
        if a in groups and b in groups:
            shared = set(groups[a]) & set(groups[b])
            if shared:
                comparisons.append(comparison(groups, a, b, shared))
    snapshot = dict(captured_at=dt.datetime.now().astimezone().isoformat(), run=RUN.name,
                    note="Completed records only define final metrics. Partial comparisons use identical seeds and cannot activate gates. In-progress history is not final evidence. 2023 is used for selection; 2024 was inspected historically and is not a blind test.",
                    counts=dict(new_training_completed=len(records), diagnostic_evaluations_completed=len(diagnostics),
                                reused_reference_training=len(references), new_training_planned_max=len(specs),
                                in_progress=sum(x["status"] == "in_progress_not_final" for x in pending)),
                    source_files=sources, station_ids=cohort["station_ids"], group_summaries=summaries,
                    comparisons=comparisons, completed_training=records, diagnostic_evaluations=diagnostics,
                    reused_references=references, unfinished_training=pending,
                    gates_snapshot=json.loads((OUT / "GATES.json").read_text()))
    write_json(OUT / "CURRENT_RESULTS.json", snapshot)
    seed_rows, farm_rows = [], []
    for item in references + records:
        metrics = item["record"]["metrics"]
        base = dict(group=item["group"], seed=item["seed"], job_id=item["id"],
                    checkpoint_epoch=item["record"]["checkpoint_epoch"])
        seed_rows.append(dict(base, **{m: metrics[m] for m in METRICS}))
        for i, station in enumerate(cohort["station_ids"]):
            farm_rows.append(dict(base, station=station, station_index=i,
                                  **{m: metrics[a][i] for m, a in zip(METRICS, ARRAYS)}))
    for name, rows in (("current_per_seed.csv", seed_rows), ("current_per_farm.csv", farm_rows)):
        with (OUT / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    lines = ["# 当前已获取结果：供 Gemini 阅读的固定快照", "", f"快照时间：{snapshot['captured_at']}。",
             "", f"新训练完成 {len(records)} 项；旧权重诊断完成 {len(diagnostics)} 项；复用旧训练参照 {len(references)} 项；正在训练 {snapshot['counts']['in_progress']} 项。旧轮 2571 项与上述新训练分开计数。",
             "", "本页只把 SUCCESS 已确认的任务计入结果。未完成任务的逐轮曲线只用于进度查看。R0 为 ENC_neutral_E2L；此处 R3 为新增状态条件生成器，勿与旧输入配置 R3 混淆。",
             "", "2023 同时用于选模和比较，E2L 又从七种编码中筛选；十种子只能检验随机性。2024 历史上已被查看，后续复评也不能称完全盲测。",
             "", "## 完成情况与指标", "", "以下是风场等权指标；不同种子数的行不能直接作公平对比。", "",
             "| 组 | 已完成种子 | MAE | RMSE | 高功率 RMSE |", "|---|---|---:|---:|---:|"]
    for x in summaries:
        lines.append(f"| {x['group']} | {','.join(map(str,x['seeds']))}（{'完整' if x['complete_five_seeds'] else '局部'}） | {x['mae']:.8f} | {x['rmse']:.8f} | {x['high_power_rmse']:.8f} |")
    lines += ["", "## 同种子配对比较", "", "负数表示误差降低；局部结果不用于阶段判定。完整比较、每种子变化与改善风场数均在 JSON 中。", "",
              "| 候选 / 参照 | 配对种子数 | ΔMAE | ΔRMSE | Δ高功率 RMSE |", "|---|---:|---:|---:|---:|"]
    selected = {("R0", "TF_R3"), ("R0", "TF_E2L"), ("R0+H", "R0"), ("R1", "R0"),
                ("R2", "R0"), ("R3", "R2"), ("R3", "R0"), ("R3", "R0+H"),
                ("TF_E2L", "TF_R3"), ("R4", "R0"), ("R4", "R3"), ("R4", "TF_E2L"),
                ("R4", "R4-noS"), ("R4", "R4-wrongS"), ("LOSS_TF", "TF_E2L")}
    for x in comparisons:
        if (x['candidate'], x['reference']) in selected:
            values = " | ".join(f"{x[m+'_change_percent']:+.4f}%" for m in METRICS)
            lines.append(f"| {x['candidate']} / {x['reference']} | {len(x['seeds'])}{'（局部）' if not x['complete_five_seeds'] else ''} | {values} |")
    lines += ["", "## 证据与训练过程入口", "",
              "- [完整结果和训练记录 JSON](CURRENT_RESULTS.json)：每项含实际配置、全部指标数组与诊断、逐轮训练/验证记录、选中 epoch、权重与来源哈希。旧权重诊断没有新训练轮次。",
              "- [逐种子 CSV](current_per_seed.csv)、[逐风场 CSV](current_per_farm.csv)：只含已完成任务与复用参照；不把诊断重评重复计为训练。",
              "- [正反馈与负反馈分析](LEARNINGS_20261116_CN.md)：固定时间的解释与边界，新增事实以本页时间为准。",
              "- [训练与选轮](../code/train.py)、[模型与损失](../code/model.py)、[状态表示及流匹配](../code/representation.py)、[完整五种子门槛](../operations/collect.py)。",
              "- [运行清单](../manifest.json)、[冻结协议](../PROTOCOL.json)、[TF 损失提前预计算说明](../LOSS_TF_PRECOMPUTE_20261116.json)。",
              "", "## 未完成的实验", "", "`loss_enabled=false` 表示完整损失复核尚未进入候选判定；5 项 LOSS_TF 可通过独立标记先训练，不能据此认为其未启动。", "",
              "| 任务 | 状态 | 已记录进度（非最终成绩） |", "|---|---|---|"]
    for x in pending:
        p = x.get("progress", {})
        info = f"epoch {p['epoch']}, step {p['step']}/{p['total_steps']}" if p else "等待依赖或调度"
        lines.append(f"| {x['id']} | {x['status']} | {info} |")
    lines += ["", "导出时核验了全部冻结文件、完成标志、请求参数、权重哈希和最小验证 MAE 选轮。原始天气/功率、权重及全量预测仍在服务器；本快照用于审阅，不等于这些数据实体已公开。"]
    (OUT / "CURRENT_RESULTS_CN.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(captured_at=snapshot['captured_at'], **snapshot['counts'],
                         final_seed_rows=len(seed_rows), final_farm_rows=len(farm_rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
