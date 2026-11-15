"""Audit and summarize the exact-CBH temporal proxy-weather experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_original_cbh_20260917_v1")
V2 = Path("/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/temporal_sr_named_models_20260917_v2")
STATIONS = [
    "1043019", "1043010", "1043036", "1043007", "1043028",
    "1043033", "1043011", "1043058", "1043015", "1043061",
]
SEEDS = list(range(42, 52))
METHODS = ["LINEAR", "DET", "ICFM", "VPCFM", "SINE_SI", "VP_EPS", "VP_V", "EDM", "MEANFLOW"]
FAMILIES = [f"CBH_ADAPT_{method}" for method in METHODS]
ORIGINAL = "CBH_ORIGINAL_56_SUBSET"
LABEL = {
    ORIGINAL: "原 CBH（56站模型在所选10站直接评测）",
    "CBH_ADAPT_LINEAR": "CBH + 线性小时天气",
    "CBH_ADAPT_DET": "CBH + DET 代理天气",
    "CBH_ADAPT_ICFM": "CBH + I-CFM 代理天气",
    "CBH_ADAPT_VPCFM": "CBH + VP-CFM 代理天气",
    "CBH_ADAPT_SINE_SI": "CBH + Sine-SI 代理天气",
    "CBH_ADAPT_VP_EPS": "CBH + VP-epsilon 代理天气",
    "CBH_ADAPT_VP_V": "CBH + VP-v 代理天气",
    "CBH_ADAPT_EDM": "CBH + EDM 代理天气",
    "CBH_ADAPT_MEANFLOW": "CBH + MeanFlow 代理天气",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(rows: list[dict]) -> None:
    order = np.argsort([row["wilcoxon_seed_p"] for row in rows])
    running = 0.0
    adjusted = np.empty(len(rows))
    for rank, index in enumerate(order):
        running = max(
            running,
            min(1.0, (len(rows) - rank) * rows[index]["wilcoxon_seed_p"]),
        )
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["holm_p"] = float(value)


def main() -> None:
    protocol_path = V2 / "configs/protocol.json"
    protocol_sha = sha256(protocol_path)
    target = np.load(ROOT / "cache/report_target.npy")
    errors: list[str] = []
    tensors: dict[str, dict[str, np.ndarray]] = {}
    original_checkpoint_hashes: dict[int, str] = {}

    if target.shape != (1632, 10, 24) or not np.isfinite(target).all():
        errors.append(f"report target shape/finite mismatch: {target.shape}")
    inherited_audits = {}
    for name in ["generator_audit", "proxy_condition_audit"]:
        audit_path = V2 / "status" / f"{name}.json"
        audit = json.loads(audit_path.read_text())
        inherited_audits[name] = {
            "passed": bool(audit.get("passed")),
            "sha256": sha256(audit_path),
        }
        if not audit.get("passed"):
            errors.append(f"inherited v2 audit failed: {audit_path}")
    cbh_gradient_path = ROOT / "status" / "CBH_GENERATOR_GRADIENT_AUDIT.json"
    if not cbh_gradient_path.exists():
        errors.append(f"missing {cbh_gradient_path}")
        cbh_gradient_audit = {"passed": False}
    else:
        cbh_gradient_audit = json.loads(cbh_gradient_path.read_text())
        if not cbh_gradient_audit.get("passed"):
            errors.append(f"CBH generator-gradient audit failed: {cbh_gradient_path}")
    for name in ["common.py", "generators.py", "predictors.py"]:
        inherited = ROOT / "vendor" / "generator_snapshot" / name
        source = V2 / "code" / name
        if sha256(inherited) != sha256(source):
            errors.append(f"generator snapshot/source mismatch: {name}")

    original_mae = np.empty((10, 10))
    original_rmse = np.empty(10)
    for seed_index, seed in enumerate(SEEDS):
        directory = ROOT / "original_eval" / f"seed{seed}"
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            errors.append(f"missing {metrics_path}")
            continue
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("model") != ORIGINAL or int(metrics.get("seed", -1)) != seed:
            errors.append(f"identity mismatch: {metrics_path}")
        if metrics.get("station_ids") != STATIONS:
            errors.append(f"station order mismatch: {metrics_path}")
        prediction = np.load(directory / "predictions_fp32.npy")
        baseline = np.load(directory / "baseline_fp32.npy")
        if prediction.shape != target.shape or baseline.shape != target.shape:
            errors.append(f"original array shape mismatch: {directory}")
            continue
        if not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
            errors.append(f"original non-finite array: {directory}")
        station_mae = np.abs(prediction - target).mean(axis=(0, 2))
        reported = np.asarray(metrics["fp32"]["per_station_mae"])
        if not np.allclose(station_mae, reported, atol=2e-7, rtol=0):
            errors.append(f"original station MAE mismatch: {metrics_path}")
        total_mae = float(np.abs(prediction - target).mean())
        total_rmse = float(np.sqrt(np.square(prediction - target).mean()))
        if abs(total_mae - metrics["fp32"]["mae"]) > 2e-7:
            errors.append(f"original MAE mismatch: {metrics_path}")
        if abs(total_rmse - metrics["fp32"]["rmse"]) > 2e-7:
            errors.append(f"original RMSE mismatch: {metrics_path}")
        original_mae[seed_index] = station_mae
        original_rmse[seed_index] = total_rmse
        original_checkpoint_hashes[seed] = metrics["checkpoint_sha256"]
        for split in ["train", "val"]:
            cache = ROOT / "cache" / "baseline" / f"{split}_seed{seed}.npy"
            entry = metrics["baseline_cache"][split]
            if sha256(cache) != entry["sha256"]:
                errors.append(f"baseline cache hash mismatch: {cache}")
            values = np.load(cache, mmap_mode="r")
            if tuple(values.shape) != tuple(entry["shape"]):
                errors.append(f"baseline cache shape mismatch: {cache}")
    tensors[ORIGINAL] = {
        "mae": original_mae,
        "rmse": np.repeat(original_rmse[:, None], 10, axis=1),
        "stop": np.full((10, 10), np.nan),
    }

    proxy_records: dict[str, list[dict]] = {family: [] for family in FAMILIES if not family.endswith("LINEAR")}
    max_initial_equivalence = 0.0
    verified_files = 0
    for family in FAMILIES:
        method = family.removeprefix("CBH_ADAPT_")
        mae = np.empty((10, 10))
        rmse = np.empty_like(mae)
        stop = np.empty_like(mae)
        expected_samples = 1 if method in {"LINEAR", "DET"} else 5
        for station_index, station in enumerate(STATIONS):
            for seed_index, seed in enumerate(SEEDS):
                directory = ROOT / "results" / family / station / f"seed{seed}"
                required = [
                    "metrics.json", "best_model.pt", "history.json",
                    "predictions_report.npy", "samples_report.npy", "proxy_metrics.json",
                ]
                missing = [name for name in required if not (directory / name).exists()]
                if missing:
                    errors.append(f"missing {directory}: {missing}")
                    continue
                metrics = json.loads((directory / "metrics.json").read_text())
                if metrics.get("model") != family or str(metrics.get("station_id")) != station or int(metrics.get("seed", -1)) != seed:
                    errors.append(f"identity mismatch: {directory}")
                if metrics.get("protocol_sha256") != protocol_sha:
                    errors.append(f"protocol hash mismatch: {directory}")
                if metrics.get("source_cbh_checkpoint_sha256") != original_checkpoint_hashes.get(seed):
                    errors.append(f"source CBH hash mismatch: {directory}")
                cache = ROOT / "cache" / "baseline" / f"val_seed{seed}.npy"
                if metrics.get("baseline_cache_sha256") != sha256(cache):
                    errors.append(f"baseline hash mismatch: {directory}")
                checkpoint = directory / "best_model.pt"
                if metrics.get("checkpoint_sha256") != sha256(checkpoint):
                    errors.append(f"checkpoint hash mismatch: {directory}")
                pretrain_hash = metrics.get("generator_pretrain_sha256")
                if method == "LINEAR":
                    if pretrain_hash is not None:
                        errors.append(f"LINEAR unexpectedly has generator hash: {directory}")
                    equivalence = float(metrics["initial_original_equivalence_max_error"])
                    max_initial_equivalence = max(max_initial_equivalence, equivalence)
                    if equivalence > 4e-5:
                        errors.append(f"initial CBH equivalence > 4e-5: {directory}: {equivalence}")
                else:
                    pretrain = V2 / "results" / f"TiDE_SR_{method}" / station / f"seed{seed}" / "pretrained_generator.pt"
                    if pretrain_hash != sha256(pretrain):
                        errors.append(f"generator pretrain hash mismatch: {directory}")
                    proxy = json.loads((directory / "proxy_metrics.json").read_text())
                    if not isinstance(proxy, dict) or set(proxy) != {"before_joint", "after_joint"}:
                        errors.append(f"proxy metrics malformed: {directory}")
                    else:
                        proxy_records[family].append(proxy)
                prediction = np.load(directory / "predictions_report.npy")
                samples = np.load(directory / "samples_report.npy")
                if prediction.shape != (1632, 24) or samples.shape != (1632, expected_samples, 24):
                    errors.append(f"prediction/sample shape mismatch: {directory}")
                elif not np.isfinite(prediction).all() or not np.isfinite(samples).all():
                    errors.append(f"prediction/sample non-finite: {directory}")
                else:
                    median = np.median(samples, axis=1)
                    if not np.allclose(median, prediction, atol=2e-6, rtol=0):
                        errors.append(f"sample median mismatch: {directory}")
                    actual_mae = float(np.abs(prediction - target[:, station_index]).mean())
                    actual_rmse = float(np.sqrt(np.square(prediction - target[:, station_index]).mean()))
                    if abs(actual_mae - float(metrics["report_mae"])) > 2e-7:
                        errors.append(f"adapted MAE mismatch: {directory}")
                    if abs(actual_rmse - float(metrics["report_rmse"])) > 2e-7:
                        errors.append(f"adapted RMSE mismatch: {directory}")
                history = json.loads((directory / "history.json").read_text())
                if len(history) != int(metrics["trained_epochs"]):
                    errors.append(f"history length mismatch: {directory}")
                for key in ["report_mae", "report_rmse", "report_crps", "sample_std", "val_stop_mae"]:
                    if not math.isfinite(float(metrics[key])):
                        errors.append(f"non-finite {key}: {directory}")
                if int(metrics["report_n"]) != 1632 or int(metrics["samples_per_prediction"]) != expected_samples:
                    errors.append(f"report protocol mismatch: {directory}")
                mae[seed_index, station_index] = metrics["report_mae"]
                rmse[seed_index, station_index] = metrics["report_rmse"]
                stop[seed_index, station_index] = metrics["val_stop_mae"]
                verified_files += len(required)
        tensors[family] = {"mae": mae, "rmse": rmse, "stop": stop}

    if errors:
        dump_json(ROOT / "status" / "FINAL_AUDIT.json", {
            "passed": False,
            "errors": errors,
            "verified_required_files_before_failure": verified_files,
        })
        raise RuntimeError("\n".join(errors[:50]))

    overall_rows: list[dict] = []
    station_rows: list[dict] = []
    seed_rows: list[dict] = []
    for family in [ORIGINAL, *FAMILIES]:
        values = tensors[family]["mae"]
        seed_mean = values.mean(axis=1)
        station_mean = values.mean(axis=0)
        for seed_index, seed in enumerate(SEEDS):
            seed_rows.append({
                "model": family, "label": LABEL[family], "seed": seed,
                "station_equal_mae": float(seed_mean[seed_index]),
            })
        for station_index, station in enumerate(STATIONS):
            station_values = values[:, station_index]
            low = int(station_values.argmin())
            high = int(station_values.argmax())
            station_rows.append({
                "model": family, "label": LABEL[family], "station_id": station,
                "mean": float(station_values.mean()), "std": float(station_values.std(ddof=1)),
                "min": float(station_values[low]), "min_seed": SEEDS[low],
                "max": float(station_values[high]), "max_seed": SEEDS[high],
            })
        low = int(seed_mean.argmin())
        high = int(seed_mean.argmax())
        overall_rows.append({
            "model": family, "label": LABEL[family],
            "mean": float(seed_mean.mean()), "std": float(seed_mean.std(ddof=1)),
            "min": float(seed_mean[low]), "min_seed": SEEDS[low],
            "max": float(seed_mean[high]), "max_seed": SEEDS[high],
            "mean_rmse": float(tensors[family]["rmse"].mean()),
            "mean_val_stop": None if family == ORIGINAL else float(tensors[family]["stop"].mean()),
            "min_station": STATIONS[int(station_mean.argmin())],
            "min_station_mean": float(station_mean.min()),
            "max_station": STATIONS[int(station_mean.argmax())],
            "max_station_mean": float(station_mean.max()),
            "stations": 10, "seeds": 10,
        })

    comparisons: list[tuple[str, str, str]] = []
    for family in FAMILIES[1:]:
        comparisons.append((family, "CBH_ADAPT_LINEAR", "proxy_method_vs_linear_adapter"))
    for family in FAMILIES:
        comparisons.append((family, ORIGINAL, "adapted_route_vs_original_cbh"))
    rng = np.random.default_rng(20260917)
    bootstrap_count = 20_000
    paired_rows: list[dict] = []
    for model, control, comparison_set in comparisons:
        difference = tensors[model]["mae"] - tensors[control]["mae"]
        seed_difference = difference.mean(axis=1)
        station_difference = difference.mean(axis=0)
        seed_draws = rng.integers(0, 10, (bootstrap_count, 10))
        station_draws = rng.integers(0, 10, (bootstrap_count, 10))
        bootstrap = np.empty(bootstrap_count)
        for index in range(bootstrap_count):
            bootstrap[index] = difference[np.ix_(seed_draws[index], station_draws[index])].mean()
        try:
            p_value = float(wilcoxon(seed_difference, alternative="two-sided", method="exact").pvalue)
        except ValueError:
            p_value = 1.0
        paired_rows.append({
            "comparison_set": comparison_set,
            "model": model, "model_label": LABEL[model],
            "control": control, "control_label": LABEL[control],
            "delta": float(difference.mean()),
            "relative_percent": float(100 * difference.mean() / tensors[control]["mae"].mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "seed_wins": int((seed_difference < 0).sum()),
            "station_wins": int((station_difference < 0).sum()),
            "wilcoxon_seed_p": p_value,
        })
    holm_adjust(paired_rows)

    proxy_rows: list[dict] = []
    for family, records in proxy_records.items():
        row: dict = {"model": family, "label": LABEL[family], "station_seed_records": len(records)}
        for task in ["4h_to_1h", "2h_to_1h"]:
            prefix = "4h" if task.startswith("4h") else "2h"
            row[f"{prefix}_linear_mae"] = float(np.mean([
                record["after_joint"][task]["linear_mae"] for record in records
            ]))
            for stage in ["before_joint", "after_joint"]:
                row[f"{prefix}_{stage}_generated_mae"] = float(np.mean([
                    record[stage][task]["generated_mae"] for record in records
                ]))
                row[f"{prefix}_{stage}_sample_std"] = float(np.mean([
                    record[stage][task]["sample_std"] for record in records
                ]))
        proxy_rows.append(row)

    analysis = ROOT / "analysis"
    dump_json(analysis / "overall_seed_summary.json", overall_rows)
    dump_json(analysis / "per_station_seed_summary.json", station_rows)
    dump_json(analysis / "paired_comparisons.json", paired_rows)
    dump_json(analysis / "proxy_summary.json", proxy_rows)
    write_csv(analysis / "overall_seed_summary.csv", overall_rows)
    write_csv(analysis / "per_station_seed_summary.csv", station_rows)
    write_csv(analysis / "seed_values.csv", seed_rows)
    write_csv(analysis / "paired_comparisons.csv", paired_rows)
    write_csv(analysis / "proxy_summary.csv", proxy_rows)

    v2_overall_path = V2 / "analysis" / "overall_seed_summary.json"
    v2_rows = json.loads(v2_overall_path.read_text())
    retained = [
        row for row in v2_rows
        if row["model"] in {
            "DLinear", "PatchTST", "HistGradientBoostingRegressor", "TiDE", "TFT", "NBEATSx",
            "TiDE_SR_DET", "TiDE_SR_ICFM", "TiDE_SR_VPCFM", "TiDE_SR_SINE_SI",
            "TiDE_SR_VP_EPS", "TiDE_SR_VP_V", "TiDE_SR_EDM", "TiDE_SR_MEANFLOW",
        }
    ]
    for row in retained:
        row["source_summary_sha256"] = sha256(v2_overall_path)
        row["comparability_note"] = (
            "same stations/seeds/report windows; predictor training differs: v2 TiDE routes use the 10% pilot, "
            "whereas original CBH was trained on full 2022 and initialized from checkpoints selected on 2023"
        )
    write_csv(analysis / "v2_named_baseline_context.csv", retained)

    snapshot = ROOT / "vendor" / "final_code_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "code").glob("*.py")):
        shutil.copy2(source, snapshot / source.name)
    source_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for directory in [ROOT / "code", snapshot, ROOT / "vendor" / "generator_snapshot"]
        for path in sorted(directory.glob("*.py"))
    }
    dump_json(ROOT / "references" / "final_code_hashes.json", source_hashes)

    audit = {
        "passed": True,
        "original_family": ORIGINAL,
        "adapted_families": FAMILIES,
        "stations": STATIONS,
        "seeds": SEEDS,
        "original_seed_evaluations": 10,
        "adapted_station_seed_results": len(FAMILIES) * len(STATIONS) * len(SEEDS),
        "expected_adapted_station_seed_results": 900,
        "verified_required_result_files": verified_files,
        "max_initial_original_equivalence_abs_error": max_initial_equivalence,
        "initial_equivalence_tolerance": 4e-5,
        "sample_median_predictions_verified": True,
        "metric_recomputation_verified": True,
        "checkpoint_and_provenance_hashes_verified": True,
        "inherited_generator_audits": inherited_audits,
        "cbh_generator_gradient_audit": {
            "passed": bool(cbh_gradient_audit.get("passed")),
            "sha256": sha256(cbh_gradient_path),
        },
        "generator_snapshot_matches_audited_v2_source": True,
        "protocol_sha256": protocol_sha,
        "report_target_sha256": sha256(ROOT / "cache" / "report_target.npy"),
        "v2_context_summary_sha256": sha256(v2_overall_path),
        "crossed_bootstrap_repetitions": bootstrap_count,
        "holm_comparison_count": len(paired_rows),
    }
    dump_json(ROOT / "status" / "FINAL_AUDIT.json", audit)
    dump_json(ROOT / "status" / "COMPLETENESS.json", {"all_complete": True, "missing": []})
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
