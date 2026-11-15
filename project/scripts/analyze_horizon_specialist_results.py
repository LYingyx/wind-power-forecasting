from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from scripts.analyze_incremental_mechanism_v2 import (
    compare_groups,
    group_predictions,
    summarize_group,
)


HORIZON_BANDS = (
    ("0.25-1h", slice(0, 4)),
    ("1.25-3h", slice(4, 12)),
    ("3.25-6h", slice(12, 24)),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    return seeds


def paired_p(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if len(values) < 2:
        return 1.0
    standard_error = values.std(ddof=1) / np.sqrt(len(values))
    if standard_error <= 1e-12:
        return 0.0 if abs(values.mean()) > 1e-12 else 1.0
    statistic = values.mean() / standard_error
    return float(2.0 * student_t.sf(abs(statistic), len(values) - 1))


def holm(rows: list[dict[str, Any]], family: str) -> None:
    selected = [index for index, row in enumerate(rows) if row["family"] == family]
    if not selected:
        return
    p_values = np.asarray([rows[index]["seedwise_p_two_sided"] for index in selected])
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, local_index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[local_index]))
        adjusted[local_index] = running
    for local_index, row_index in enumerate(selected):
        rows[row_index]["holm_p_within_family"] = float(adjusted[local_index])


def aligned_single_model_deltas(
    candidate: Path, reference: Path, seeds: list[int], split: str
) -> np.ndarray:
    """Return paired per-seed absolute-error deltas [seed, origin, lead]."""
    left_seeds, _ = group_predictions(candidate, seeds, split)
    right_seeds, _ = group_predictions(reference, seeds, split)
    deltas = []
    for left, right in zip(left_seeds, right_seeds, strict=True):
        np.testing.assert_array_equal(left["origin_unix"], right["origin_unix"])
        np.testing.assert_allclose(left["cluster_target_mw"], right["cluster_target_mw"])
        left_error = np.abs(
            left["cluster_prediction_mw"] - left["cluster_target_mw"]
        )
        right_error = np.abs(
            right["cluster_prediction_mw"] - right["cluster_target_mw"]
        )
        deltas.append(left_error.astype(np.float64) - right_error.astype(np.float64))
    return np.stack(deltas, axis=0)


def mean_t_interval(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and two-sided 95% t interval across independent training seeds."""
    values = np.asarray(values, dtype=np.float64)
    point = float(values.mean())
    if values.size < 2:
        return point, point, point
    half_width = float(
        student_t.ppf(0.975, df=values.size - 1)
        * values.std(ddof=1)
        / np.sqrt(values.size)
    )
    return point, point - half_width, point + half_width


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal horizon-specialist statistics")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--root", default="artifacts/horizon_specialist_v1")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    root = (project / args.root).resolve()
    seeds = parse_seeds(args.seeds)
    output = root / f"analysis_{args.split}"
    figures = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reference_validation_roots = {
        "Normal": project
        / "artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h",
        "KD": project / "artifacts/ensemble_stabilization_v1/groups/kd_050",
        "FullHistory": project
        / "artifacts/conditional_forcing_validation_v1/groups/anchor_020",
    }
    references = dict(reference_validation_roots)
    if args.split == "test":
        # Frozen test predictions were exported once into the paper-evidence
        # tree.  Keep validation-based checkpoint selection tied to the
        # original training runs; only test evaluation reads these exports.
        export_root = project / "artifacts/paper_evidence_test_v1/cluster_exports"
        references = {
            "Normal": export_root / "Normal",
            "KD": export_root / "KD",
            "FullHistory": export_root / "FullHistory",
        }
    candidates = {
        path.name: path
        for path in sorted((root / "groups").iterdir())
        if path.is_dir()
        and all(
            (path / f"seed{seed}/predictions_{args.split}.npz").is_file()
            for seed in seeds
        )
    }
    if not candidates:
        raise FileNotFoundError(f"No complete candidate groups for split={args.split}")
    summaries = []
    for label, path in {**references, **candidates}.items():
        if all(
            (path / f"seed{seed}/predictions_{args.split}.npz").is_file()
            and (path / f"seed{seed}/metrics_{args.split}.json").is_file()
            for seed in seeds
        ):
            row = summarize_group(path, seeds, args.split)
            row["group"] = label
            summaries.append(row)
    write_csv(output / "summaries.csv", summaries)

    representative_rows = []
    all_evaluation_roots = {**references, **candidates}
    all_validation_roots = {**reference_validation_roots, **candidates}
    for label, path in all_evaluation_roots.items():
        validation_path = all_validation_roots[label]
        validation_values = []
        for seed in seeds:
            metrics_path = validation_path / f"seed{seed}/metrics_val.json"
            if not metrics_path.is_file():
                break
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            validation_values.append((seed, float(metrics["cluster_mae_mw"])))
        if len(validation_values) != len(seeds):
            continue
        median = float(np.median([value for _, value in validation_values]))
        representative_seed, validation_mae = min(
            validation_values, key=lambda item: (abs(item[1] - median), item[0])
        )
        evaluation_path = path / f"seed{representative_seed}/metrics_{args.split}.json"
        evaluation_mae = None
        if evaluation_path.is_file():
            evaluation_mae = float(
                json.loads(evaluation_path.read_text(encoding="utf-8"))[
                    "cluster_mae_mw"
                ]
            )
        representative_rows.append(
            {
                "group": label,
                "selection_split": "val",
                "representative_seed": representative_seed,
                "validation_mae_mw": validation_mae,
                "ten_seed_validation_median_mw": median,
                f"representative_{args.split}_mae_mw": evaluation_mae,
                "selection_rule": "seed closest to 10-seed validation median",
            }
        )
    write_csv(output / "representative_single_models.csv", representative_rows)

    representative_by_group = {row["group"]: row for row in representative_rows}
    representative_comparisons = []
    evaluation_key = f"representative_{args.split}_mae_mw"
    for name in candidates:
        reference_name = "KD" if name.startswith("kd_") else "Normal"
        candidate_row = representative_by_group.get(name)
        reference_row = representative_by_group.get(reference_name)
        if candidate_row is None or reference_row is None:
            continue
        candidate_mae = candidate_row.get(evaluation_key)
        reference_mae = reference_row.get(evaluation_key)
        if candidate_mae is None or reference_mae is None:
            continue
        delta = float(candidate_mae) - float(reference_mae)
        representative_comparisons.append(
            {
                "comparison": f"{name}_vs_{reference_name}",
                "candidate_seed": candidate_row["representative_seed"],
                "reference_seed": reference_row["representative_seed"],
                "candidate_mae_mw": candidate_mae,
                "reference_mae_mw": reference_mae,
                "candidate_minus_reference_mw": delta,
                "within_2mw_noninferiority_margin_descriptive": delta <= 2.0,
                "selection_rule": "each checkpoint selected on validation median proximity before test",
                "formal_inference": False,
            }
        )
    write_csv(
        output / "representative_single_model_comparisons.csv",
        representative_comparisons,
    )

    comparisons: list[dict[str, Any]] = []
    comparison_specs: list[tuple[str, Path, str, Path, str]] = []
    for name, path in candidates.items():
        if name.startswith("kd_"):
            family = "kd_single_model"
            reference_name = "KD"
        elif name.startswith("l"):
            family = "lightweight"
            reference_name = "Normal"
        else:
            family = "dual_specialist"
            reference_name = "Normal"
        comparison_specs.append(
            (name, path, reference_name, references[reference_name], family)
        )
    if "h0_dual_full_taper" in candidates:
        for name in ("h1_dual_mixed_taper", "h2_dual_learned_taper"):
            if name in candidates:
                comparison_specs.append(
                    (
                        name,
                        candidates[name],
                        "h0_dual_full_taper",
                        candidates["h0_dual_full_taper"],
                        "dual_specialist",
                    )
                )
    horizon_rows = []
    lead_rows = []
    for index, (candidate_name, candidate_path, reference_name, reference_path, family) in enumerate(comparison_specs):
        row = compare_groups(
            candidate_path,
            reference_path,
            seeds,
            args.bootstrap_repeats,
            51000 + index,
            args.split,
        )
        row.update(
            comparison=f"{candidate_name}_vs_{reference_name}",
            candidate=candidate_name,
            reference=reference_name,
            family=family,
        )
        row["seedwise_p_two_sided"] = paired_p(row["seedwise_differences_mw"])
        comparisons.append(row)
        seed_delta = aligned_single_model_deltas(
            candidate_path, reference_path, seeds, args.split
        )
        seed_lead_delta = seed_delta.mean(axis=1)
        for lead_index in range(seed_delta.shape[2]):
            point, low, high = mean_t_interval(seed_lead_delta[:, lead_index])
            lead_rows.append(
                {
                    "comparison": row["comparison"],
                    "lead_hours": (lead_index + 1) / 4.0,
                    "single_model_candidate_minus_reference_mw": point,
                    "seedwise_ci95_low_mw": low,
                    "seedwise_ci95_high_mw": high,
                    "independent_training_seeds": len(seeds),
                }
            )
        for band, selection in HORIZON_BANDS:
            seed_band_delta = seed_delta[:, :, selection].mean(axis=(1, 2))
            band_point, band_low, band_high = mean_t_interval(seed_band_delta)
            horizon_rows.append(
                {
                    "comparison": row["comparison"],
                    "horizon_band": band,
                    "single_model_candidate_minus_reference_mw": band_point,
                    "seedwise_ci95_low_mw": band_low,
                    "seedwise_ci95_high_mw": band_high,
                    "independent_training_seeds": len(seeds),
                }
            )
    for family in {row["family"] for row in comparisons}:
        holm(comparisons, family)
    for row in comparisons:
        if row["family"] == "kd_single_model":
            row["formal_success"] = bool(
                row["seedwise_difference_mean_mw"] <= 2.0
                and row["seedwise_ci95_high_mw"] <= 2.0
            )
            row["decision_rule"] = (
                "single-model non-inferiority to current KD with 2 MW margin"
            )
        else:
            row["structural_superiority_confirmed"] = bool(
                row["seedwise_ci95_high_mw"] < 0.0
                and row.get("holm_p_within_family", 1.0) < 0.05
            )
            row["formal_success"] = row["structural_superiority_confirmed"]
            relevant_bands = [
                band
                for band in horizon_rows
                if band["comparison"] == row["comparison"]
            ]
            worst_band = max(
                (
                    float(band["single_model_candidate_minus_reference_mw"])
                    for band in relevant_bands
                ),
                default=float("inf"),
            )
            row["safe_single_model_noninferiority"] = bool(
                row["seedwise_ci95_high_mw"] <= 2.0
                and row["seedwise_candidate_wins"] >= len(seeds) / 2
                and worst_band <= 1.0
            )
            row["proceed_to_kd"] = bool(
                row["structural_superiority_confirmed"]
                or row["safe_single_model_noninferiority"]
            )
            row["decision_rule"] = (
                "10-seed paired single-model superiority; ensemble is diagnostic only"
            )
            row["kd_protocol_amendment"] = (
                "Before 2024 test, user allowed comparable single-model performance; "
                "KD may proceed when 2 MW non-inferiority, >=50% seed wins, and <=1 MW "
                "mean degradation in every horizon band all hold. This does not convert "
                "non-inferiority into a superiority claim."
            )
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(output / "horizon_bands.csv", horizon_rows)
    write_csv(output / "per_lead.csv", lead_rows)

    diagnostic_rows = []
    for name, path in candidates.items():
        for seed in seeds:
            prediction_path = path / f"seed{seed}/predictions_{args.split}.npz"
            with np.load(prediction_path) as values:
                row: dict[str, Any] = {"group": name, "seed": seed}
                if "weather_view_gate" in values:
                    gate = values["weather_view_gate"].astype(np.float64)
                    row.update(
                        gate_short_mean=float(gate[:, :4].mean()),
                        gate_middle_mean=float(gate[:, 4:12].mean()),
                        gate_long_mean=float(gate[:, 12:].mean()),
                    )
                if "baseline_anchor_abs_drift_factor" in values:
                    drift = values["baseline_anchor_abs_drift_factor"].astype(np.float64) * 6827.01
                    row.update(
                        anchor_drift_short_mw=float(drift[:, :4].mean()),
                        anchor_drift_middle_mw=float(drift[:, 4:12].mean()),
                        anchor_drift_long_mw=float(drift[:, 12:].mean()),
                    )
                if "near_specialist_prediction_mw" in values:
                    target = values["cluster_target_mw"].astype(np.float64)
                    near_error = values["near_specialist_prediction_mw"].astype(np.float64) - target
                    full_error = values["full_specialist_prediction_mw"].astype(np.float64) - target
                    row["specialist_signed_error_correlation"] = float(
                        np.corrcoef(near_error.ravel(), full_error.ravel())[0, 1]
                    )
                diagnostic_rows.append(row)
    write_csv(output / "mechanism_diagnostics.csv", diagnostic_rows)
    payload = {
        "protocol": {
            "split": args.split,
            "seeds": seeds,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_unit": "UTC day",
            "primary_unit": "independent single checkpoint",
            "per_lead_and_band_ci_unit": "independent training seed",
            "ensemble_role": "teacher and variance diagnostic only",
            "negative_difference_means_candidate_better": True,
            "test_is_reused_internal_holdout": args.split == "test",
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "horizon_bands": horizon_rows,
        "mechanism_diagnostics": diagnostic_rows,
        "representative_single_models": representative_rows,
        "representative_single_model_comparisons": representative_comparisons,
    }
    (output / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for comparison in {row["comparison"] for row in lead_rows}:
        selected = [row for row in lead_rows if row["comparison"] == comparison]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        x = np.asarray([row["lead_hours"] for row in selected])
        y = np.asarray(
            [row["single_model_candidate_minus_reference_mw"] for row in selected]
        )
        low = np.asarray([row["seedwise_ci95_low_mw"] for row in selected])
        high = np.asarray([row["seedwise_ci95_high_mw"] for row in selected])
        axis.axhline(0.0, color="black", linewidth=1)
        axis.plot(x, y, marker="o", markersize=3)
        axis.fill_between(x, low, high, alpha=0.2)
        axis.set(
            xlabel="Forecast horizon (h)",
            ylabel="Single model: candidate - reference MAE (MW)",
            title=comparison,
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(figures / f"{args.split}_{comparison}_per_lead.png", dpi=180)
        plt.close(figure)
    print(json.dumps({"output": str(output), "groups": list(candidates)}, indent=2))


if __name__ == "__main__":
    main()
