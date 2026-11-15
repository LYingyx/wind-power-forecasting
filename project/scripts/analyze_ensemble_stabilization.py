from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr, t as student_t

try:
    from scripts.analyze_conditional_forcing_validation import block_bootstrap_matrix
except ModuleNotFoundError:
    from analyze_conditional_forcing_validation import block_bootstrap_matrix  # type: ignore[no-redef]


SEEDS = list(range(42, 52))
LEGACY_SEEDS = [42, 43, 44]
HORIZON_BANDS = (
    ("0.25-1h", slice(0, 4)),
    ("1.25-3h", slice(4, 12)),
    ("3.25-6h", slice(12, 24)),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def t_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    return mean, std, mean - half, mean + half


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def load_group(
    root: Path,
    seeds: list[int],
    metrics_name: str = "metrics_val.json",
    predictions_name: str = "predictions_val.npz",
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    metrics = []
    predictions = []
    origins = None
    targets = None
    for seed in seeds:
        run = root / f"seed{seed}"
        metric_path = run / metrics_name
        prediction_path = run / predictions_name
        if not metric_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Incomplete group run: {run} ({metrics_name})")
        metrics.append(load_json(metric_path))
        prediction = load_prediction(prediction_path)
        current_origins = prediction["origin_unix"].astype(np.int64)
        current_targets = prediction["cluster_target_mw"].astype(np.float64)
        if origins is None:
            origins = current_origins
            targets = current_targets
        else:
            np.testing.assert_array_equal(origins, current_origins)
            np.testing.assert_allclose(targets, current_targets, atol=1e-5, rtol=0.0)
        predictions.append(prediction["cluster_prediction_mw"].astype(np.float64))
    assert origins is not None and targets is not None
    return metrics, np.stack(predictions), targets, origins


def summarize(label: str, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics, predictions, target, origins = load_group(
        spec["root"], spec["seeds"], spec.get("metrics", "metrics_val.json"),
        spec.get("predictions", "predictions_val.npz")
    )
    seed_mae = np.asarray([float(value["cluster_mae_mw"]) for value in metrics])
    mean_prediction = predictions.mean(axis=0)
    median_prediction = np.median(predictions, axis=0)
    mean_ensemble_mae = float(np.abs(mean_prediction - target).mean())
    median_ensemble_mae = float(np.abs(median_prediction - target).mean())
    best_index = int(np.argmin(seed_mae))
    row = {
        "group": label,
        "role": spec["role"],
        "seeds": len(spec["seeds"]),
        "seed_values": spec["seeds"],
        "cluster_mae_mw_mean": float(seed_mae.mean()),
        "cluster_mae_mw_sample_std": float(seed_mae.std(ddof=1)),
        "best_seed": int(spec["seeds"][best_index]),
        "best_seed_mae_mw": float(seed_mae[best_index]),
        "mean_ensemble_mae_mw": mean_ensemble_mae,
        "median_ensemble_mae_mw": median_ensemble_mae,
        "single_mean_minus_mean_ensemble_mw": float(seed_mae.mean() - mean_ensemble_mae),
        "station_mae_mw_mean": float(
            np.mean([value["station_mae_mw"] for value in metrics])
        ),
        "parameter_count": int(metrics[0]["parameter_count"]),
        "primary_10_seed_protocol": len(spec["seeds"]) == 10,
    }
    raw = {
        "metrics": metrics,
        "predictions": predictions,
        "target": target,
        "origins": origins,
        "seed_mae": seed_mae,
        "mean_prediction": mean_prediction,
    }
    return row, raw


def comparison(
    candidate: str,
    reference: str,
    raw: dict[str, dict[str, Any]],
    repeats: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cand = raw[candidate]
    ref = raw[reference]
    np.testing.assert_array_equal(cand["origins"], ref["origins"])
    np.testing.assert_allclose(cand["target"], ref["target"], atol=1e-5, rtol=0.0)
    if cand["predictions"].shape[0] != ref["predictions"].shape[0]:
        raise ValueError("Seedwise comparisons require the same seed count")
    seed_difference = cand["seed_mae"] - ref["seed_mae"]
    seed_stats = t_interval(seed_difference)
    candidate_error = np.abs(cand["mean_prediction"] - cand["target"])
    reference_error = np.abs(ref["mean_prediction"] - ref["target"])
    delta = candidate_error - reference_error
    point, low, high, days = block_bootstrap_matrix(
        delta.mean(axis=1, keepdims=True), cand["origins"], repeats, bootstrap_seed
    )
    statistic = seed_stats[0] / max(seed_stats[1] / np.sqrt(len(seed_difference)), 1e-12)
    row = {
        "comparison": f"{candidate}_vs_{reference}",
        "candidate": candidate,
        "reference": reference,
        "seedwise_candidate_minus_reference_mw": seed_stats[0],
        "seedwise_sample_std_mw": seed_stats[1],
        "seedwise_ci95_low_mw": seed_stats[2],
        "seedwise_ci95_high_mw": seed_stats[3],
        "candidate_wins": int(np.sum(seed_difference < 0.0)),
        "seedwise_p_two_sided": float(
            2.0 * student_t.sf(abs(statistic), len(seed_difference) - 1)
        ),
        "ensemble_candidate_minus_reference_mw": float(point[0]),
        "ensemble_ci95_low_mw": float(low[0]),
        "ensemble_ci95_high_mw": float(high[0]),
        "utc_day_blocks": days,
        "bootstrap_repeats": repeats,
    }
    bands = []
    for band_index, (band, lead_slice) in enumerate(HORIZON_BANDS):
        band_point, band_low, band_high, _ = block_bootstrap_matrix(
            delta[:, lead_slice].mean(axis=1, keepdims=True),
            cand["origins"], repeats, bootstrap_seed + 100 + band_index,
        )
        bands.append(
            {
                "comparison": row["comparison"],
                "horizon_band": band,
                "candidate_minus_reference_mw": float(band_point[0]),
                "ci95_low_mw": float(band_low[0]),
                "ci95_high_mw": float(band_high[0]),
            }
        )
    return row, bands


def apply_holm(rows: list[dict[str, Any]], indices: list[int]) -> None:
    p_values = np.asarray([rows[index]["seedwise_p_two_sided"] for index in indices])
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, local_index in enumerate(order):
        current = min(1.0, (len(indices) - rank) * p_values[local_index])
        running = max(running, current)
        adjusted[local_index] = running
    for local_index, row_index in enumerate(indices):
        rows[row_index]["holm_p_stabilization_family"] = float(adjusted[local_index])


def ensemble_diagnostics(
    label: str, values: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prediction = values["predictions"]
    target = values["target"]
    mean_prediction = prediction.mean(axis=0)
    disagreement = prediction.std(axis=0, ddof=1)
    mean_single_abs = np.abs(prediction - target[None]).mean(axis=0)
    ensemble_abs = np.abs(mean_prediction - target)
    gain = mean_single_abs - ensemble_abs
    single_sq = np.square(prediction - target[None]).mean(axis=0)
    center_sq = np.square(mean_prediction - target)
    ambiguity = np.square(prediction - mean_prediction[None]).mean(axis=0)
    identity_error = float(np.max(np.abs(single_sq - center_sq - ambiguity)))
    signed_errors = (prediction - target[None]).reshape(prediction.shape[0], -1)
    correlations = np.corrcoef(signed_errors)
    off_diagonal = correlations[np.triu_indices(len(prediction), k=1)]
    band_rows = []
    decile_rows = []
    for band, lead_slice in HORIZON_BANDS:
        d = disagreement[:, lead_slice].reshape(-1)
        g = gain[:, lead_slice].reshape(-1)
        band_rows.append(
            {
                "group": label,
                "horizon_band": band,
                "mean_disagreement_mw": float(d.mean()),
                "mean_ensemble_gain_mw": float(g.mean()),
                "pearson_disagreement_gain": float(pearsonr(d, g).statistic),
                "spearman_disagreement_gain": float(spearmanr(d, g).statistic),
                "mean_single_mae_mw": float(mean_single_abs[:, lead_slice].mean()),
                "ensemble_mae_mw": float(ensemble_abs[:, lead_slice].mean()),
                "mean_single_mse_mw2": float(single_sq[:, lead_slice].mean()),
                "ensemble_center_mse_mw2": float(center_sq[:, lead_slice].mean()),
                "ambiguity_mw2": float(ambiguity[:, lead_slice].mean()),
            }
        )
        edges = np.quantile(d, np.linspace(0.0, 1.0, 11))
        bins = np.searchsorted(edges[1:-1], d, side="right")
        band_ensemble_error = ensemble_abs[:, lead_slice].reshape(-1)
        for decile in range(10):
            mask = bins == decile
            decile_rows.append(
                {
                    "group": label,
                    "horizon_band": band,
                    "disagreement_decile": decile + 1,
                    "samples": int(mask.sum()),
                    "mean_disagreement_mw": float(d[mask].mean()),
                    "mean_ensemble_gain_mw": float(g[mask].mean()),
                    "mean_ensemble_error_mw": float(band_ensemble_error[mask].mean()),
                }
            )
    payload = {
        "mean_pairwise_signed_error_correlation": float(off_diagonal.mean()),
        "min_pairwise_signed_error_correlation": float(off_diagonal.min()),
        "max_pairwise_signed_error_correlation": float(off_diagonal.max()),
        "overall_disagreement_mw": float(disagreement.mean()),
        "overall_ensemble_gain_mw": float(gain.mean()),
        "mse_ambiguity_identity_max_abs_error_mw2": identity_error,
        "pointwise_mae_gain_min_mw": float(gain.min()),
    }
    return band_rows, decile_rows, payload


def save_diagnostic_plots(
    output: Path,
    raw: dict[str, dict[str, Any]],
    labels: list[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recent = raw["recent_normal"]
    prediction = recent["predictions"]
    target = recent["target"]
    disagreement = prediction.std(axis=0, ddof=1)
    gain = (
        np.abs(prediction - target[None]).mean(axis=0)
        - np.abs(prediction.mean(axis=0) - target)
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    for axis, (band, lead_slice) in zip(axes, HORIZON_BANDS, strict=True):
        plot = axis.hexbin(
            disagreement[:, lead_slice].reshape(-1),
            gain[:, lead_slice].reshape(-1),
            gridsize=60,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        axis.set_title(band)
        axis.set_xlabel("Seed disagreement (MW)")
        axis.set_ylabel("Pointwise ensemble gain (MW)")
        figure.colorbar(plot, ax=axis, label="log10 count")
    figure.savefig(output / "disagreement_vs_ensemble_gain.png", dpi=180)
    plt.close(figure)

    horizon = np.arange(1, 25) / 4.0
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for label in labels:
        values = raw[label]
        member = values["predictions"]
        truth = values["target"]
        axes[0].plot(
            horizon, member.std(axis=0, ddof=1).mean(axis=0), label=label
        )
        member_gain = (
            np.abs(member - truth[None]).mean(axis=0)
            - np.abs(member.mean(axis=0) - truth)
        )
        axes[1].plot(horizon, member_gain.mean(axis=0), label=label)
    axes[0].set_title("Disagreement by horizon")
    axes[0].set_ylabel("Mean seed std (MW)")
    axes[1].set_title("Ensemble gain by horizon")
    axes[1].set_ylabel("Mean MAE gain (MW)")
    for axis in axes:
        axis.set_xlabel("Lead time (h)")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.savefig(output / "ensemble_diagnostics_by_horizon.png", dpi=180)
    plt.close(figure)

    heatmap_labels = labels[:3]
    figure, axes = plt.subplots(
        1, len(heatmap_labels), figsize=(4.2 * len(heatmap_labels), 3.8),
        constrained_layout=True,
    )
    if len(heatmap_labels) == 1:
        axes = [axes]
    image = None
    for axis, label in zip(axes, heatmap_labels, strict=True):
        values = raw[label]
        signed = (
            values["predictions"] - values["target"][None]
        ).reshape(len(values["predictions"]), -1)
        image = axis.imshow(np.corrcoef(signed), vmin=0.8, vmax=1.0, cmap="magma")
        axis.set_title(label)
        axis.set_xlabel("Seed index")
        axis.set_ylabel("Seed index")
    assert image is not None
    figure.colorbar(image, ax=axes, label="Signed-error correlation")
    figure.savefig(output / "seed_error_correlation.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ensemble stabilization experiments")
    parser.add_argument("--experiment-root", default="artifacts/ensemble_stabilization_v1")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    root = (project / args.experiment_root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    conditional = project / "artifacts/conditional_forcing_validation_v1/groups"
    specs: dict[str, dict[str, Any]] = {
        "power_only": {
            "root": conditional / "history_baseline", "seeds": SEEDS,
            "role": "descriptive power-only reference",
        },
        "weather_only": {
            "root": conditional / "factor_weather_only", "seeds": SEEDS,
            "role": "descriptive weather-only factor reference",
        },
        "direct_joint_d64": {
            "root": project / "artifacts/incremental_mechanism_v2/groups/direct_d64",
            "seeds": SEEDS, "role": "primary strongest matched direct baseline",
        },
        "legacy_fusion_prediction_weight": {
            "root": project / "artifacts/paper_experiments_pad5_v1/baselines_fusion/ms0_prediction_weight",
            "seeds": LEGACY_SEEDS, "role": "descriptive only; three seeds",
        },
        "full_history_anchor_020": {
            "root": conditional / "anchor_020", "seeds": SEEDS,
            "role": "previous full-history correction reference",
        },
        "recent_normal": {
            "root": root / "groups/weak_anchor_recent_1h", "seeds": SEEDS,
            "metrics": "metrics_val_normal.json", "predictions": "predictions_val_normal.npz",
            "role": "frozen scientific main model",
        },
        "recent_ema": {
            "root": root / "groups/weak_anchor_recent_1h", "seeds": SEEDS,
            "metrics": "metrics_val_ema.json", "predictions": "predictions_val_ema.npz",
            "role": "same-run EMA at Normal-selected epoch",
        },
        "recent_checkpoint_avg": {
            "root": root / "groups/weak_anchor_recent_1h", "seeds": SEEDS,
            "metrics": "metrics_val_checkpoint_avg.json",
            "predictions": "predictions_val_checkpoint_avg.npz",
            "role": "last-five checkpoints ending at Normal-selected epoch",
        },
    }
    optional = {
        "kd_025": "plain ensemble KD, alpha=0.25",
        "kd_050": "plain ensemble KD, alpha=0.50",
        "kd_aware_025": "disagreement-aware KD, alpha=0.25",
        "kd_aware_050": "disagreement-aware KD, alpha=0.50",
    }
    for group, role in optional.items():
        group_root = root / "groups" / group
        if all((group_root / f"seed{seed}" / "COMPLETE").is_file() for seed in SEEDS):
            specs[group] = {"root": group_root, "seeds": SEEDS, "role": role}

    required = ("recent_normal", "recent_ema", "recent_checkpoint_avg")
    for label in required:
        spec = specs[label]
        for seed in SEEDS:
            if not (spec["root"] / f"seed{seed}" / spec["metrics"]).is_file():
                raise FileNotFoundError(f"Formal main-stage analysis is incomplete: {label}/seed{seed}")

    summaries = []
    raw: dict[str, dict[str, Any]] = {}
    for label, spec in specs.items():
        summary, values = summarize(label, spec)
        summaries.append(summary)
        raw[label] = values

    pairs = [
        ("recent_normal", "direct_joint_d64"),
        ("recent_normal", "full_history_anchor_020"),
        ("recent_ema", "recent_normal"),
        ("recent_checkpoint_avg", "recent_normal"),
    ]
    for group in optional:
        if group in specs:
            pairs.append((group, "recent_normal"))
            pairs.append((group, "recent_ema"))
    if "kd_025" in specs and "kd_050" in specs:
        pairs.append(("kd_050", "kd_025"))
    if "kd_aware_025" in specs and "kd_025" in specs:
        pairs.append(("kd_aware_025", "kd_025"))
    if "kd_aware_050" in specs and "kd_050" in specs:
        pairs.append(("kd_aware_050", "kd_050"))
    comparisons = []
    horizon_rows = []
    for index, (candidate, reference) in enumerate(pairs):
        row, bands = comparison(
            candidate, reference, raw, args.bootstrap_repeats, 31000 + 10 * index
        )
        comparisons.append(row)
        horizon_rows.extend(bands)
    stabilization_indices = list(range(2, len(comparisons)))
    if stabilization_indices:
        apply_holm(comparisons, stabilization_indices)

    diagnostic_labels = ["direct_joint_d64", "full_history_anchor_020", "recent_normal"]
    diagnostic_labels.extend(group for group in optional if group in specs)
    diagnostic_bands = []
    diagnostic_deciles = []
    diagnostic_payload = {}
    for label in diagnostic_labels:
        bands, deciles, payload = ensemble_diagnostics(label, raw[label])
        diagnostic_bands.extend(bands)
        diagnostic_deciles.extend(deciles)
        diagnostic_payload[label] = payload
    save_diagnostic_plots(output, raw, diagnostic_labels)

    summary_by_group = {row["group"]: row for row in summaries}
    normal = summary_by_group["recent_normal"]
    recovery = []
    denominator = normal["single_mean_minus_mean_ensemble_mw"]
    for group in ("recent_ema", "recent_checkpoint_avg", *optional.keys()):
        if group in summary_by_group:
            recovery.append(
                {
                    "group": group,
                    "single_mae_improvement_vs_normal_mw": (
                        normal["cluster_mae_mw_mean"]
                        - summary_by_group[group]["cluster_mae_mw_mean"]
                    ),
                    "ensemble_gain_recovery_fraction": (
                        normal["cluster_mae_mw_mean"]
                        - summary_by_group[group]["cluster_mae_mw_mean"]
                    ) / denominator,
                    "candidate_single_ensemble_gap_mw": summary_by_group[group][
                        "single_mean_minus_mean_ensemble_mw"
                    ],
                }
            )

    recent_diag = diagnostic_payload["recent_normal"]
    direct_diag = diagnostic_payload["direct_joint_d64"]
    kd_gate = {
        "normal_single_ensemble_gap_mw": normal["single_mean_minus_mean_ensemble_mw"],
        "recent_error_correlation": recent_diag["mean_pairwise_signed_error_correlation"],
        "direct_error_correlation": direct_diag["mean_pairwise_signed_error_correlation"],
        "recent_minus_direct_error_correlation": (
            recent_diag["mean_pairwise_signed_error_correlation"]
            - direct_diag["mean_pairwise_signed_error_correlation"]
        ),
        "recent_ensemble_minus_direct_ensemble_mw": (
            normal["mean_ensemble_mae_mw"]
            - summary_by_group["direct_joint_d64"]["mean_ensemble_mae_mw"]
        ),
    }
    kd_gate["go_plain_kd"] = bool(
        kd_gate["normal_single_ensemble_gap_mw"] >= 8.0
        and kd_gate["recent_minus_direct_error_correlation"] <= -0.01
        and kd_gate["recent_ensemble_minus_direct_ensemble_mw"] <= 1.0
    )

    payload = {
        "protocol": {
            "selection_split": "2023 validation",
            "test_used_in_this_stage": False,
            "pristine_test_available": False,
            "pristine_test_note": (
                "Legacy paper-suite artifacts already contain test metrics; this stage does not read them."
            ),
            "seeds": SEEDS,
            "bootstrap": "sample-weighted paired UTC-day blocks",
            "bootstrap_repeats": args.bootstrap_repeats,
            "primary_metric": "10-seed mean validation cluster MAE",
            "mean_teacher": "arithmetic mean of ten Normal checkpoints",
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "horizon_comparisons": horizon_rows,
        "ensemble_diagnostic_bands": diagnostic_bands,
        "ensemble_diagnostics": diagnostic_payload,
        "recovery": recovery,
        "kd_gate": kd_gate,
    }
    write_json(output / "analysis.json", payload)
    write_csv(output / "summaries.csv", summaries)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "horizon_comparisons.csv", horizon_rows)
    write_csv(output / "ensemble_diagnostic_bands.csv", diagnostic_bands)
    write_csv(output / "disagreement_deciles.csv", diagnostic_deciles)
    write_csv(output / "recovery.csv", recovery)
    print(json.dumps(payload["kd_gate"], indent=2))
    print(f"[done] analysis={output}")


if __name__ == "__main__":
    main()
