from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
import torch

try:
    from scripts.analyze_conditional_forcing_validation import block_bootstrap_matrix
    from scripts.analyze_incremental_mechanism_v2 import (
        compare_groups,
        group_predictions,
        summarize_group,
    )
except ModuleNotFoundError:
    from analyze_conditional_forcing_validation import block_bootstrap_matrix  # type: ignore[no-redef]
    from analyze_incremental_mechanism_v2 import (  # type: ignore[no-redef]
        compare_groups,
        group_predictions,
        summarize_group,
    )


SEEDS = list(range(42, 52))
NEW_GROUPS = {
    "film": "horizon_film",
    "gated_multiplicative": "horizon_gated_multiplicative",
    "conditioned_query": "horizon_conditioned_query",
    "attention_bias": "horizon_attention_bias",
    "conditional_lora": "horizon_conditional_lora",
}
HORIZON_BANDS = (
    ("0.25-1h", slice(0, 4)),
    ("1.25-3h", slice(4, 12)),
    ("3.25-6h", slice(12, 24)),
)


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
    sample_std = float(values.std(ddof=1))
    half_width = float(
        student_t.ppf(0.975, df=len(values) - 1)
        * sample_std
        / np.sqrt(len(values))
    )
    return mean, sample_std, mean - half_width, mean + half_width


def add_seedwise_p_values(
    rows: list[dict[str, Any]], primary_count: int
) -> None:
    raw = []
    for row in rows:
        mean = float(row["seedwise_difference_mean_mw"])
        std = float(row["seedwise_difference_sample_std_mw"])
        statistic = mean / max(std / np.sqrt(len(SEEDS)), 1e-12)
        p_value = float(2.0 * student_t.sf(abs(statistic), df=len(SEEDS) - 1))
        row["seedwise_p_two_sided"] = p_value
        row["primary_family"] = len(raw) < primary_count
        raw.append(p_value)
    primary = np.asarray(raw[:primary_count], dtype=np.float64)
    order = np.argsort(primary)
    adjusted = np.empty_like(primary)
    running = 0.0
    family_size = len(primary)
    for rank, index in enumerate(order):
        current = min(1.0, (family_size - rank) * primary[index])
        running = max(running, current)
        adjusted[index] = running
    for index, row in enumerate(rows):
        row["seedwise_holm_p_primary_family"] = (
            float(adjusted[index]) if index < primary_count else None
        )


def ensemble_error(group: Path) -> tuple[np.ndarray, np.ndarray]:
    _, prediction = group_predictions(group, SEEDS)
    return (
        np.abs(
            prediction["cluster_prediction_mw"].astype(np.float64)
            - prediction["cluster_target_mw"].astype(np.float64)
        ),
        prediction["origin_unix"].astype(np.int64),
    )


def contrast_statistics(
    group_paths: dict[str, Path], comparisons: list[tuple[str, str]], repeats: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: dict[str, np.ndarray] = {}
    origins = None
    for label, path in group_paths.items():
        errors[label], current_origins = ensemble_error(path)
        if origins is None:
            origins = current_origins
        else:
            np.testing.assert_array_equal(origins, current_origins)
    assert origins is not None
    scalar_rows = []
    band_rows = []
    for index, (candidate, reference) in enumerate(comparisons):
        label = f"{candidate}_vs_{reference}_benefit"
        delta = errors[reference] - errors[candidate]
        point, low, high, days = block_bootstrap_matrix(
            delta.mean(axis=1, keepdims=True), origins, repeats, 22100 + index
        )
        scalar_rows.append(
            {
                "contrast": label,
                "candidate": candidate,
                "reference": reference,
                "benefit_mw": float(point[0]),
                "ci95_low_mw": float(low[0]),
                "ci95_high_mw": float(high[0]),
                "utc_day_blocks": days,
            }
        )
        for band_index, (band, lead_slice) in enumerate(HORIZON_BANDS):
            band_point, band_low, band_high, band_days = block_bootstrap_matrix(
                delta[:, lead_slice].mean(axis=1, keepdims=True),
                origins,
                repeats,
                22200 + 10 * index + band_index,
            )
            band_rows.append(
                {
                    "contrast": label,
                    "candidate": candidate,
                    "reference": reference,
                    "horizon_band": band,
                    "benefit_mw": float(band_point[0]),
                    "ci95_low_mw": float(band_low[0]),
                    "ci95_high_mw": float(band_high[0]),
                    "utc_day_blocks": band_days,
                }
            )
    return scalar_rows, band_rows


def gate_statistics(
    group_paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    horizon_rows = []
    seed_rows = []
    payload: dict[str, Any] = {}
    for label in NEW_GROUPS:
        curves = []
        for seed in SEEDS:
            checkpoint = torch.load(
                group_paths[label] / f"seed{seed}" / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )
            logits = checkpoint["model_state"]["horizon_gate_logits"]
            gate = (2.0 * torch.sigmoid(logits.float())).numpy().astype(np.float64)
            if gate.shape != (24,):
                raise ValueError(f"Unexpected gate shape for {label}/seed{seed}: {gate.shape}")
            curves.append(gate)
            seed_rows.append(
                {
                    "group": label,
                    "seed": seed,
                    "gate_mean": float(gate.mean()),
                    "gate_abs_deviation_from_one": float(np.abs(gate - 1.0).mean()),
                    "gate_long_minus_short": float(gate[12:].mean() - gate[:4].mean()),
                    "gate_within_curve_range": float(gate.max() - gate.min()),
                }
            )
        array = np.stack(curves)
        for lead in range(24):
            stats = t_interval(array[:, lead])
            horizon_rows.append(
                {
                    "group": label,
                    "lead": lead + 1,
                    "lead_hours": (lead + 1) / 4.0,
                    "gate_mean": stats[0],
                    "gate_sample_std": stats[1],
                    "gate_ci95_low": stats[2],
                    "gate_ci95_high": stats[3],
                }
            )
        long_short = array[:, 12:].mean(axis=1) - array[:, :4].mean(axis=1)
        long_short_stats = t_interval(long_short)
        centered = array - array.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        correlations = []
        for left in range(len(SEEDS)):
            for right in range(left + 1, len(SEEDS)):
                denominator = norms[left] * norms[right]
                if denominator > 1e-12:
                    correlations.append(
                        float(np.dot(centered[left], centered[right]) / denominator)
                    )
        payload[label] = {
            "gate_curves": array.tolist(),
            "mean_abs_deviation_from_one": float(np.abs(array - 1.0).mean()),
            "mean_across_seed_std": float(array.std(axis=0, ddof=1).mean()),
            "long_minus_short_mean": long_short_stats[0],
            "long_minus_short_sample_std": long_short_stats[1],
            "long_minus_short_ci95_low": long_short_stats[2],
            "long_minus_short_ci95_high": long_short_stats[3],
            "long_greater_than_short_seeds": int(np.sum(long_short > 0.0)),
            "mean_pairwise_centered_curve_correlation": (
                float(np.mean(correlations)) if correlations else None
            ),
        }
    return horizon_rows, seed_rows, payload


def diagnostic_statistics(
    group_paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    settings = {
        "film": "interaction_update_abs_mean",
        "gated_multiplicative": "interaction_update_abs_mean",
        "conditioned_query": "weather_attention_shift_l1",
        "attention_bias": "weather_attention_shift_l1",
        "conditional_lora": "interaction_update_abs_mean",
    }
    horizon_rows = []
    seed_rows = []
    payload: dict[str, Any] = {}
    for label, key in settings.items():
        curves = []
        for seed in SEEDS:
            with np.load(
                group_paths[label] / f"seed{seed}" / "predictions_val.npz"
            ) as archive:
                values = archive[key].astype(np.float64)
            if values.shape != (32550, 24):
                raise ValueError(f"Unexpected {key} shape for {label}/seed{seed}")
            curve = values.mean(axis=0)
            curves.append(curve)
            seed_rows.append(
                {
                    "group": label,
                    "seed": seed,
                    "diagnostic": key,
                    "overall_mean": float(curve.mean()),
                    "long_minus_short": float(curve[12:].mean() - curve[:4].mean()),
                }
            )
        array = np.stack(curves)
        normalized = array / np.maximum(array.mean(axis=1, keepdims=True), 1e-12)
        for lead in range(24):
            raw_stats = t_interval(array[:, lead])
            normalized_stats = t_interval(normalized[:, lead])
            horizon_rows.append(
                {
                    "group": label,
                    "diagnostic": key,
                    "lead": lead + 1,
                    "lead_hours": (lead + 1) / 4.0,
                    "mean": raw_stats[0],
                    "sample_std": raw_stats[1],
                    "ci95_low": raw_stats[2],
                    "ci95_high": raw_stats[3],
                    "normalized_mean": normalized_stats[0],
                    "normalized_sample_std": normalized_stats[1],
                    "normalized_ci95_low": normalized_stats[2],
                    "normalized_ci95_high": normalized_stats[3],
                }
            )
        long_short = normalized[:, 12:].mean(axis=1) - normalized[:, :4].mean(axis=1)
        long_short_stats = t_interval(long_short)
        payload[label] = {
            "diagnostic": key,
            "seed_curves": array.tolist(),
            "normalized_seed_curves": normalized.tolist(),
            "normalized_long_minus_short_mean": long_short_stats[0],
            "normalized_long_minus_short_sample_std": long_short_stats[1],
            "normalized_long_minus_short_ci95_low": long_short_stats[2],
            "normalized_long_minus_short_ci95_high": long_short_stats[3],
            "long_greater_than_short_seeds": int(np.sum(long_short > 0.0)),
        }
    return horizon_rows, seed_rows, payload


def gate_ablation_statistics(
    group_paths: dict[str, Path], repeats: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    band_rows = []
    for ablation_index, ablation in enumerate(("zero", "one")):
        for group_index, label in enumerate(NEW_GROUPS):
            trained_predictions = []
            ablated_predictions = []
            seed_benefits = []
            origins = None
            targets = None
            for seed in SEEDS:
                with np.load(
                    group_paths[label] / f"seed{seed}" / "predictions_val.npz"
                ) as trained, np.load(
                    group_paths[label]
                    / f"seed{seed}"
                    / f"predictions_gate_{ablation}_val.npz"
                ) as ablated:
                    np.testing.assert_array_equal(
                        trained["origin_unix"], ablated["origin_unix"]
                    )
                    np.testing.assert_allclose(
                        trained["cluster_target_mw"], ablated["cluster_target_mw"]
                    )
                    current_origins = trained["origin_unix"].astype(np.int64)
                    current_targets = trained["cluster_target_mw"].astype(np.float64)
                    if origins is None:
                        origins = current_origins
                        targets = current_targets
                    else:
                        np.testing.assert_array_equal(origins, current_origins)
                        np.testing.assert_allclose(targets, current_targets)
                    trained_prediction = trained["cluster_prediction_mw"].astype(np.float64)
                    ablated_prediction = ablated["cluster_prediction_mw"].astype(np.float64)
                trained_predictions.append(trained_prediction)
                ablated_predictions.append(ablated_prediction)
                seed_benefits.append(
                    float(np.abs(ablated_prediction - current_targets).mean())
                    - float(np.abs(trained_prediction - current_targets).mean())
                )
            assert origins is not None and targets is not None
            trained_ensemble = np.mean(trained_predictions, axis=0)
            ablated_ensemble = np.mean(ablated_predictions, axis=0)
            delta = np.abs(ablated_ensemble - targets) - np.abs(
                trained_ensemble - targets
            )
            point, low, high, days = block_bootstrap_matrix(
                delta.mean(axis=1, keepdims=True),
                origins,
                repeats,
                22600 + 10 * ablation_index + group_index,
            )
            seed_stats = t_interval(np.asarray(seed_benefits))
            rows.append(
                {
                    "group": label,
                    "ablation_gate": ablation,
                    "trained_over_ablation_ensemble_benefit_mw": float(point[0]),
                    "ensemble_ci95_low_mw": float(low[0]),
                    "ensemble_ci95_high_mw": float(high[0]),
                    "seedwise_benefit_mean_mw": seed_stats[0],
                    "seedwise_benefit_sample_std_mw": seed_stats[1],
                    "seedwise_ci95_low_mw": seed_stats[2],
                    "seedwise_ci95_high_mw": seed_stats[3],
                    "trained_better_seeds": int(
                        np.sum(np.asarray(seed_benefits) > 0.0)
                    ),
                    "seedwise_benefits_mw": seed_benefits,
                    "utc_day_blocks": days,
                }
            )
            for band_index, (band, lead_slice) in enumerate(HORIZON_BANDS):
                band_point, band_low, band_high, band_days = block_bootstrap_matrix(
                    delta[:, lead_slice].mean(axis=1, keepdims=True),
                    origins,
                    repeats,
                    22800 + 100 * ablation_index + 10 * group_index + band_index,
                )
                band_rows.append(
                    {
                        "group": label,
                        "ablation_gate": ablation,
                        "horizon_band": band,
                        "trained_over_ablation_benefit_mw": float(band_point[0]),
                        "ci95_low_mw": float(band_low[0]),
                        "ci95_high_mw": float(band_high[0]),
                        "utc_day_blocks": band_days,
                    }
                )
    return rows, band_rows


def frozen_baseline_audit(groups_root: Path, conditional_root: Path) -> dict[str, Any]:
    exact = []
    replay = []
    origins = None
    targets = None
    for group in NEW_GROUPS.values():
        for seed in SEEDS:
            prior_state = torch.load(
                conditional_root
                / "groups"
                / "history_baseline"
                / f"seed{seed}"
                / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )["model_state"]
            checkpoint = torch.load(
                groups_root / group / f"seed{seed}" / "best_model.pt",
                map_location="cpu",
                weights_only=False,
            )["model_state"]
            exact.append(
                all(
                    f"baseline_model.{name}" in checkpoint
                    and torch.equal(checkpoint[f"baseline_model.{name}"], value)
                    for name, value in prior_state.items()
                )
            )
            with np.load(
                groups_root / group / f"seed{seed}" / "predictions_val.npz"
            ) as candidate, np.load(
                conditional_root
                / "groups"
                / "history_baseline"
                / f"seed{seed}"
                / "predictions_val.npz"
            ) as prior:
                np.testing.assert_array_equal(candidate["origin_unix"], prior["origin_unix"])
                np.testing.assert_allclose(candidate["cluster_target_mw"], prior["cluster_target_mw"])
                if origins is None:
                    origins = candidate["origin_unix"].copy()
                    targets = candidate["cluster_target_mw"].copy()
                else:
                    np.testing.assert_array_equal(candidate["origin_unix"], origins)
                    np.testing.assert_allclose(candidate["cluster_target_mw"], targets)
                replay.append(
                    float(
                        np.abs(
                            candidate["baseline_cluster_prediction_mw"].astype(np.float64)
                            - prior["cluster_prediction_mw"].astype(np.float64)
                        ).mean()
                    )
                )
    assert origins is not None
    day_count = len(np.unique(origins.astype("datetime64[s]").astype("datetime64[D]")))
    parameter_counts = []
    for group in NEW_GROUPS.values():
        metrics = json.loads(
            (groups_root / group / "seed42" / "metrics_val.json").read_text()
        )
        parameter_counts.append(int(metrics["parameter_count"]))
    return {
        "checkpoint_state_exact_all_runs": all(exact),
        "checkpoint_state_exact_runs": exact,
        "prediction_replay_abs_difference_mean_mw": float(np.mean(replay)),
        "prediction_replay_abs_difference_max_run_mw": float(np.max(replay)),
        "forecast_origins": int(len(origins)),
        "utc_days": day_count,
        "parameter_count_min": min(parameter_counts),
        "parameter_count_max": max(parameter_counts),
        "parameter_count_spread": max(parameter_counts) - min(parameter_counts),
        "parameter_count_spread_fraction": (
            (max(parameter_counts) - min(parameter_counts)) / min(parameter_counts)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze horizon-control operator matrix")
    parser.add_argument("--experiment-root", default="artifacts/horizon_control_matrix_v1")
    parser.add_argument("--blackbox-root", default="artifacts/state_forcing_ladder_v1")
    parser.add_argument(
        "--explicit-root", default="artifacts/explicit_interaction_matrix_v1"
    )
    parser.add_argument(
        "--conditional-root", default="artifacts/conditional_forcing_validation_v1"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.experiment_root).resolve()
    groups_root = root / "groups"
    if not (root / "HORIZON_CONTROL_COMPLETE").is_file():
        raise FileNotFoundError(root / "HORIZON_CONTROL_COMPLETE")
    if not (root / "HORIZON_GATE_ABLATION_COMPLETE").is_file():
        raise FileNotFoundError(root / "HORIZON_GATE_ABLATION_COMPLETE")
    missing = [
        str(groups_root / group / f"seed{seed}" / "COMPLETE")
        for group in NEW_GROUPS.values()
        for seed in SEEDS
        if not (groups_root / group / f"seed{seed}" / "COMPLETE").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing completed runs:\n" + "\n".join(missing))
    test_artifacts = [path for path in root.rglob("*") if path.is_file() and "test" in path.name.lower()]
    if test_artifacts:
        raise ValueError(f"Unexpected test artifacts: {test_artifacts}")

    group_paths = {
        label: groups_root / group for label, group in NEW_GROUPS.items()
    }
    group_paths["blackbox"] = (
        Path(args.blackbox_root).resolve()
        / "groups"
        / "frozen_cluster_recent_1h"
    )
    group_paths["ungated_multiplicative"] = (
        Path(args.explicit_root).resolve()
        / "groups"
        / "explicit_multiplicative"
    )
    labels = ["blackbox", "ungated_multiplicative", *NEW_GROUPS.keys()]
    summaries = [summarize_group(group_paths[label], SEEDS) for label in labels]
    for summary, label in zip(summaries, labels, strict=True):
        summary["label"] = label

    comparisons = [(label, "blackbox") for label in NEW_GROUPS]
    comparisons.append(("gated_multiplicative", "ungated_multiplicative"))
    comparisons.extend(
        (label, "gated_multiplicative")
        for label in NEW_GROUPS
        if label != "gated_multiplicative"
    )
    paired = [
        {
            "comparison": f"{candidate}_vs_{reference}",
            **compare_groups(
                group_paths[candidate],
                group_paths[reference],
                SEEDS,
                args.bootstrap_repeats,
                22000 + index,
            ),
        }
        for index, (candidate, reference) in enumerate(comparisons)
    ]
    add_seedwise_p_values(paired, primary_count=len(NEW_GROUPS))
    contrasts, bands = contrast_statistics(
        group_paths, comparisons, args.bootstrap_repeats
    )
    gate_horizon, gate_seed, gate_payload = gate_statistics(group_paths)
    diagnostic_horizon, diagnostic_seed, diagnostic_payload = diagnostic_statistics(
        group_paths
    )
    gate_ablation, gate_ablation_bands = gate_ablation_statistics(
        group_paths, args.bootstrap_repeats
    )
    audit = frozen_baseline_audit(
        groups_root, Path(args.conditional_root).resolve()
    )

    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "analysis").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "split": "2023 validation",
            "test_split_used": False,
            "model_seeds": SEEDS,
            "bootstrap": "sample-weighted paired UTC-day blocks",
            "bootstrap_repeats": args.bootstrap_repeats,
            "new_groups": NEW_GROUPS,
            "blackbox_reused": str(group_paths["blackbox"]),
            "ungated_multiplicative_reused": str(
                group_paths["ungated_multiplicative"]
            ),
        },
        "summaries": summaries,
        "paired_comparisons": paired,
        "ensemble_contrasts": contrasts,
        "ensemble_horizon_bands": bands,
        "gate_statistics": gate_payload,
        "operator_diagnostics": diagnostic_payload,
        "gate_ablation": gate_ablation,
        "gate_ablation_horizon_bands": gate_ablation_bands,
        "frozen_baseline_audit": audit,
    }
    write_csv(output / "summaries.csv", summaries)
    write_csv(output / "paired_comparisons.csv", paired)
    write_csv(output / "ensemble_contrasts.csv", contrasts)
    write_csv(output / "horizon_bands.csv", bands)
    write_csv(output / "gate_horizon.csv", gate_horizon)
    write_csv(output / "gate_seed.csv", gate_seed)
    write_csv(output / "diagnostic_horizon.csv", diagnostic_horizon)
    write_csv(output / "diagnostic_seed.csv", diagnostic_seed)
    write_csv(output / "gate_ablation.csv", gate_ablation)
    write_csv(output / "gate_ablation_horizon_bands.csv", gate_ablation_bands)
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "analysis.json")
    print(json.dumps({"output": str(output), "comparisons": len(paired)}, indent=2))


if __name__ == "__main__":
    main()
