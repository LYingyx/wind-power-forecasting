from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
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
NEW_GROUPS = (
    "explicit_additive",
    "explicit_multiplicative",
    "explicit_asymmetric_modulation",
)
LABELS = (
    "blackbox",
    "additive",
    "multiplicative",
    "asymmetric_modulation",
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
    group_paths: dict[str, Path], repeats: int
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    errors: dict[str, np.ndarray] = {}
    origins = None
    for label, path in group_paths.items():
        errors[label], current_origins = ensemble_error(path)
        if origins is None:
            origins = current_origins
        else:
            np.testing.assert_array_equal(origins, current_origins)
    assert origins is not None
    contrasts = {
        "additive_vs_blackbox_benefit": errors["blackbox"] - errors["additive"],
        "multiplicative_vs_blackbox_benefit": (
            errors["blackbox"] - errors["multiplicative"]
        ),
        "asymmetric_modulation_vs_blackbox_benefit": (
            errors["blackbox"] - errors["asymmetric_modulation"]
        ),
        "multiplicative_vs_additive_benefit": (
            errors["additive"] - errors["multiplicative"]
        ),
        "asymmetric_modulation_vs_multiplicative_benefit": (
            errors["multiplicative"] - errors["asymmetric_modulation"]
        ),
    }
    scalar: dict[str, Any] = {}
    horizon_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    horizon_band_rows = []
    horizon_bands = (
        ("0.25-1h", slice(0, 4)),
        ("1.25-3h", slice(4, 12)),
        ("3.25-6h", slice(12, 24)),
    )
    for index, (label, delta) in enumerate(contrasts.items()):
        horizon_values[label] = block_bootstrap_matrix(
            delta, origins, repeats, 11200 + index
        )
        point, low, high, days = block_bootstrap_matrix(
            delta.mean(axis=1, keepdims=True), origins, repeats, 11300 + index
        )
        scalar[label] = {
            "benefit_mw": float(point[0]),
            "ci95_low_mw": float(low[0]),
            "ci95_high_mw": float(high[0]),
            "utc_day_blocks": days,
        }
        for band_index, (band, lead_slice) in enumerate(horizon_bands):
            band_point, band_low, band_high, band_days = block_bootstrap_matrix(
                delta[:, lead_slice].mean(axis=1, keepdims=True),
                origins,
                repeats,
                11500 + 10 * index + band_index,
            )
            horizon_band_rows.append(
                {
                    "contrast": label,
                    "horizon_band": band,
                    "benefit_mw": float(band_point[0]),
                    "ci95_low_mw": float(band_low[0]),
                    "ci95_high_mw": float(band_high[0]),
                    "utc_day_blocks": band_days,
                }
            )
    horizon_rows = []
    for lead in range(24):
        row: dict[str, Any] = {"lead": lead + 1, "lead_hours": (lead + 1) / 4.0}
        for label, (point, low, high, days) in horizon_values.items():
            row[f"{label}_mw"] = float(point[lead])
            row[f"{label}_ci95_low_mw"] = float(low[lead])
            row[f"{label}_ci95_high_mw"] = float(high[lead])
            row["utc_day_blocks"] = days
        horizon_rows.append(row)
    return scalar, horizon_rows, horizon_band_rows


def gate_statistics(group: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curves = []
    logits = []
    target = np.linspace(0.0, 3.0, 24)
    for seed in SEEDS:
        state = torch.load(
            group / f"seed{seed}" / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state"]
        current_logits = state["horizon_gate_logits"].detach().numpy().astype(np.float64)
        knots = 2.0 / (1.0 + np.exp(-current_logits))
        curve = np.interp(target, np.arange(4, dtype=np.float64), knots)
        curves.append(curve)
        logits.append(current_logits)
    curves_array = np.stack(curves)
    normalized = curves_array / curves_array.mean(axis=1, keepdims=True)
    rows = []
    for lead in range(24):
        raw = t_interval(curves_array[:, lead])
        norm = t_interval(normalized[:, lead])
        rows.append(
            {
                "lead": lead + 1,
                "lead_hours": (lead + 1) / 4.0,
                "raw_gate_mean": raw[0],
                "raw_gate_sample_std": raw[1],
                "raw_gate_ci95_low": raw[2],
                "raw_gate_ci95_high": raw[3],
                "normalized_gate_mean": norm[0],
                "normalized_gate_sample_std": norm[1],
                "normalized_gate_ci95_low": norm[2],
                "normalized_gate_ci95_high": norm[3],
            }
        )
    long_short = normalized[:, -4:].mean(axis=1) - normalized[:, :4].mean(axis=1)
    long_short_stats = t_interval(long_short)
    return (
        {
            "raw_gate_curves": curves_array.tolist(),
            "normalized_gate_curves": normalized.tolist(),
            "gate_logits": np.stack(logits).tolist(),
            "normalized_long_minus_short_mean": long_short_stats[0],
            "normalized_long_minus_short_sample_std": long_short_stats[1],
            "normalized_long_minus_short_ci95_low": long_short_stats[2],
            "normalized_long_minus_short_ci95_high": long_short_stats[3],
            "long_greater_than_short_seeds": int(np.sum(long_short > 0.0)),
        },
        rows,
    )


def component_statistics(
    groups_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summaries = []
    horizon_rows = []
    additivity_max = {}
    settings = {
        "explicit_additive": "cluster_state_main_correction_mw",
        "explicit_multiplicative": "cluster_state_weather_interaction_correction_mw",
    }
    for group, second_key in settings.items():
        main_values = []
        second_values = []
        raw_values = []
        seed_rows = []
        maximum = 0.0
        for seed in SEEDS:
            with np.load(groups_root / group / f"seed{seed}" / "predictions_val.npz") as archive:
                main = archive["cluster_weather_main_correction_mw"].astype(np.float64)
                second = archive[second_key].astype(np.float64)
                raw = archive["cluster_raw_correction_mw"].astype(np.float64)
            maximum = max(maximum, float(np.max(np.abs(raw - main - second))))
            main_values.append(main)
            second_values.append(second)
            raw_values.append(raw)
            seed_rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "weather_main_abs_mw": float(np.abs(main).mean()),
                    "second_branch_abs_mw": float(np.abs(second).mean()),
                    "raw_correction_abs_mw": float(np.abs(raw).mean()),
                    "opposing_sign_fraction": float(np.mean(main * second < 0.0)),
                }
            )
        additivity_max[group] = maximum
        summaries.extend(seed_rows)
        main_stack = np.stack(main_values)
        second_stack = np.stack(second_values)
        raw_stack = np.stack(raw_values)
        for lead in range(24):
            horizon_rows.append(
                {
                    "group": group,
                    "lead": lead + 1,
                    "lead_hours": (lead + 1) / 4.0,
                    "weather_main_abs_mw": float(np.abs(main_stack[..., lead]).mean()),
                    "second_branch_abs_mw": float(np.abs(second_stack[..., lead]).mean()),
                    "raw_correction_abs_mw": float(np.abs(raw_stack[..., lead]).mean()),
                    "opposing_sign_fraction": float(
                        np.mean(main_stack[..., lead] * second_stack[..., lead] < 0.0)
                    ),
                }
            )
    return summaries, horizon_rows, additivity_max


def frozen_baseline_audit(
    groups_root: Path, conditional_root: Path
) -> dict[str, Any]:
    exact = {}
    replay = {}
    for group in NEW_GROUPS:
        exact[group] = []
        replay[group] = []
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
            exact[group].append(
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
                np.testing.assert_allclose(
                    candidate["cluster_target_mw"], prior["cluster_target_mw"]
                )
                replay[group].append(
                    float(
                        np.abs(
                            candidate["baseline_cluster_prediction_mw"].astype(np.float64)
                            - prior["cluster_prediction_mw"].astype(np.float64)
                        ).mean()
                    )
                )
    all_replay = [value for values in replay.values() for value in values]
    return {
        "checkpoint_state_exact_all_runs": all(all(values) for values in exact.values()),
        "checkpoint_state_exact_by_group": {
            group: all(values) for group, values in exact.items()
        },
        "checkpoint_state_exact_run_values_by_group": exact,
        "prediction_replay_abs_difference_mean_mw": float(np.mean(all_replay)),
        "prediction_replay_abs_difference_max_run_mw": float(np.max(all_replay)),
        "prediction_replay_by_group_mw": replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze explicit correction fusion matrix")
    parser.add_argument(
        "--experiment-root", default="artifacts/explicit_interaction_matrix_v1"
    )
    parser.add_argument("--blackbox-root", default="artifacts/state_forcing_ladder_v1")
    parser.add_argument(
        "--conditional-root", default="artifacts/conditional_forcing_validation_v1"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    root = Path(args.experiment_root).resolve()
    groups_root = root / "groups"
    blackbox = (
        Path(args.blackbox_root).resolve()
        / "groups"
        / "frozen_cluster_recent_1h"
    )
    conditional = Path(args.conditional_root).resolve()
    if not (root / "EXPLICIT_INTERACTION_COMPLETE").is_file():
        raise FileNotFoundError(root / "EXPLICIT_INTERACTION_COMPLETE")
    missing = [
        str(groups_root / group / f"seed{seed}" / "COMPLETE")
        for group in NEW_GROUPS
        for seed in SEEDS
        if not (groups_root / group / f"seed{seed}" / "COMPLETE").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing completed runs:\n" + "\n".join(missing))

    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (root / "analysis").resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    group_paths = {
        "blackbox": blackbox,
        "additive": groups_root / "explicit_additive",
        "multiplicative": groups_root / "explicit_multiplicative",
        "asymmetric_modulation": groups_root / "explicit_asymmetric_modulation",
    }
    summaries = [summarize_group(group_paths[label], SEEDS) for label in LABELS]
    comparisons_to_run = (
        ("additive", "blackbox"),
        ("multiplicative", "blackbox"),
        ("asymmetric_modulation", "blackbox"),
        ("multiplicative", "additive"),
        ("asymmetric_modulation", "multiplicative"),
    )
    comparisons = [
        {
            "comparison": f"{candidate}_vs_{reference}",
            **compare_groups(
                group_paths[candidate],
                group_paths[reference],
                SEEDS,
                args.bootstrap_repeats,
                11400 + index,
            ),
        }
        for index, (candidate, reference) in enumerate(comparisons_to_run)
    ]
    contrasts, horizon, horizon_bands = contrast_statistics(
        group_paths, args.bootstrap_repeats
    )
    component_seed, component_horizon, additivity = component_statistics(groups_root)
    audit = frozen_baseline_audit(groups_root, conditional)
    payload = {
        "protocol": {
            "split": "2023 validation",
            "test_split_used": False,
            "model_seeds": SEEDS,
            "bootstrap": "sample-weighted paired UTC-day blocks",
            "bootstrap_repeats": args.bootstrap_repeats,
            "blackbox_reused": str(blackbox),
            "new_groups": list(NEW_GROUPS),
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "ensemble_contrasts": contrasts,
        "ensemble_horizon_bands": horizon_bands,
        "component_additivity_max_abs_mw": additivity,
        "frozen_baseline_audit": audit,
    }
    write_csv(output / "summaries.csv", summaries)
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(
        output / "ensemble_contrasts.csv",
        [{"contrast": label, **values} for label, values in contrasts.items()],
    )
    write_csv(output / "horizon.csv", horizon)
    write_csv(output / "horizon_bands.csv", horizon_bands)
    write_csv(output / "component_seed.csv", component_seed)
    write_csv(output / "component_horizon.csv", component_horizon)
    temporary = output / f".analysis.json.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "analysis.json")
    print(json.dumps({"output": str(output), "comparisons": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
