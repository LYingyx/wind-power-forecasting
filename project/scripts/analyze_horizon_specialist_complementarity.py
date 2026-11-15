from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from scripts.analyze_incremental_mechanism_v2 import group_predictions


SEEDS = list(range(42, 52))
HORIZON_BANDS = (
    ("0.25-1h", slice(0, 4)),
    ("1.25-3h", slice(4, 12)),
    ("3.25-6h", slice(12, 24)),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_simplex(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    members = predictions.shape[-1]

    def objective(weight: np.ndarray) -> float:
        combined = np.tensordot(predictions, weight, axes=([-1], [0]))
        return float(np.abs(combined - target).mean())

    starts = [np.full(members, 1.0 / members)]
    for index in range(members):
        start = np.zeros(members)
        start[index] = 1.0
        starts.append(start)
    best_weight = starts[0]
    best_value = objective(best_weight)
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * members,
            constraints={"type": "eq", "fun": lambda value: value.sum() - 1.0},
            options={"maxiter": 250, "ftol": 1e-8},
        )
        candidate = np.clip(result.x, 0.0, 1.0)
        candidate /= candidate.sum()
        value = objective(candidate)
        if value < best_value:
            best_weight, best_value = candidate, value
    return best_weight


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(prediction.astype(np.float64) - target).mean())


def band_mae(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        label: mae(prediction[:, selection], target[:, selection])
        for label, selection in HORIZON_BANDS
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validated complementarity audit for horizon specialists"
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-dir", default="artifacts/horizon_specialist_v1/stage0"
    )
    parser.add_argument("--split", choices=("val",), default="val")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    output = (project / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    roots = {
        "Normal": project
        / "artifacts/ensemble_stabilization_v1/groups/weak_anchor_recent_1h",
        "NearestHour": project
        / "artifacts/paper_ablation_completion_v1/groups/nearest_hour",
        "NoAnchor": project
        / "artifacts/paper_ablation_completion_v1/groups/no_anchor",
        "KD": project / "artifacts/ensemble_stabilization_v1/groups/kd_050",
        "KDNoAnchor": project
        / "artifacts/paper_ablation_completion_v1/groups/kd_no_anchor",
        "FullHistory": project
        / "artifacts/conditional_forcing_validation_v1/groups/anchor_020",
    }
    labels = list(roots)
    ensembles = {}
    for label in tqdm(labels, desc="Load validation ensembles", unit="group"):
        _, ensembles[label] = group_predictions(roots[label], SEEDS, args.split)
    reference = ensembles[labels[0]]
    for label in labels[1:]:
        np.testing.assert_array_equal(
            ensembles[label]["origin_unix"], reference["origin_unix"]
        )
        np.testing.assert_allclose(
            ensembles[label]["cluster_target_mw"],
            reference["cluster_target_mw"],
        )

    target = reference["cluster_target_mw"].astype(np.float64)
    origin = reference["origin_unix"]
    stack = np.stack(
        [ensembles[label]["cluster_prediction_mw"] for label in labels], axis=-1
    ).astype(np.float64)
    single_rows = []
    for index, label in enumerate(labels):
        row: dict[str, Any] = {"model": label, "overall_mae_mw": mae(stack[..., index], target)}
        row.update({f"{key}_mae_mw": value for key, value in band_mae(stack[..., index], target).items()})
        single_rows.append(row)
    write_csv(output / "constituent_mae.csv", single_rows)

    residual_rows = []
    residual = stack - target[..., None]
    for band, selection in HORIZON_BANDS:
        flat = residual[:, selection].reshape(-1, len(labels))
        correlation = np.corrcoef(flat, rowvar=False)
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                residual_rows.append(
                    {
                        "horizon_band": band,
                        "left": labels[left],
                        "right": labels[right],
                        "signed_error_correlation": float(correlation[left, right]),
                    }
                )
    write_csv(output / "residual_correlations.csv", residual_rows)

    dates = origin.astype("datetime64[s]").astype("datetime64[M]").astype(int)
    months = dates % 12 + 1
    quarters = (months - 1) // 3
    methods = ("global_convex", "band_convex", "band_switch")
    cross_validated = {
        method: np.empty_like(target, dtype=np.float64) for method in methods
    }
    fold_rows = []
    weight_rows = []
    for held_quarter in tqdm(range(4), desc="Quarter-blocked blends", unit="fold"):
        train = quarters != held_quarter
        held = quarters == held_quarter
        global_weight = fit_simplex(stack[train], target[train])
        cross_validated["global_convex"][held] = np.tensordot(
            stack[held], global_weight, axes=([-1], [0])
        )
        band_prediction = cross_validated["band_convex"]
        switch_prediction = cross_validated["band_switch"]
        for label, weight in zip(labels, global_weight, strict=True):
            weight_rows.append(
                {
                    "held_quarter": held_quarter + 1,
                    "method": "global_convex",
                    "horizon_band": "all",
                    "model": label,
                    "weight": float(weight),
                }
            )
        for band, selection in HORIZON_BANDS:
            weight = fit_simplex(stack[train, selection], target[train, selection])
            band_prediction[held, selection] = np.tensordot(
                stack[held, selection], weight, axes=([-1], [0])
            )
            train_mae = np.abs(
                stack[train, selection] - target[train, selection, None]
            ).mean(axis=(0, 1))
            selected = int(train_mae.argmin())
            switch_prediction[held, selection] = stack[held, selection, selected]
            for model_label, value in zip(labels, weight, strict=True):
                weight_rows.append(
                    {
                        "held_quarter": held_quarter + 1,
                        "method": "band_convex",
                        "horizon_band": band,
                        "model": model_label,
                        "weight": float(value),
                    }
                )
            weight_rows.append(
                {
                    "held_quarter": held_quarter + 1,
                    "method": "band_switch",
                    "horizon_band": band,
                    "model": labels[selected],
                    "weight": 1.0,
                }
            )
        for method in methods:
            fold_rows.append(
                {
                    "held_quarter": held_quarter + 1,
                    "method": method,
                    "held_origins": int(held.sum()),
                    "mae_mw": mae(cross_validated[method][held], target[held]),
                }
            )
    write_csv(output / "cross_validation_folds.csv", fold_rows)
    write_csv(output / "blend_weights.csv", weight_rows)

    best_constituent = min(single_rows, key=lambda row: row["overall_mae_mw"])
    summary_rows = []
    for method, prediction in cross_validated.items():
        row = {"method": method, "overall_mae_mw": mae(prediction, target)}
        row.update({f"{key}_mae_mw": value for key, value in band_mae(prediction, target).items()})
        row["improvement_vs_best_constituent_mw"] = (
            float(best_constituent["overall_mae_mw"]) - float(row["overall_mae_mw"])
        )
        summary_rows.append(row)
    oracle_index = np.abs(stack - target[..., None]).argmin(axis=-1)
    oracle = np.take_along_axis(stack, oracle_index[..., None], axis=-1)[..., 0]
    oracle_row = {"method": "samplewise_oracle", "overall_mae_mw": mae(oracle, target)}
    oracle_row.update({f"{key}_mae_mw": value for key, value in band_mae(oracle, target).items()})
    oracle_row["improvement_vs_best_constituent_mw"] = (
        float(best_constituent["overall_mae_mw"]) - float(oracle_row["overall_mae_mw"])
    )
    summary_rows.append(oracle_row)
    write_csv(output / "blend_summary.csv", summary_rows)

    constrained = [row for row in summary_rows if row["method"] != "samplewise_oracle"]
    best_blend = min(constrained, key=lambda row: row["overall_mae_mw"])
    band_improvements = [
        min(float(row[f"{band}_mae_mw"]) for row in single_rows)
        - float(best_blend[f"{band}_mae_mw"])
        for band, _ in HORIZON_BANDS
    ]
    go_high_cost = bool(
        float(best_blend["improvement_vs_best_constituent_mw"]) >= 1.0
        or max(band_improvements) >= 2.0
    )
    decision = {
        "split": "2023 validation",
        "seeds": SEEDS,
        "best_constituent": best_constituent,
        "best_cross_validated_blend": best_blend,
        "band_improvement_vs_best_constituent_in_each_band_mw": {
            band: value for (band, _), value in zip(HORIZON_BANDS, band_improvements, strict=True)
        },
        "go_high_cost_specialists": go_high_cost,
        "rule": "go if overall improvement >=1 MW or any band improvement >=2 MW",
    }
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(9, 4.8))
    lead = np.arange(1, 25) / 4.0
    for label, prediction in cross_validated.items():
        axis.plot(lead, np.abs(prediction - target).mean(axis=0), label=label)
    best_index = labels.index(str(best_constituent["model"]))
    axis.plot(
        lead,
        np.abs(stack[..., best_index] - target).mean(axis=0),
        color="black",
        linewidth=2,
        label=f"best constituent: {best_constituent['model']}",
    )
    axis.set(xlabel="Forecast horizon (h)", ylabel="Cluster MAE (MW)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "cross_validated_blend_by_lead.png", dpi=180)
    plt.close(figure)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
