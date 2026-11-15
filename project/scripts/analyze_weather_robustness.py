from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stratified weather-permutation robustness")
    parser.add_argument("--root", default="artifacts/paper_evidence_test_v1/weather_robustness")
    parser.add_argument("--group", action="append", default=["Normal", "KD"])
    parser.add_argument("--original-root", action="append", default=[
        "Normal=artifacts/paper_evidence_test_v1/cluster_exports/Normal",
        "KD=artifacts/paper_evidence_test_v1/cluster_exports/KD",
    ])
    parser.add_argument("--model-seeds", default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--permutations", default="1701,1702,1703,1704,1705")
    parser.add_argument("--output-dir", default="artifacts/paper_evidence_test_v1/weather_robustness_analysis")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output_dir).resolve()
    seeds = [int(x) for x in args.model_seeds.split(",") if x]
    permutations = [int(x) for x in args.permutations.split(",") if x]
    originals = {label: Path(path).resolve() for label, path in (item.split("=", 1) for item in args.original_root)}
    rows, lead_rows = [], []
    prefix = os.environ.get("WIND_PROGRESS_PREFIX", "").strip()
    position_value = os.environ.get("WIND_PROGRESS_POSITION", "").strip()
    leave = os.environ.get("WIND_PROGRESS_LEAVE", "1").lower() not in {"0", "false", "no"}
    total = len(args.group) * len(seeds) * len(permutations)
    with tqdm(total=total, desc=f"{prefix} weather analysis".strip(), unit="archive", dynamic_ncols=True, position=int(position_value) if position_value else None, leave=leave, disable=False) as progress:
        for label in args.group:
            for seed in seeds:
                with np.load(originals[label] / f"seed{seed}/predictions_{args.split}.npz") as archive:
                    original = archive["cluster_prediction_mw"].astype(np.float64)
                    target = archive["cluster_target_mw"].astype(np.float64)
                original_mae = float(np.abs(original - target).mean())
                for permutation in permutations:
                    progress.set_postfix_str(f"{label} seed={seed} perm={permutation}", refresh=False)
                    path = root / label / f"seed{seed}/perm{permutation}/predictions_{args.split}.npz"
                    with np.load(path) as archive:
                        prediction = archive["cluster_prediction_mw"].astype(np.float64)
                        np.testing.assert_allclose(target, archive["cluster_target_mw"], atol=1e-4, rtol=0.0)
                    permuted_mae = float(np.abs(prediction - target).mean())
                    rows.append({"group": label, "model_seed": seed, "permutation_seed": permutation, "original_mae_mw": original_mae, "permuted_mae_mw": permuted_mae, "degradation_mw": permuted_mae - original_mae})
                    lead_delta = np.abs(prediction - target).mean(axis=0) - np.abs(original - target).mean(axis=0)
                    for lead, value in enumerate(lead_delta, start=1):
                        lead_rows.append({"group": label, "model_seed": seed, "permutation_seed": permutation, "lead_hours": lead / 4.0, "degradation_mw": float(value)})
                    progress.update(1)
    write_csv(output / "weather_permutation_runs.csv", rows); write_csv(output / "weather_permutation_leads.csv", lead_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = args.group
    values = [[row["degradation_mw"] for row in rows if row["group"] == label] for label in labels]
    axes[0].boxplot(values, tick_labels=labels, showfliers=False); axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(ylabel="MAE degradation (MW)", title="Correct vs stratified-permuted NWP")
    for label in labels:
        selected = [row for row in lead_rows if row["group"] == label]
        matrix = np.asarray([row["degradation_mw"] for row in selected]).reshape(len(seeds) * len(permutations), 24)
        axes[1].plot(np.arange(1, 25) / 4, matrix.mean(axis=0), label=label)
        axes[1].fill_between(np.arange(1, 25) / 4, np.quantile(matrix, 0.1, axis=0), np.quantile(matrix, 0.9, axis=0), alpha=0.15)
    axes[1].axhline(0, color="black", lw=0.8); axes[1].set(xlabel="Lead time (h)", ylabel="MAE degradation (MW)", title="Weather relevance by horizon"); axes[1].legend()
    for ax in axes: ax.grid(alpha=0.25)
    output.mkdir(parents=True, exist_ok=True); fig.savefig(output / "weather_permutation_robustness.png", dpi=260, bbox_inches="tight"); fig.savefig(output / "weather_permutation_robustness.pdf", bbox_inches="tight"); plt.close(fig)
    summary = {"split": args.split, "groups": {label: {"mean_degradation_mw": float(np.mean(values[index])), "p10_mw": float(np.quantile(values[index], 0.1)), "p90_mw": float(np.quantile(values[index], 0.9))} for index, label in enumerate(labels)}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
