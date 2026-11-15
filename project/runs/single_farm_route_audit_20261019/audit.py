"""Read-only audit of archived metrics and protocol-matched route comparisons.

Run: .runtime/chronos/bin/python runs/single_farm_route_audit_20261019/audit.py
Outputs are confined to this directory. No training or checkpoint changes.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ART = ROOT / "artifacts"
PREV = ROOT / "runs/single_farm_review_20261018"
CROSS = ART / "cross_farm_shared_specific_20260920_v1"


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)


def paired(values, comparison, split="test", budget=""):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    rng = np.random.default_rng(20261019)
    ci = np.quantile(values[rng.integers(len(values), size=(20000, len(values)))].mean(1), [.025, .975])
    return dict(comparison=comparison, split=split, budget=budget,
                delta=values.mean(), ci_low=ci[0], ci_high=ci[1],
                wins=int((values < 0).sum()), farms=len(values),
                unit="farm after averaging matched seeds and season blocks")


def main():
    conn = sqlite3.connect(f"file:{ROOT / 'results/CATALOG.sqlite'}?mode=ro", uri=True)
    obs = pd.read_sql_query("select * from observations", conn)
    conn.close()
    obs["seed"] = pd.to_numeric(obs.seed)
    duplicate = int(obs.duplicated(["experiment", "run_id", "split"]).sum())
    assert duplicate == 0
    assert np.isfinite(obs.mae).all() and (obs.mae >= 0).all()
    index = pd.read_csv(ROOT / "results/EXPERIMENT_INDEX.csv").fillna("")
    counts = obs.groupby("experiment").agg(records=("mae", "size"), runs=("run_id", "nunique"),
                                              models=("model", "nunique"))
    inventory = index.merge(counts, left_on="experiment", right_index=True, how="left").fillna(0)
    assert (inventory.run_split_records == inventory.records).all()
    save(inventory, "experiment_inventory")

    # Check every indexed scalar against its primary metric JSON/CSV, not a report.
    groups = defaultdict(list)
    for row in obs.itertuples(index=False):
        groups[row.metric_file].append(row)
    failures = []
    comparisons = 0
    max_difference = 0.0
    for i, (rel, rows) in enumerate(groups.items(), 1):
        path = ROOT / rel
        try:
            if path.suffix == ".csv":
                frame = pd.read_csv(path, dtype={"station_id": str})
                assert not frame.duplicated(["family", "station_id", "seed", "split"]).any()
                lookup = frame.set_index(["family", "station_id", "seed", "split"])
                payloads = [lookup.loc[(r.model, str(r.station), int(r.seed), r.split)].to_dict() for r in rows]
            else:
                data = json.loads(path.read_text())
                payloads = [data.get(r.split, data) for r in rows]
            for row, data in zip(rows, payloads):
                for name in ("mae", "rmse", "r2"):
                    expected = getattr(row, name)
                    key = getattr(row, name + "_key")
                    if pd.isna(expected) or not key:
                        continue
                    metrics = data if key in data else data["metrics"]
                    actual = float(metrics[key])
                    difference = abs(actual - float(expected))
                    if not np.isfinite(actual) or difference > 1e-10:
                        failures.append(dict(path=rel, split=row.split, metric=name,
                                             actual=actual, indexed=float(expected)))
                    max_difference = max(max_difference, difference)
                    comparisons += 1
        except (OSError, KeyError, ValueError, AssertionError, TypeError) as exc:
            failures.append(dict(path=rel, error=repr(exc)))
        if i % 10000 == 0:
            print(f"Primary metric files checked: {i}/{len(groups)}", flush=True)
    print(f"Primary audit finished: {len(groups)} files, {len(failures)} failures", flush=True)
    (OUT / "metric_audit_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    assert not failures, "Inspect metric_audit_failures.json before using results"

    ids = np.load(ROOT / "data/model_cache_pad5_f8/static/station_id.npy").astype(str)
    two = pd.read_csv(CROSS / "analysis/two_by_two_station_seed.csv")
    assert len(two) == 4 * 10 * 2 * 56
    assert not two.duplicated(["cell", "seed", "split", "site_index"]).any()
    two["station"] = ids[two.site_index.astype(int)]
    two["model"] = "CBH_" + two.cell
    source = [two[["model", "split", "seed", "station", "mae"]]]
    rows = []
    for r in obs[obs.experiment.eq("shared_all_station_baselines_20260922_v1")].itertuples():
        d = json.loads((ROOT / r.metric_file).read_text())[r.split]
        assert list(map(str, d["station_ids"])) == list(ids)
        for station, value in zip(ids, d["per_station_mae"]):
            rows.append(dict(model=r.model, split=r.split, seed=r.seed, station=station, mae=value))
    for r in obs[obs.experiment.eq("column_weather_20260928_v1")].itertuples():
        d = json.loads((ROOT / r.metric_file).read_text())
        assert len(d["per_station_mae_cf"]) == len(ids)
        for station, value in zip(ids, d["per_station_mae_cf"]):
            rows.append(dict(model=r.model, split=r.split, seed=r.seed, station=station, mae=value))
    source.append(pd.DataFrame(rows))
    source = pd.concat(source, ignore_index=True)
    assert not source.duplicated(["model", "split", "seed", "station"]).any()
    assert source.groupby(["model", "split"]).size().eq(560).all()
    farm = source.groupby(["station", "split", "model"]).mae.mean().unstack()
    previous = pd.read_csv(PREV / "tables/source_all56_per_farm.csv", dtype={"station": str}).set_index(["station", "split"])
    common = farm.columns.intersection(previous.columns)
    source_previous_difference = float((farm[common] - previous[common]).abs().max().max())
    assert source_previous_difference < 1e-8
    source_summary = source.groupby(["model", "split", "seed"]).mae.mean().groupby(["model", "split"]).agg(["mean", "std"]).reset_index()
    save(source_summary, "source_model_summary")
    meta = pd.read_csv(CROSS / "data/farm_quality_audit.csv", dtype={"station_id": str})
    meta = meta.rename(columns={"station_id": "station", "name": "station_name"}).set_index("station")
    all56 = farm.reset_index().merge(meta[["station_name", "capacity_mw"]], left_on="station", right_index=True)
    all56["shared_vs_independent_gain_pct"] = 100 * (all56.CBH_IL - all56.CBH_SL) / all56.CBH_IL
    all56["local_cbh_vs_transformer_gain_pct"] = 100 * (all56.TRANSFORMER_FC - all56.CBH_SL) / all56.TRANSFORMER_FC
    save(all56, "source_all56")
    selection = json.loads((PREV / "farm_selection.json").read_text())
    chosen = selection["source56"]["selected"]
    assert len(chosen) == 6 and not selection["source56"]["uses_prediction_errors"]
    representatives = all56[all56.split.eq("test") & all56.station.isin(chosen + ["1043056"])].copy()
    representatives["role"] = np.where(representatives.station.eq("1043056"), "post-hoc failure diagnostic", "training-descriptor medoid")
    save(representatives, "representative_source_farms")
    contrasts = []
    for split in ["val", "test"]:
        w = farm.xs(split, level="split")
        for left, right in [("CBH_SL", "CBH_IL"), ("CBH_SC", "CBH_SL"), ("CBH_SL", "TRANSFORMER_FC"), ("E2C", "E0"), ("E2C", "TRANSFORMER_FC")]:
            contrasts.append(paired(w[left] - w[right], f"{left}-{right}", split))

    # Freeze scratch family using dev/H1, then compare the same five seeds on final/H2.
    scratch_exp = "selected30_fewshot_baselines_20260920_v1"
    dev = obs[obs.experiment.eq(scratch_exp) & obs.cohort.eq("dev") & obs.stage.eq("target_main") & obs.split.eq("val")]
    devmeans = dev.groupby(["budget", "model", "station"]).mae.mean().groupby(["budget", "model"]).mean()
    frozen = json.loads((ART / scratch_exp / "analysis/dev_family_selection_pre_final.json").read_text())
    selected = {budget: data["selected_family"] for budget, data in frozen["budgets"].items()}
    for budget, model in selected.items():
        assert devmeans.loc[budget].idxmin() == model
    target = obs[obs.cohort.eq("final") & obs.stage.eq("target_main") & obs.split.eq("test") & obs.seed.between(42, 46)].copy()
    mappings = {
        "E1_pretrained_TRANSFORMER_FC_all_parameter_finetune": "Pretrained_Transformer",
        "final_E2C_T5_full_correction_adaptation": "E2C_T5",
        "T5_r16": "CBH_T5",
    }
    arms = target[target.model.isin(mappings)].copy()
    arms["model"] = arms.model.map(mappings)
    scratch = target[target.experiment.eq(scratch_exp) & target.apply(lambda r: selected.get(r.budget) == r.model, axis=1)].copy()
    scratch["selected_family"] = scratch.model
    scratch["model"] = "Scratch_dev_selected"
    target = pd.concat([arms, scratch], ignore_index=True)
    for model, group in target.groupby("model"):
        assert len(group) == 1400 and group.station.nunique() == 20
    assert not target.duplicated(["model", "station", "seed", "budget", "season"]).any()
    target_farm = target.groupby(["station", "budget", "model"]).mae.mean().unstack()
    target_summary = target_farm.groupby("budget").mean().reindex(["7d", "30d", "90d", "180d", "full"])
    target_summary["selected_scratch_family"] = pd.Series(selected)
    target_summary["e2c_gain_vs_scratch_pct"] = 100 * (1 - target_summary.E2C_T5 / target_summary.Scratch_dev_selected)
    save(target_summary.reset_index(), "target_budget_comparison")
    save(target_farm.reset_index().merge(meta[["station_name", "capacity_mw"]], left_on="station", right_index=True), "target_all20")
    for budget, group in target.groupby("budget"):
        w = group.pivot(index=["station", "seed", "season"], columns="model", values="mae")
        assert not w.isna().any().any()
        for left, right in [("E2C_T5", "Pretrained_Transformer"), ("E2C_T5", "Scratch_dev_selected"), ("Pretrained_Transformer", "Scratch_dev_selected"), ("E2C_T5", "CBH_T5")]:
            contrasts.append(paired((w[left] - w[right]).groupby("station").mean(), f"{left}-{right}", budget=budget))
    save(pd.DataFrame(contrasts), "paired_comparisons")
    audit = dict(observations=len(obs), unique_experiment_runs=len(obs[["experiment", "run_id"]].drop_duplicates()),
                 indexed_experiment_directories=len(index), metric_files=len(groups),
                 scalar_comparisons=comparisons, metric_failures=len(failures),
                 max_scalar_difference=max_difference, duplicate_run_split_keys=duplicate,
                 source_station_seed_rows=len(source), source_farms=len(ids),
                 source_vs_previous_max_difference=source_previous_difference,
                 matched_target_rows=len(target), target_farms=target.station.nunique(),
                 source_representatives=chosen, target_representatives=selection["target20"]["selected"],
                 bootstrap="20,000 farm resamples; RNG seed 20261019; temporal uncertainty not resampled here",
                 scope="All indexed scalar files checked; main tables recomputed from raw metrics. No new training or full inference. Earlier prediction-level audit retained separately.")
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    provenance = [ROOT / "results/CATALOG.sqlite", CROSS / "analysis/two_by_two_station_seed.csv", PREV / "farm_selection.json", PREV / "prediction_verification.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance}
    (OUT / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(source_summary[source_summary.model.isin(["CBH_SL", "CBH_IL", "CBH_SC", "E0", "E2C", "TRANSFORMER_FC"])].to_string(index=False))
    print(target_summary.to_string())


if __name__ == "__main__":
    main()
