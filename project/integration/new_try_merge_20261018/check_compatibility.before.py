"""Integration checks using real cached inputs; does not train or resume queues."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys

from wind_paths import ROOT, SLAST, CROSS, FEWSHOT, select_source, read_slast_config

def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()

def slast():
    select_source("slast")
    import torch
    from wind_forcing.data import ForecastDataset
    from wind_forcing.data.dataset import ForecastCollator
    from wind_forcing.models import build_forecast_model
    dataset = ForecastDataset(ROOT / "data/model_cache_pad5_f8", "val", preload_weather=False, verbose=False)
    batch = ForecastCollator(dataset)([0, 1])
    checked = []
    for seed in range(42, 52):
        path = SLAST / f"results/SLAST_seed{seed}/best_model.pt"
        cp = torch.load(path, map_location="cpu", weights_only=False)
        config = read_slast_config(seed)
        assert cp["model_config"] == config["model"]
        assert cp["data_manifest_identity"] == dataset.manifest["preprocess_fingerprint"]
        assert config["model"]["summary_mode"] == "last"
        for split in ("val", "test"):
            metrics = json.loads((SLAST / f"analysis/{split}/SLAST_seed{seed}/metrics.json").read_text())
            assert metrics["checkpoint_sha256"] == sha(path)
        baseline = torch.load(SLAST / f"inputs/baselines/seed{seed}.pt", map_location="cpu", weights_only=False)
        for name, value in baseline["model_state"].items():
            assert torch.equal(value, cp["model_state"]["baseline_model." + name]), name
        model = build_forecast_model(cp["model_config"], dataset.static, history_steps=24, forecast_steps=24)
        model.load_state_dict(cp["model_state"], strict=True)
        model.eval()
        with torch.inference_mode():
            result = model(**batch)["station_factor"]
        assert result.shape == (2, 56, 24) and torch.isfinite(result).all()
        checked.append(seed)
    dataset.close()
    return {"seeds": checked, "strict_weight_load": True, "baseline_tensors_identical": True,
            "checkpoint_hashes_match_recorded_metrics": True, "real_input_forward_shape": [2, 56, 24]}

def transfer():
    select_source("cbh")
    sys.path.insert(0, str(ROOT / "runtime/experiments/transfer"))
    import numpy as np
    import torch
    import transfer_experiment as m
    checked = []
    for seed in range(42, 52):
        model, audit = m.build_target_model(0, seed, True)
        assert audit["copied_tensor_count"] > 0
        checked.append(seed)
    model, _ = m.build_target_model(0, 42, True)
    split = m.TargetSplit("val", 0)
    batch = split.batch(np.asarray([0, 1]), torch.device("cpu"))
    core = m.CBHCore(model)
    metadata = m.normalized_target_metadata(m.station_static(0)[0])
    protocols = []
    for protocol in ("T1", "T2", "T3", "T4"):
        adapter = m.source_adapter_path(protocol.replace("T", "F"), 16, 42) if protocol in {"T3", "T4"} else None
        predictor = m.TransferPredictor(core, protocol, 16, metadata, adapter, freeze_core=True).eval()
        with torch.inference_mode():
            result = predictor(batch)
        assert result.shape == (2, 24) and torch.isfinite(result).all()
        protocols.append(protocol)
    return {"source_seeds_loaded": checked, "protocols_real_input_forward": protocols, "shape": [2, 24]}

def fewshot():
    select_source("cbh")
    sys.path.insert(0, str(ROOT / "runtime/experiments/fewshot"))
    import numpy as np
    import torch
    import train_one as m
    from models import FAMILIES, build_model, input_group
    station = m.STATIONS[0]
    data = m.Split(station, 0, "val", torch.device("cpu"), True, np.asarray([0, 1]))
    families = []
    models = {}
    for family in FAMILIES:
        model = build_model(family, 64).eval()
        with torch.inference_mode():
            output = model(data.history, None if input_group(family) == "history_only" else data.weather, data.phase)
        assert output.shape == (2, 24) and torch.isfinite(output).all(), family
        families.append(family)
        models[family] = model
    checked = 0
    for path in sorted((FEWSHOT / "results").rglob("best_model.pt")):
        cp = torch.load(path, map_location="cpu", weights_only=False)
        family = cp["manifest"]["family"]
        models[family].load_state_dict(cp["model_state"], strict=True)
        checked += 1
    return {"families_real_input_forward": families, "checkpoints_strictly_loaded": checked, "shape": [2, 24]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["slast", "transfer", "fewshot"])
    args = parser.parse_args()
    if args.profile:
        result = {"profile": args.profile, "passed": True, "details": globals()[args.profile]()}
        print("CHECK_JSON=" + json.dumps(result))
        return
    import torch
    report = {"project_root": str(ROOT), "environment_prefix": sys.prefix,
              "python": sys.version.split()[0], "torch": torch.__version__, "checks": {}, "limitations": []}
    for profile in ("slast", "transfer", "fewshot"):
        process = subprocess.run([sys.executable, "-B", __file__, "--profile", profile],
                                 cwd=ROOT, capture_output=True, text=True)
        lines = [line[len("CHECK_JSON="):] for line in process.stdout.splitlines() if line.startswith("CHECK_JSON=")]
        report["checks"][profile] = json.loads(lines[-1]) if process.returncode == 0 and lines else {
            "passed": False, "stdout": process.stdout[-4000:], "stderr": process.stderr[-6000:]}
        print(profile, "PASS" if report["checks"][profile]["passed"] else "FAIL", flush=True)
    syntax_errors = []
    count = 0
    for directory in ("src", "scripts", "runtime", "artifacts"):
        for path in (ROOT / directory).rglob("*.py"):
            try:
                ast.parse(path.read_text(), filename=str(path))
                count += 1
            except Exception as exc:
                syntax_errors.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})
    report["checks"]["python_syntax"] = {"passed": not syntax_errors, "files": count, "errors": syntax_errors}
    if torch.cuda.is_available():
        x = torch.arange(64, device="cuda:0", dtype=torch.float32)
        assert torch.isfinite(x @ x)
        torch.cuda.synchronize()
        report["checks"]["cuda_basic_operation"] = {"passed": True, "device": torch.cuda.get_device_name(0)}
    else:
        report["limitations"].append("CUDA unavailable; CPU model checks only.")
    report["limitations"] += [
        "column_weather/E2L is results-only: source/configuration/DEM/statistics missing; seed44 lacks checkpoint/validation and seed47 is absent.",
        "Legacy experiment queues and historical scripts remain provenance records; use the project launcher for supported operations.",
        "The raw HDF5 used only to rebuild the local cache is an external backup dependency.",
        "Forward checks do not constitute full-dataset metric reproduction or completion of unfinished experiments."
    ]
    report["supported_profiles_pass"] = all(v["passed"] for v in report["checks"].values())
    report["entire_project_fully_reproducible"] = False
    (ROOT / "integration/COMPATIBILITY.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["supported_profiles_pass"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
