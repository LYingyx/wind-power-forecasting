"""Verified entry points for preserved, experiment-specific source versions."""
from pathlib import Path
import argparse
import importlib
import json
import os
import subprocess
import sys

from wind_paths import ROOT, SLAST, select_source, source, read_slast_config, new_output

def predict(args):
    select_source("slast")
    import numpy as np
    import torch
    from wind_forcing.data import ForecastDataset
    from wind_forcing.data.dataset import ForecastCollator
    from wind_forcing.models import build_forecast_model
    config = read_slast_config(args.seed)
    checkpoint = torch.load(SLAST / f"results/SLAST_seed{args.seed}/best_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["model_config"] != config["model"]:
        raise ValueError("Checkpoint/configuration mismatch")
    dataset = ForecastDataset(config["data"]["cache_dir"], args.split, preload_weather=False, verbose=False)
    try:
        if checkpoint["data_manifest_identity"] != dataset.manifest["preprocess_fingerprint"]:
            raise ValueError("Checkpoint/data fingerprint mismatch")
        model = build_forecast_model(checkpoint["model_config"], dataset.static, history_steps=24, forecast_steps=24)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(args.device).eval()
        ids = list(range(args.start, min(args.start + args.samples, len(dataset))))
        if not ids:
            raise ValueError("Requested sample range is empty")
        batch = ForecastCollator(dataset)(ids)
        with torch.inference_mode():
            output = model(**{k: v.to(args.device) for k, v in batch.items()})
        values = {k: v.float().cpu().numpy() for k, v in output.items() if isinstance(v, torch.Tensor)}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError("Non-finite model output")
        if args.output:
            path = new_output(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.savez_compressed(stream, **values, origin_unix=batch["origin_unix"].numpy())
        print(json.dumps({"seed": args.seed, "split": args.split, "samples": len(ids),
                          "station_factor_shape": list(values["station_factor"].shape), "device": args.device,
                          "output": args.output, "scope": "Inference only; no new full-dataset metrics."}))
    finally:
        dataset.close()

def train_slast(args):
    import yaml
    output = new_output(args.output)
    output.mkdir(parents=True)
    config = read_slast_config(args.seed)
    config_path = output / "runtime_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, "-B", str(SLAST / "snapshot/scripts/train_forced_evolution.py"),
               "--config", str(config_path), "--history-checkpoint", str(SLAST / f"inputs/baselines/seed{args.seed}.pt"),
               "--output-dir", str(output), "--device", args.device]
    if args.epochs is not None:
        command += ["--epochs", str(args.epochs)]
    environment = dict(os.environ, PYTHONPATH=str(source("slast")))
    subprocess.run(command, env=environment, cwd=ROOT, check=True)

def experimental(profile, argv):
    select_source("cbh")
    code = ROOT / "runtime/experiments" / profile
    sys.path.insert(0, str(code))
    if profile == "fewshot":
        module = importlib.import_module("train_one")
        original = module.output_dir
        base = module.EXP
        def output(args):
            return ROOT / "runs/fewshot" / original(args).relative_to(base)
        module.output_dir = output
    else:
        module = importlib.import_module("transfer_experiment")
        original = module.output_directory
        base = module.EXP
        def output(args, station):
            return ROOT / "runs/transfer" / original(args, station).relative_to(base)
        module.output_directory = output
    if "--device" not in argv and not any(x.startswith("--device=") for x in argv):
        argv += ["--device", "cpu"]
    sys.argv = [profile, *argv]
    args = module.parse_args()
    destination = module.output_dir(args) if profile == "fewshot" else module.output_directory(args, args.station_id)
    if not destination.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Output must stay within this project's runs directory")
    new_output(destination)
    if profile == "transfer" and args.protocol in {"T3", "T4"}:
        module.source_adapter_path(args.protocol.replace("T", "F"), args.rank, args.seed)
    module.run(args)

def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"transfer", "fewshot"}:
        return experimental(sys.argv[1], sys.argv[2:])
    parser = argparse.ArgumentParser(description="Merged wind forecasting project. Archived queues are not resumed.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("slast-predict", "slast-train"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, choices=range(42, 52), default=42)
        p.add_argument("--device", default="cpu")
        p.add_argument("--output", required=name == "slast-train")
        if name == "slast-predict":
            p.add_argument("--split", choices=["train", "val", "test"], default="val")
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--samples", type=int, default=2)
        else:
            p.add_argument("--epochs", type=int)
    sub.add_parser("check")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "slast-predict":
        if args.start < 0 or args.samples < 1:
            parser.error("start must be nonnegative and samples positive")
        predict(args)
    elif args.command == "slast-train":
        train_slast(args)
    elif args.command == "check":
        subprocess.run([sys.executable, "-B", str(ROOT / "runtime/check_compatibility.py")], check=True, cwd=ROOT)
    else:
        print((ROOT / "integration/COMPATIBILITY.json").read_text())

if __name__ == "__main__":
    main()
