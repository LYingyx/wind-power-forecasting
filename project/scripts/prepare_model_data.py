from __future__ import annotations

import argparse

from wind_forcing.config import load_config
from wind_forcing.data import ModelDataPreparer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare model-ready wind-cluster arrays")
    parser.add_argument(
        "--config", default="configs/forced_evolution_pad5.yaml", help="Experiment YAML"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    ModelDataPreparer(config["data"]).build()


if __name__ == "__main__":
    main()
