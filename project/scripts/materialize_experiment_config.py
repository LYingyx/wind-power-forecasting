from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from wind_forcing.config import load_config


def set_nested(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use dotted.path=value: {expression}")
    dotted, raw_value = expression.split("=", 1)
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError(f"Invalid override path: {dotted}")
    target: dict[str, Any] = config
    for part in parts[:-1]:
        value = target.get(part)
        if not isinstance(value, dict):
            raise KeyError(f"Override path does not exist: {dotted}")
        target = value
    if parts[-1] not in target:
        raise KeyError(f"Override key does not exist: {dotted}")
    target[parts[-1]] = yaml.safe_load(raw_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one immutable experiment YAML")
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.base)
    config.pop("_config_path", None)
    config.pop("_project_root", None)
    for expression in args.overrides:
        set_nested(config, expression)
    config["experiment"] = {"id": args.experiment_id}

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=False)
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        if existing != rendered:
            raise FileExistsError(
                f"Experiment config already exists with different content: {output}"
            )
        print(f"[config] Reusing {output}")
        return
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(f"[config] Created {output}")


if __name__ == "__main__":
    main()
