from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect completed experiment metrics")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows: list[dict[str, Any]] = []
    for complete in sorted(root.glob("**/COMPLETE")):
        directory = complete.parent
        if directory.name == "selected":
            continue
        val_path = directory / "metrics_val.json"
        test_path = directory / "metrics_test.json"
        if not val_path.is_file() and not test_path.is_file():
            continue
        for split, path in (("val", val_path), ("test", test_path)):
            if not path.is_file():
                continue
            metrics = load_json(path)
            row: dict[str, Any] = {
                "experiment": str(directory.relative_to(root)),
                "split": split,
            }
            for key, value in metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    row[key] = value
            rows.append(row)
    rows.sort(key=lambda item: (item["experiment"], item["split"]))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_temp = output / f".summary.json.tmp.{os.getpid()}"
    with json_temp.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=True, indent=2)
    os.replace(json_temp, output / "summary.json")

    fields = sorted({key for row in rows for key in row})
    csv_temp = output / f".summary.csv.tmp.{os.getpid()}"
    with csv_temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temp, output / "summary.csv")
    print(f"[summary] rows={len(rows)}, output={output}")


if __name__ == "__main__":
    main()
