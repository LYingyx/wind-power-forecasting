from __future__ import annotations

import argparse
import sys

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one explicit tqdm bar from newline-delimited job completions"
    )
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--desc", required=True)
    parser.add_argument("--unit", default="job")
    parser.add_argument("--position", type=int, default=0)
    args = parser.parse_args()
    if args.total < 0:
        raise ValueError("--total must be non-negative")
    with tqdm(
        total=args.total,
        desc=args.desc,
        unit=args.unit,
        dynamic_ncols=True,
        position=args.position,
        leave=True,
        mininterval=0.1,
        disable=False,
    ) as progress:
        for line in sys.stdin:
            label = line.strip()
            if not label:
                continue
            progress.set_postfix_str(label, refresh=False)
            progress.update(1)


if __name__ == "__main__":
    main()
