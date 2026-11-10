#!/usr/bin/env python3
"""Summarize ETTh1 PPM vs SPARC-PPM comparison files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = ["method", "horizon", "CRPS", "QICE", "ND", "NRMSE", "CRPS-Sum", "QICE-Sum", "ND-Sum", "NRMSE-Sum", "path"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("pred_*/comparison_pred_*.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["path"] = str(path)
                rows.append(row)
    rows.sort(key=lambda r: (int(r["horizon"]), r["method"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
