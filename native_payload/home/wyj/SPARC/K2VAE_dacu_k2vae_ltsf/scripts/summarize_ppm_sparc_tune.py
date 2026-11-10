#!/usr/bin/env python3
"""Summarize SPARC-PPM tuning runs and mark configs beating same-run PPM."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "tag",
    "method",
    "horizon",
    "CRPS",
    "QICE",
    "ND",
    "NRMSE",
    "ppm_CRPS",
    "ppm_QICE",
    "crps_delta",
    "qice_delta",
    "beats_ppm_crps_qice",
    "path",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("*/pred_*/comparison*_pred_*.csv")):
        tag = path.parts[-3]
        with path.open(newline="") as f:
            comparison_rows = list(csv.DictReader(f))
        ppm_rows = [row for row in comparison_rows if row["method"] == "PPM"]
        if not ppm_rows:
            continue
        ppm = ppm_rows[0]
        for row in comparison_rows:
            if row["method"] == "PPM":
                continue
            crps = float(row["CRPS"])
            qice = float(row["QICE"])
            ppm_crps = float(ppm["CRPS"])
            ppm_qice = float(ppm["QICE"])
            out = {
                "tag": tag,
                "method": row["method"],
                "horizon": row["horizon"],
                "CRPS": row["CRPS"],
                "QICE": row["QICE"],
                "ND": row["ND"],
                "NRMSE": row["NRMSE"],
                "ppm_CRPS": ppm["CRPS"],
                "ppm_QICE": ppm["QICE"],
                "crps_delta": crps - ppm_crps,
                "qice_delta": qice - ppm_qice,
                "beats_ppm_crps_qice": crps < ppm_crps and qice < ppm_qice,
                "path": str(path),
            }
            rows.append(out)
    rows.sort(key=lambda r: (int(r["horizon"]), not bool(r["beats_ppm_crps_qice"]), float(r["CRPS"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
