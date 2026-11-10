import argparse
import csv
from pathlib import Path


SELECTED = [
    "test_CRPS",
    "test_ND",
    "test_ND-Sum",
    "test_NRMSE",
    "test_MSE",
    "test_MAE_Coverage",
    "test_IntervalWidth[0.9]",
    "test_ErrorUncertaintyCorr",
    "test_AUSE",
]

PREFERRED_VARIANTS = [
    "baseline",
    "fast_posthoc",
    "fast_posthoc_v2",
    "fast_posthoc_v2_scale_only",
    "fast_posthoc_v2_crps_strong",
    "affine",
    "fast_joint",
]


def iter_rows(root: Path):
    for path in root.rglob("*horizons_results.csv"):
        parts = path.relative_to(root).parts
        if len(parts) < 3:
            continue
        variant = parts[0]
        horizon = parts[1].replace("pred_", "", 1) if parts[1].startswith("pred_") else ""
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                yield {
                    "variant": variant,
                    "horizon": row.get("horizon") or horizon,
                    "source": str(path),
                    **{metric: row.get(metric, "") for metric in SELECTED},
                }


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Summarize fast probabilistic DACU K2VAE comparison runs.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = list(iter_rows(args.root))
    by_key = {(row["variant"], row["horizon"]): row for row in rows}
    horizons = sorted({row["horizon"] for row in rows}, key=lambda value: int(value) if value.isdigit() else value)
    discovered_variants = sorted({row["variant"] for row in rows})
    preferred = [variant for variant in PREFERRED_VARIANTS if variant in discovered_variants]
    variants = preferred + [variant for variant in discovered_variants if variant not in preferred]

    out_rows = []
    for horizon in horizons:
        baseline = by_key.get(("baseline", horizon), {})
        for variant in variants:
            row = by_key.get((variant, horizon))
            if row is None:
                continue
            out = {
                "horizon": horizon,
                "variant": variant,
                "source": row["source"],
            }
            for metric in SELECTED:
                value = row.get(metric, "")
                base = baseline.get(metric, "")
                out[metric] = value
                value_f = to_float(value)
                base_f = to_float(base)
                out[f"delta_vs_baseline_{metric}"] = "" if value_f is None or base_f is None else value_f - base_f
            crps = to_float(row.get("test_CRPS"))
            base_crps = to_float(baseline.get("test_CRPS"))
            nd = to_float(row.get("test_ND"))
            base_nd = to_float(baseline.get("test_ND"))
            out["beats_crps"] = "" if crps is None or base_crps is None else crps < base_crps
            out["beats_nd"] = "" if nd is None or base_nd is None else nd <= base_nd
            out_rows.append(out)

    out_path = args.out or args.root / "summary.csv"
    fieldnames = ["horizon", "variant", "beats_crps", "beats_nd", "source"]
    for metric in SELECTED:
        fieldnames.extend([metric, f"delta_vs_baseline_{metric}"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
