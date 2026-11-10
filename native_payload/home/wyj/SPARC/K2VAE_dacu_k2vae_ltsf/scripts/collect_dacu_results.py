import argparse
import csv
from pathlib import Path


SELECTED = [
    "test_CRPS",
    "test_ND",
    "test_ND-Sum",
    "test_NRMSE",
    "test_MASE",
    "test_MAE_Coverage",
    "test_IntervalWidth[0.9]",
    "test_ErrorUncertaintyCorr",
    "test_AUSE",
]


def iter_metric_files(root: Path):
    yield from root.rglob("*horizons_results.csv")


def infer_experiment(path: Path, root: Path):
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts

    variant = ""
    model = ""
    horizon = ""
    if len(parts) >= 4 and parts[0] in {"baseline", "dacu"}:
        variant = parts[0]
        model = parts[1]
        if parts[2].startswith("pred_"):
            horizon = parts[2].replace("pred_", "", 1)

    return variant, model, horizon


def main():
    parser = argparse.ArgumentParser(description="Collect DACU experiment CSV metrics.")
    parser.add_argument("root", type=Path, help="Experiment root, e.g. ./results_dacu")
    parser.add_argument("--out", type=Path, default=Path("dacu_summary.csv"))
    args = parser.parse_args()

    rows = []
    for path in iter_metric_files(args.root):
        try:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    variant, model, inferred_horizon = infer_experiment(path, args.root)
                    selected = {k: row.get(k, "") for k in SELECTED}
                    if any(selected.values()):
                        selected["variant"] = variant
                        selected["model"] = model
                        selected["horizon"] = row.get("horizon", "") or inferred_horizon
                        selected["source"] = str(path)
                        rows.append(selected)
        except UnicodeDecodeError:
            continue

    paired = {}
    for row in rows:
        key = (row["model"], row["horizon"])
        if not key[0] or not key[1] or row["variant"] not in {"baseline", "dacu"}:
            continue
        paired.setdefault(key, {})[row["variant"]] = row

    output_rows = []
    for (model, horizon), variants in sorted(paired.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        baseline = variants.get("baseline", {})
        dacu = variants.get("dacu", {})
        out = {
            "model": model,
            "horizon": horizon,
            "baseline_source": baseline.get("source", ""),
            "dacu_source": dacu.get("source", ""),
        }
        for metric in SELECTED:
            base_value = baseline.get(metric, "")
            dacu_value = dacu.get(metric, "")
            out[f"baseline_{metric}"] = base_value
            out[f"dacu_{metric}"] = dacu_value
            try:
                out[f"delta_{metric}"] = float(dacu_value) - float(base_value)
            except (TypeError, ValueError):
                out[f"delta_{metric}"] = ""
        try:
            beats_crps = float(dacu["test_CRPS"]) < float(baseline["test_CRPS"])
            beats_nd = float(dacu["test_ND"]) <= float(baseline["test_ND"])
            out["beats_baseline"] = beats_crps and beats_nd
            out["beats_crps"] = beats_crps
            out["beats_nd"] = beats_nd
        except (KeyError, TypeError, ValueError):
            out["beats_baseline"] = ""
            out["beats_crps"] = ""
            out["beats_nd"] = ""
        output_rows.append(out)

    fieldnames = [
        "model",
        "horizon",
        "beats_baseline",
        "beats_crps",
        "beats_nd",
        "baseline_source",
        "dacu_source",
    ]
    for metric in SELECTED:
        fieldnames.extend([f"baseline_{metric}", f"dacu_{metric}", f"delta_{metric}"])

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} paired rows to {args.out}")


if __name__ == "__main__":
    main()
