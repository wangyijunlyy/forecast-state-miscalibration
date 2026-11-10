import argparse
import csv
from pathlib import Path


BASELINE_PATH = Path(__file__).with_name("paper_k2vae_ltsf_baselines.csv")
FIELDNAMES = [
    "dataset",
    "horizon",
    "paper_CRPS",
    "paper_CRPS_std",
    "dacu_test_CRPS",
    "delta_CRPS",
    "beats_paper_CRPS",
    "paper_NMAE",
    "paper_NMAE_std",
    "dacu_test_ND",
    "delta_ND",
    "beats_paper_NMAE",
    "beats_paper_k2vae",
    "source",
]


def _load_baselines(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["dataset"], row["horizon"]): row for row in rows}


def _read_first_result(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _infer_dataset_horizon(path: Path, root: Path):
    parts = path.relative_to(root).parts
    for index, part in enumerate(parts):
        if part != "dacu_k2vae":
            continue
        if len(parts) <= index + 2:
            return None, None
        dataset = parts[index + 1]
        pred_part = parts[index + 2]
        if not pred_part.startswith("pred_"):
            return None, None
        return dataset, pred_part.replace("pred_", "", 1)
    return None, None


def _iter_dacu_results(root: Path):
    for path in root.rglob("*horizons_results.csv"):
        try:
            dataset, inferred_horizon = _infer_dataset_horizon(path, root)
        except ValueError:
            continue
        if not dataset or not inferred_horizon:
            continue
        row = _read_first_result(path)
        horizon = row.get("horizon") or inferred_horizon
        yield dataset, str(horizon), row, path


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_delta(left, right):
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return ""
    return str(left_value - right_value)


def _compare_less(left, right):
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return ""
    return left_value < right_value


def _compare_less_equal(left, right):
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return ""
    return left_value <= right_value


def build_summary(root: Path, baseline_path: Path = BASELINE_PATH):
    baselines = _load_baselines(baseline_path)
    found = {}
    for dataset, horizon, row, path in _iter_dacu_results(root):
        found[(dataset, horizon)] = (row, path)

    output = []
    for dataset, horizon in sorted(
        baselines,
        key=lambda key: (key[0], int(key[1])),
    ):
        paper = baselines[(dataset, horizon)]
        dacu_row, source = found.get((dataset, horizon), ({}, ""))
        dacu_crps = dacu_row.get("test_CRPS", "")
        dacu_nd = dacu_row.get("test_ND", "")
        beats_crps = _compare_less(dacu_crps, paper["paper_CRPS"])
        beats_nmae = _compare_less_equal(dacu_nd, paper["paper_NMAE"])
        beats = beats_crps and beats_nmae if isinstance(beats_crps, bool) and isinstance(beats_nmae, bool) else ""
        output.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "paper_CRPS": paper["paper_CRPS"],
                "paper_CRPS_std": paper["paper_CRPS_std"],
                "dacu_test_CRPS": dacu_crps,
                "delta_CRPS": _format_delta(dacu_crps, paper["paper_CRPS"]),
                "beats_paper_CRPS": beats_crps,
                "paper_NMAE": paper["paper_NMAE"],
                "paper_NMAE_std": paper["paper_NMAE_std"],
                "dacu_test_ND": dacu_nd,
                "delta_ND": _format_delta(dacu_nd, paper["paper_NMAE"]),
                "beats_paper_NMAE": beats_nmae,
                "beats_paper_k2vae": beats,
                "source": str(source),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compare DACU K2VAE LTSF results with K2VAE paper Tables 9 and 10."
    )
    parser.add_argument("root", type=Path, help="Experiment root containing dacu_k2vae/<dataset>/pred_<horizon>")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args()

    rows = build_summary(args.root, args.baseline)
    out_path = args.out or (args.root / "summary_vs_paper_k2vae.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    finished = sum(1 for row in rows if row["dacu_test_CRPS"])
    wins = sum(1 for row in rows if row["beats_paper_k2vae"] is True)
    print(f"Wrote {len(rows)} rows to {out_path}; finished={finished}, wins={wins}")


if __name__ == "__main__":
    main()
