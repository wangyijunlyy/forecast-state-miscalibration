import argparse
import csv
from pathlib import Path


# K2VAE paper Tables 9 and 10, ETTh1-L rows.
PAPER_BASELINES = {
    "csdi": {
        96: {"CRPS": 0.437, "NMAE": 0.557},
        192: {"CRPS": 0.496, "NMAE": 0.625},
        336: {"CRPS": 0.454, "NMAE": 0.574},
        720: {"CRPS": 0.528, "NMAE": 0.657},
    },
    "timegrad": {
        96: {"CRPS": 0.455, "NMAE": 0.585},
        192: {"CRPS": 0.516, "NMAE": 0.680},
        336: {"CRPS": 0.512, "NMAE": 0.666},
        720: {"CRPS": 0.523, "NMAE": 0.672},
    },
    "patchtst": {
        96: {"CRPS": 0.312, "NMAE": 0.407},
        192: {"CRPS": 0.313, "NMAE": 0.405},
        336: {"CRPS": 0.319, "NMAE": 0.412},
        720: {"CRPS": 0.323, "NMAE": 0.428},
    },
    "itransformer": {
        96: {"CRPS": 0.292, "NMAE": 0.386},
        192: {"CRPS": 0.298, "NMAE": 0.388},
        336: {"CRPS": 0.327, "NMAE": 0.415},
        720: {"CRPS": 0.350, "NMAE": 0.449},
    },
}


def _read_first_result(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def _infer_model_horizon(path: Path, root: Path):
    parts = path.relative_to(root).parts
    if len(parts) < 4:
        return None, None

    if parts[0] == "dacu" and parts[2].startswith("pred_"):
        return parts[1], int(parts[2].replace("pred_", "", 1))

    if parts[0] in PAPER_BASELINES and parts[1].startswith("pred_"):
        return parts[0], int(parts[1].replace("pred_", "", 1))

    return None, None


def iter_dacu_results(root: Path):
    for path in root.rglob("*horizons_results.csv"):
        try:
            model, inferred_horizon = _infer_model_horizon(path, root)
        except (ValueError, IndexError):
            continue

        if model not in PAPER_BASELINES or inferred_horizon not in PAPER_BASELINES[model]:
            continue

        row = _read_first_result(path)
        horizon = int(row.get("horizon") or inferred_horizon)
        yield model, horizon, row, path


def main():
    parser = argparse.ArgumentParser(
        description="Compare ETTh1-L DACU results with K2VAE paper baseline means."
    )
    parser.add_argument("root", type=Path, help="Experiment root containing dacu/<model>/pred_<horizon>")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    found = {}
    for model, horizon, row, path in iter_dacu_results(args.root):
        found[(model, horizon)] = (row, path)

    rows = []
    for model in ["csdi", "timegrad", "patchtst", "itransformer"]:
        for horizon in [96, 192, 336, 720]:
            paper = PAPER_BASELINES[model][horizon]
            dacu_row, path = found.get((model, horizon), ({}, ""))
            out = {
                "model": model,
                "horizon": horizon,
                "paper_CRPS": paper["CRPS"],
                "dacu_test_CRPS": dacu_row.get("test_CRPS", ""),
                "delta_CRPS": "",
                "beats_paper_CRPS": "",
                "paper_NMAE": paper["NMAE"],
                "dacu_test_ND": dacu_row.get("test_ND", ""),
                "delta_ND": "",
                "beats_paper_NMAE": "",
                "beats_paper_baseline": "",
                "source": str(path),
            }
            try:
                dacu_crps = float(out["dacu_test_CRPS"])
                dacu_nd = float(out["dacu_test_ND"])
                out["delta_CRPS"] = dacu_crps - paper["CRPS"]
                out["delta_ND"] = dacu_nd - paper["NMAE"]
                out["beats_paper_CRPS"] = dacu_crps < paper["CRPS"]
                out["beats_paper_NMAE"] = dacu_nd <= paper["NMAE"]
                out["beats_paper_baseline"] = out["beats_paper_CRPS"] and out["beats_paper_NMAE"]
            except (TypeError, ValueError):
                pass
            rows.append(out)

    fieldnames = [
        "model",
        "horizon",
        "paper_CRPS",
        "dacu_test_CRPS",
        "delta_CRPS",
        "beats_paper_CRPS",
        "paper_NMAE",
        "dacu_test_ND",
        "delta_ND",
        "beats_paper_NMAE",
        "beats_paper_baseline",
        "source",
    ]
    out_path = args.out or (args.root / "summary_vs_paper_baselines.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    finished = sum(1 for row in rows if row["dacu_test_CRPS"])
    wins = sum(1 for row in rows if row["beats_paper_baseline"] is True)
    print(f"Wrote {len(rows)} reference rows to {out_path}; finished={finished}, wins={wins}")


if __name__ == "__main__":
    main()
