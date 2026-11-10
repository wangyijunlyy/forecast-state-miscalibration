import argparse
import csv
import re
from pathlib import Path


PAPER = {
    96: (0.264, 0.336),
    192: (0.290, 0.372),
    336: (0.308, 0.394),
    720: (0.314, 0.396),
}

VARIANT_ORDER = [
    "backbone_diag",
    "no_consistency",
    "no_consistency_weak_cal",
    "weak_aug_weak_cal",
    "weak_aug_no_cal",
]


def read_result(path: Path):
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def best_checkpoint(run_dir: Path):
    candidates = []
    for path in (run_dir / "ckpt").glob("epoch=*-val_CRPS=*.ckpt"):
        match = re.match(r"epoch=(\d+)-val_CRPS=([0-9.]+)\.ckpt", path.name)
        if match:
            candidates.append((float(match.group(2)), int(match.group(1))))
    return min(candidates) if candidates else ("", "")


def variant_rank(name: str):
    try:
        return VARIANT_ORDER.index(name)
    except ValueError:
        return len(VARIANT_ORDER)


def collect_rows(root: Path):
    rows = []
    for result_path in root.glob("*/*/*/horizons_results.csv"):
        variant = result_path.parents[2].name
        horizon = int(result_path.parents[1].name.replace("pred_", ""))
        result = read_result(result_path)
        best_val, best_epoch = best_checkpoint(result_path.parent)
        paper_crps, paper_nd = PAPER[horizon]
        crps = float(result["test_CRPS"])
        nd = float(result["test_ND"])
        rows.append(
            {
                "variant": variant,
                "horizon": horizon,
                "best_epoch": best_epoch,
                "best_val_CRPS": best_val,
                "test_CRPS": crps,
                "paper_CRPS": paper_crps,
                "delta_CRPS": crps - paper_crps,
                "test_ND": nd,
                "paper_NMAE": paper_nd,
                "delta_ND": nd - paper_nd,
                "beats_paper_k2vae": crps < paper_crps and nd < paper_nd,
                "source": str(result_path),
            }
        )
    rows.sort(key=lambda row: (variant_rank(row["variant"]), row["horizon"]))
    return rows


def write_csv(rows, out_path: Path):
    fields = [
        "variant",
        "horizon",
        "best_epoch",
        "best_val_CRPS",
        "test_CRPS",
        "paper_CRPS",
        "delta_CRPS",
        "test_ND",
        "paper_NMAE",
        "delta_ND",
        "beats_paper_k2vae",
        "source",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = collect_rows(args.root)
    out_path = args.out or args.root / "summary.csv"
    write_csv(rows, out_path)
    print(f"wrote {len(rows)} rows to {out_path}")
    print("variant,horizon,test_CRPS,delta_CRPS,test_ND,delta_ND,beats_paper_k2vae")
    for row in rows:
        print(
            f"{row['variant']},{row['horizon']},{row['test_CRPS']:.6f},"
            f"{row['delta_CRPS']:.6f},{row['test_ND']:.6f},"
            f"{row['delta_ND']:.6f},{row['beats_paper_k2vae']}"
        )


if __name__ == "__main__":
    main()
