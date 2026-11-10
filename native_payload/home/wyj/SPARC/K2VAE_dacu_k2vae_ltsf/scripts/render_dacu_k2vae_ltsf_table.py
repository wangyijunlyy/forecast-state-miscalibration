import argparse
import csv
from pathlib import Path


FIELDNAMES = [
    "Dataset",
    "Horizon",
    "Paper CRPS",
    "DACU CRPS",
    "Delta CRPS",
    "CRPS Lower?",
    "Paper NMAE",
    "DACU ND",
    "Delta ND",
    "ND Lower?",
    "Both Lower?",
]


def _fmt(value: str) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def _paper(mean: str, std: str) -> str:
    if not mean:
        return ""
    if not std:
        return _fmt(mean)
    return f"{_fmt(mean)} +/- {_fmt(std)}"


def _yes_no(value: str) -> str:
    if value == "True":
        return "yes"
    if value == "False":
        return "no"
    return ""


def _sort_key(row):
    dataset_order = {
        "etth1": 0,
        "etth2": 1,
        "ettm1": 2,
        "ettm2": 3,
        "traffic_ltsf": 4,
        "electricity_ltsf": 5,
        "exchange_ltsf": 6,
        "weather_ltsf": 7,
        "illness_ltsf": 8,
    }
    return (dataset_order.get(row["dataset"], 99), int(row["horizon"]))


def build_markdown(rows):
    completed = [row for row in rows if row["dacu_test_CRPS"]]
    wins = [row for row in completed if row["beats_paper_k2vae"] == "True"]

    lines = [
        "# DACU K2VAE vs Paper K2VAE",
        "",
        "Lower CRPS and lower ND/NMAE are better.",
        "",
        "| " + " | ".join(FIELDNAMES) + " |",
        "| " + " | ".join(["---"] * len(FIELDNAMES)) + " |",
    ]
    for row in sorted(rows, key=_sort_key):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["horizon"],
                    _paper(row["paper_CRPS"], row["paper_CRPS_std"]),
                    _fmt(row["dacu_test_CRPS"]),
                    _fmt(row["delta_CRPS"]),
                    _yes_no(row["beats_paper_CRPS"]),
                    _paper(row["paper_NMAE"], row["paper_NMAE_std"]),
                    _fmt(row["dacu_test_ND"]),
                    _fmt(row["delta_ND"]),
                    _yes_no(row["beats_paper_NMAE"]),
                    _yes_no(row["beats_paper_k2vae"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            f"Wins: {len(wins)}/{len(completed)} completed rows",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Render DACU K2VAE paper comparison as Markdown.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with args.summary.open(newline="") as f:
        rows = list(csv.DictReader(f))

    markdown = build_markdown(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown)
    else:
        print(markdown, end="")


if __name__ == "__main__":
    main()
