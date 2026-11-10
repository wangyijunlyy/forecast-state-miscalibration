#!/usr/bin/env python3
"""Create compact comparison tables from K2VAE-SPARC+PIT long-form results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHODS = ["K2VAE", "K2VAE-SPARC", "K2VAE-SPARC+PIT"]
METRICS = ["CRPS", "QICE", "ND", "NRMSE", "MSE"]
LOWER_IS_BETTER = set(METRICS)


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def signed_delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def relative_delta(candidate: float | None, baseline: float | None) -> float | None:
    delta = signed_delta(candidate, baseline)
    if delta is None or baseline in (None, 0):
        return None
    return 100.0 * delta / abs(baseline)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def best_method(values: dict[str, float | None], metric: str) -> str:
    candidates = [(method, value) for method, value in values.items() if value is not None]
    if not candidates:
        return ""
    reverse = metric not in LOWER_IS_BETTER
    return sorted(candidates, key=lambda item: item[1], reverse=reverse)[0][0]


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = row.get("dataset", "")
            horizon = row.get("horizon", "")
            method = row.get("method", "")
            if dataset and horizon and method:
                grouped[(dataset, horizon)][method] = row
    return dict(grouped)


def write_wide(grouped: dict[tuple[str, str], dict[str, dict[str, str]]], out_path: Path) -> None:
    fields = ["dataset", "horizon", "completed"]
    for metric in METRICS:
        fields.extend(f"{metric}_{method}" for method in METHODS)
        fields.extend(
            [
                f"{metric}_pit_minus_k2vae",
                f"{metric}_pit_minus_sparc",
                f"{metric}_pit_rel_vs_k2vae_pct",
                f"{metric}_pit_rel_vs_sparc_pct",
                f"best_{metric}_method",
            ]
        )

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, horizon in sorted(grouped, key=lambda item: (item[0], int(item[1]))):
            methods = grouped[(dataset, horizon)]
            row = {
                "dataset": dataset,
                "horizon": horizon,
                "completed": str(all(method in methods for method in METHODS)).lower(),
            }
            for metric in METRICS:
                values = {method: as_float(methods.get(method, {}).get(metric)) for method in METHODS}
                for method in METHODS:
                    row[f"{metric}_{method}"] = fmt(values[method])
                pit = values["K2VAE-SPARC+PIT"]
                k2vae = values["K2VAE"]
                sparc = values["K2VAE-SPARC"]
                row[f"{metric}_pit_minus_k2vae"] = fmt(signed_delta(pit, k2vae))
                row[f"{metric}_pit_minus_sparc"] = fmt(signed_delta(pit, sparc))
                row[f"{metric}_pit_rel_vs_k2vae_pct"] = fmt(relative_delta(pit, k2vae))
                row[f"{metric}_pit_rel_vs_sparc_pct"] = fmt(relative_delta(pit, sparc))
                row[f"best_{metric}_method"] = best_method(values, metric)
            writer.writerow(row)


def summarize_overall(
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]], out_path: Path
) -> None:
    fields = [
        "metric",
        "baseline",
        "n",
        "mean_baseline",
        "mean_pit",
        "mean_delta_pit_minus_baseline",
        "mean_relative_delta_pct",
        "pit_wins",
        "ties",
        "pit_losses",
    ]
    rows = []
    for metric in METRICS:
        for baseline_name in ["K2VAE", "K2VAE-SPARC"]:
            baseline_values = []
            pit_values = []
            deltas = []
            rels = []
            wins = ties = losses = 0
            for methods in grouped.values():
                baseline = as_float(methods.get(baseline_name, {}).get(metric))
                pit = as_float(methods.get("K2VAE-SPARC+PIT", {}).get(metric))
                delta = signed_delta(pit, baseline)
                rel = relative_delta(pit, baseline)
                if baseline is None or pit is None or delta is None:
                    continue
                baseline_values.append(baseline)
                pit_values.append(pit)
                deltas.append(delta)
                if rel is not None:
                    rels.append(rel)
                if abs(delta) <= 1e-12:
                    ties += 1
                elif (metric in LOWER_IS_BETTER and delta < 0) or (metric not in LOWER_IS_BETTER and delta > 0):
                    wins += 1
                else:
                    losses += 1
            n = len(deltas)
            rows.append(
                {
                    "metric": metric,
                    "baseline": baseline_name,
                    "n": n,
                    "mean_baseline": fmt(sum(baseline_values) / n if n else None),
                    "mean_pit": fmt(sum(pit_values) / n if n else None),
                    "mean_delta_pit_minus_baseline": fmt(sum(deltas) / n if n else None),
                    "mean_relative_delta_pct": fmt(sum(rels) / len(rels) if rels else None),
                    "pit_wins": wins,
                    "ties": ties,
                    "pit_losses": losses,
                }
            )

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary_ppm_style_sparc_pit.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregate outputs. Defaults to the summary directory.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_summary(args.summary)
    write_wide(grouped, output_dir / "summary_wide_k2vae_sparc_pit.csv")
    summarize_overall(grouped, output_dir / "summary_overall_k2vae_sparc_pit.csv")
    print(f"wrote {len(grouped)} task comparisons to {output_dir}")


if __name__ == "__main__":
    main()
