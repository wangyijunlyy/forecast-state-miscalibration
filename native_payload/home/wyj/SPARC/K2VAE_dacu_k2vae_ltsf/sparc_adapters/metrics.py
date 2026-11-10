"""K2VAE-compatible metrics for external SPARC adapters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _ensure_target(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim == 2:
        targets = targets[..., None]
    if targets.ndim != 3:
        raise ValueError(f"Expected targets as [B, H, C], got {targets.shape}")
    return np.nan_to_num(targets)


def _ensure_forecasts(forecasts: np.ndarray) -> np.ndarray:
    forecasts = np.asarray(forecasts, dtype=np.float64)
    if forecasts.ndim == 3:
        forecasts = forecasts[..., None]
    if forecasts.ndim != 4:
        raise ValueError(f"Expected forecasts as [B, N, H, C], got {forecasts.shape}")
    return np.nan_to_num(forecasts)


def _quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    return float(2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q))))


def _sequence_metrics(
    targets: np.ndarray,
    forecasts: np.ndarray,
    quantiles_num: int = 10,
    samples_dim: int = 1,
) -> Dict[str, float]:
    quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
    mean_forecasts = forecasts.mean(axis=samples_dim)
    median_forecasts = np.quantile(forecasts, 0.5, axis=samples_dim)

    mse = float(np.mean(np.square(targets - mean_forecasts)))
    abs_error = float(np.sum(np.abs(targets - median_forecasts)))
    abs_target_sum = float(np.sum(np.abs(targets)))
    abs_target_mean = float(np.mean(np.abs(targets)))
    abs_target_sum = max(abs_target_sum, 1e-8)
    abs_target_mean = max(abs_target_mean, 1e-8)

    metrics: Dict[str, float] = {
        "MSE": mse,
        "abs_error": abs_error,
        "abs_target_sum": abs_target_sum,
        "abs_target_mean": abs_target_mean,
        "RMSE": float(np.sqrt(mse)),
        "NRMSE": float(np.sqrt(mse) / abs_target_mean),
        "ND": float(abs_error / abs_target_sum),
    }
    metrics["weighted_ND"] = metrics["ND"]

    quantile_losses = []
    weighted_quantile_losses = []
    coverages = []
    for q in quantiles:
        q_forecasts = np.quantile(forecasts, q, axis=samples_dim)
        q_loss = _quantile_loss(targets, q_forecasts, float(q))
        weighted_q_loss = q_loss / abs_target_sum
        coverage = float(np.mean(targets < q_forecasts))

        q_key = f"QuantileLoss[{q}]"
        wq_key = f"wQuantileLoss[{q}]"
        cov_key = f"Coverage[{q}]"
        metrics[q_key] = q_loss
        metrics[wq_key] = weighted_q_loss
        metrics[cov_key] = coverage
        quantile_losses.append(q_loss)
        weighted_quantile_losses.append(weighted_q_loss)
        coverages.append(coverage)

    metrics["mean_absolute_QuantileLoss"] = float(np.mean(quantile_losses))
    metrics["CRPS"] = float(np.mean(weighted_quantile_losses))
    metrics["MAE_Coverage"] = float(np.mean([abs(metrics[f"Coverage[{q}]"] - q) for q in quantiles]))

    all_coverages = [0.0] + coverages + [1.0]
    observed_proportions = np.diff(all_coverages)
    optimal_proportion = 1.0 / (len(quantiles) + 1)
    metrics["QICE"] = float(np.mean(np.abs(observed_proportions - optimal_proportion)))
    return metrics


def evaluate_samples_like_k2vae(
    targets: np.ndarray,
    forecasts: np.ndarray,
    quantiles_num: int = 10,
) -> Dict[str, float]:
    """Evaluate [B, N, H, C] samples with K2VAE/ProbTS metric semantics."""

    targets = _ensure_target(targets)
    forecasts = _ensure_forecasts(forecasts)
    if forecasts.shape[0] != targets.shape[0] or forecasts.shape[2:] != targets.shape[1:]:
        raise ValueError(f"Forecast shape {forecasts.shape} is incompatible with target shape {targets.shape}")

    per_sequence: Dict[str, list[float]] = {}
    for index in range(targets.shape[0]):
        metrics = _sequence_metrics(
            targets[index : index + 1],
            forecasts[index : index + 1],
            quantiles_num=quantiles_num,
            samples_dim=1,
        )
        for key, value in metrics.items():
            per_sequence.setdefault(key, []).append(value)

    output = {key: float(np.mean(values)) for key, values in per_sequence.items()}

    sum_targets = targets.sum(axis=-1)
    sum_forecasts = forecasts.sum(axis=-1)
    sum_metrics = evaluate_sequence_sum_like_k2vae(sum_targets, sum_forecasts, quantiles_num=quantiles_num)
    for key, value in sum_metrics.items():
        output[f"{key}-Sum"] = value
    return output


def evaluate_sequence_sum_like_k2vae(
    targets: np.ndarray,
    forecasts: np.ndarray,
    quantiles_num: int = 10,
) -> Dict[str, float]:
    targets = _ensure_target(targets)
    forecasts = _ensure_forecasts(forecasts)
    per_sequence: Dict[str, list[float]] = {}
    for index in range(targets.shape[0]):
        metrics = _sequence_metrics(
            targets[index : index + 1],
            forecasts[index : index + 1],
            quantiles_num=quantiles_num,
            samples_dim=1,
        )
        for key, value in metrics.items():
            per_sequence.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in per_sequence.items()}


def write_horizons_results_csv(
    output_dir: str | Path,
    metrics: Dict[str, float],
    model_args: str,
    horizon: Optional[int] = None,
    filename: str = "horizons_results.csv",
) -> Path:
    """Write one K2VAE-style horizons_results.csv row."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    row: Dict[str, str | float | int] = {"model_args": model_args}
    row.update(metrics)
    if horizon is not None:
        row["horizon"] = horizon

    preferred = [
        "model_args",
        "horizon",
        "CRPS",
        "ND",
        "weighted_ND",
        "NRMSE",
        "MSE",
        "QICE",
        "CRPS-Sum",
        "ND-Sum",
        "weighted_ND-Sum",
        "NRMSE-Sum",
        "MSE-Sum",
        "QICE-Sum",
    ]
    fields = [field for field in preferred if field in row]
    fields.extend(sorted(field for field in row if field not in fields))

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    return output_path
