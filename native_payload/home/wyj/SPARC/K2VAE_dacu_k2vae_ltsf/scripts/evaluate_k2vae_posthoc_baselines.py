#!/usr/bin/env python3
"""Evaluate post-hoc calibration baselines on native K2VAE samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probts.data import ProbTSDataModule
from probts.data.data_manager import DataManager
from sparc_adapters.metrics import evaluate_samples_like_k2vae
from sparc_adapters.posthoc_baselines import (
    CQRCalibrator,
    ConditionalPITCalibrator,
    DistributionCalibrationCalibrator,
    GlobalAffineCalibrator,
    flatten_forecast,
    interval_metrics,
)
from scripts.evaluate_k2vae_sparc_pit_warp import (
    MetricAverager,
    PITHistogramAccumulator,
    best_checkpoint,
    load_module,
    normalized_batch,
    set_seed,
    tensor_to_denorm_numpy,
)


class CalibrationReservoir:
    """Priority reservoir of forecast coordinates and their sample rows."""

    def __init__(self, max_points: int = 50000, seed: int = 1):
        if max_points <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")
        self.max_points = int(max_points)
        self.rng = np.random.default_rng(seed)
        self._priorities = np.empty(0, dtype=np.float64)
        self._rows = None
        self._targets = np.empty(0, dtype=np.float64)
        self._horizon_ids = np.empty(0, dtype=np.int64)
        self._channel_ids = np.empty(0, dtype=np.int64)
        self.horizon_length = 0
        self.channel_count = 0
        self.seen_points = 0

    def update(self, samples: np.ndarray, targets: np.ndarray) -> None:
        rows, flat_targets, horizon_ids, channel_ids = flatten_forecast(samples, targets)
        if self._rows is not None and rows.shape[1] != self._rows.shape[1]:
            raise ValueError(f"Sample count changed from {self._rows.shape[1]} to {rows.shape[1]}")
        self.horizon_length = samples.shape[2]
        self.channel_count = samples.shape[3]
        self.seen_points += rows.shape[0]
        priorities = self.rng.random(rows.shape[0])
        if rows.shape[0] > self.max_points:
            selected = np.argpartition(priorities, -self.max_points)[-self.max_points :]
            priorities = priorities[selected]
            rows = rows[selected]
            flat_targets = flat_targets[selected]
            horizon_ids = horizon_ids[selected]
            channel_ids = channel_ids[selected]
        if self._rows is None:
            combined_rows = rows
        else:
            combined_rows = np.concatenate([self._rows, rows], axis=0)
        combined_priorities = np.concatenate([self._priorities, priorities])
        combined_targets = np.concatenate([self._targets, flat_targets])
        combined_horizon = np.concatenate([self._horizon_ids, horizon_ids])
        combined_channel = np.concatenate([self._channel_ids, channel_ids])
        if combined_priorities.size > self.max_points:
            keep = np.argpartition(combined_priorities, -self.max_points)[-self.max_points :]
            combined_priorities = combined_priorities[keep]
            combined_rows = combined_rows[keep]
            combined_targets = combined_targets[keep]
            combined_horizon = combined_horizon[keep]
            combined_channel = combined_channel[keep]
        self._priorities = combined_priorities
        self._rows = combined_rows
        self._targets = combined_targets
        self._horizon_ids = combined_horizon
        self._channel_ids = combined_channel

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._rows is None or self._rows.shape[0] == 0:
            raise ValueError("Calibration reservoir is empty")
        return self._rows, self._targets, self._horizon_ids, self._channel_ids


class GlobalAffineAccumulator:
    """Sufficient statistics for global Gaussian affine calibration."""

    def __init__(self) -> None:
        self.count = 0
        self.sum_inverse_variance = 0.0
        self.sum_residual_inverse_variance = 0.0
        self.sum_squared_standardized_residual = 0.0

    def update(self, samples: np.ndarray, targets: np.ndarray) -> None:
        rows, flat_targets, _, _ = flatten_forecast(samples, targets)
        means = rows.mean(axis=1)
        scales = rows.std(axis=1, ddof=rows.shape[1] > 1).clip(1e-6)
        residual = flat_targets - means
        inverse_variance = 1.0 / np.square(scales)
        self.count += residual.size
        self.sum_inverse_variance += float(np.sum(inverse_variance))
        self.sum_residual_inverse_variance += float(np.sum(residual * inverse_variance))
        self.sum_squared_standardized_residual += float(np.sum(np.square(residual / scales)))

    def calibrator(self) -> GlobalAffineCalibrator:
        if self.count == 0:
            raise ValueError("Global affine accumulator is empty")
        bias = self.sum_residual_inverse_variance / self.sum_inverse_variance
        scale_squared = (
            self.sum_squared_standardized_residual
            - 2.0 * bias * self.sum_residual_inverse_variance
            + bias * bias * self.sum_inverse_variance
        ) / self.count
        return GlobalAffineCalibrator(bias=float(bias), scale=float(np.clip(np.sqrt(max(scale_squared, 0.0)), 0.05, 20.0)))


def fit_reservoir_baselines(
    reservoir: CalibrationReservoir,
    affine: GlobalAffineCalibrator,
    isotonic,
    interval_levels: Sequence[float],
    conditional_bins: int,
    distribution_bins: int,
    min_bin_size: int,
) -> Dict[str, object]:
    rows, targets, horizon_ids, channel_ids = reservoir.arrays()
    means = rows.mean(axis=1)
    scales = rows.std(axis=1, ddof=rows.shape[1] > 1).clip(1e-6)
    pit = np.mean(rows <= targets[:, None], axis=1)
    conditional = ConditionalPITCalibrator.fit_summaries(
        pit,
        scales,
        horizon_ids,
        channel_ids,
        horizon_length=reservoir.horizon_length,
        channel_count=reservoir.channel_count,
        n_horizon_bins=conditional_bins,
        n_scale_bins=conditional_bins,
        n_channel_bins=conditional_bins,
        min_bin_size=min_bin_size,
    )
    distribution = DistributionCalibrationCalibrator.fit_summaries(
        (targets - means) / scales,
        means,
        scales,
        horizon_ids,
        horizon_length=reservoir.horizon_length,
        n_mean_bins=distribution_bins,
        n_scale_bins=distribution_bins,
        n_horizon_bins=distribution_bins,
        min_bin_size=min_bin_size,
    )
    cqr = CQRCalibrator.fit(
        rows[:, :, None, None],
        targets[:, None, None],
        levels=interval_levels,
    )
    return {
        "GlobalAffine": affine,
        "IsotonicPIT": isotonic,
        "ConditionalPIT": conditional,
        "DistributionCalibration": distribution,
        "CQR": cqr,
    }


def _denormalize(module, samples: np.ndarray, device: torch.device, dtype: torch.dtype) -> np.ndarray:
    tensor = torch.as_tensor(samples, device=device, dtype=dtype)
    return tensor_to_denorm_numpy(module, tensor)


def _write_metric_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "method",
        "CRPS",
        "QICE",
        "ND",
        "NRMSE",
        "MSE",
        "delta_CRPS_vs_k2vae",
        "delta_QICE_vs_k2vae",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = rows[0]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            current = dict(row)
            current["delta_CRPS_vs_k2vae"] = float(current["CRPS"]) - float(baseline["CRPS"])
            current["delta_QICE_vs_k2vae"] = float(current["QICE"]) - float(baseline["QICE"])
            writer.writerow({field: current.get(field, "") for field in fields})


def _write_interval_rows(path: Path, rows: Sequence[Dict[str, object]], levels: Sequence[float]) -> None:
    fields = ["method", "coverage_error", "mean_interval_score", "mean_interval_width", "levels"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            current = dict(row)
            current["levels"] = ";".join(str(level) for level in levels)
            writer.writerow({field: current.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument("--dataset", default="etth1")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--quantiles-num", type=int, default=20)
    parser.add_argument("--fit-split", choices=["train", "val"], default="train")
    parser.add_argument("--fit-limit-batches", type=int, default=100)
    parser.add_argument("--test-limit-batches", type=int, default=0)
    parser.add_argument("--max-calibration-points", type=int, default=50000)
    parser.add_argument("--conditional-bins", type=int, default=4)
    parser.add_argument("--distribution-bins", type=int, default=4)
    parser.add_argument("--min-bin-size", type=int, default=128)
    parser.add_argument("--pit-warp-knots", type=int, default=41)
    parser.add_argument("--interval-levels", type=float, nargs="+", default=[0.5, 0.8, 0.9])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = best_checkpoint(args.checkpoint) if args.checkpoint.is_dir() else args.checkpoint
    print(f"[checkpoint] {checkpoint}", flush=True)
    manager = DataManager(
        dataset=args.dataset,
        path=str(args.data_dir),
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        split_val=True,
        scaler="standard",
    )
    datamodule = ProbTSDataModule(
        data_manager=manager,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )
    module = load_module(checkpoint, datamodule, device)
    native_forecast = (
        (lambda batch_data, n: module.forecaster.backbone.forecast(batch_data, n))
        if hasattr(module.forecaster, "backbone")
        else (lambda batch_data, n: module.forecaster.forecast(batch_data, n))
    )

    reservoir = CalibrationReservoir(args.max_calibration_points, seed=args.seed)
    affine_accumulator = GlobalAffineAccumulator()
    pit_histogram = PITHistogramAccumulator(args.num_samples)
    fit_loader = datamodule.train_dataloader() if args.fit_split == "train" else datamodule.val_dataloader()
    with torch.no_grad():
        for batch_idx, batch in enumerate(fit_loader, start=1):
            if args.fit_limit_batches > 0 and batch_idx > args.fit_limit_batches:
                break
            batch_data = normalized_batch(module, batch, device)
            pred_len = batch_data.future_target_cdf.shape[1]
            samples = native_forecast(batch_data, args.num_samples)[:, :, :pred_len]
            samples_np = samples.detach().cpu().numpy()
            targets_np = batch_data.future_target_cdf.detach().cpu().numpy()
            reservoir.update(samples_np, targets_np)
            affine_accumulator.update(samples_np, targets_np)
            pit_histogram.update(samples_np, targets_np)
            del samples, samples_np, targets_np, batch_data
            if batch_idx % 20 == 0:
                print(f"[fit] {batch_idx} batches reservoir={reservoir.arrays()[0].shape[0]}", flush=True)

    affine = affine_accumulator.calibrator()
    isotonic = pit_histogram.to_calibrator(n_knots=args.pit_warp_knots, strength=1.0)
    calibrators = fit_reservoir_baselines(
        reservoir,
        affine,
        isotonic,
        interval_levels=args.interval_levels,
        conditional_bins=args.conditional_bins,
        distribution_bins=args.distribution_bins,
        min_bin_size=args.min_bin_size,
    )
    print(
        f"[fit] points_seen={reservoir.seen_points} retained={reservoir.arrays()[0].shape[0]} "
        f"affine_bias={affine.bias:.6f} affine_scale={affine.scale:.6f} "
        f"conditional_maps={len(calibrators['ConditionalPIT'].bin_warps)} "
        f"distribution_pools={len(calibrators['DistributionCalibration'].bin_residuals)}",
        flush=True,
    )

    metric_averagers = {
        method: MetricAverager()
        for method in ("K2VAE", "GlobalAffine", "IsotonicPIT", "ConditionalPIT", "DistributionCalibration")
    }
    interval_averagers = {method: MetricAverager() for method in ("K2VAE", "CQR")}
    test_loader = datamodule.test_dataloader()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            if args.test_limit_batches > 0 and batch_idx > args.test_limit_batches:
                break
            batch_data = normalized_batch(module, batch, device)
            pred_len = batch_data.future_target_cdf.shape[1]
            native_tensor = native_forecast(batch_data, args.num_samples)[:, :, :pred_len]
            native = native_tensor.detach().cpu().numpy()
            targets_norm = batch_data.future_target_cdf.detach().cpu().numpy()
            targets_denorm = tensor_to_denorm_numpy(module, batch_data.future_target_cdf)
            weight = targets_denorm.shape[0]

            transformed = {
                "K2VAE": native,
                "GlobalAffine": calibrators["GlobalAffine"].transform(native),
                "IsotonicPIT": calibrators["IsotonicPIT"].warp_samples(native),
                "ConditionalPIT": calibrators["ConditionalPIT"].transform(native),
                "DistributionCalibration": calibrators["DistributionCalibration"].transform(native),
            }
            for method, samples_norm in transformed.items():
                samples_denorm = _denormalize(module, samples_norm, device, native_tensor.dtype)
                metrics = evaluate_samples_like_k2vae(
                    targets_denorm,
                    samples_denorm,
                    quantiles_num=args.quantiles_num,
                )
                metric_averagers[method].update(metrics, weight)

            interval_averagers["K2VAE"].update(
                interval_metrics(native, targets_norm, levels=args.interval_levels),
                weight,
            )
            interval_averagers["CQR"].update(
                interval_metrics(native, targets_norm, levels=args.interval_levels, calibrator=calibrators["CQR"]),
                weight,
            )
            del batch_data, native_tensor, native, targets_norm, targets_denorm, transformed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_idx % 5 == 0:
                print(f"[eval] {batch_idx} batches", flush=True)

    metric_rows = [{"method": method, **averager.result()} for method, averager in metric_averagers.items()]
    interval_rows = [{"method": method, **averager.result()} for method, averager in interval_averagers.items()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = args.output_dir / "comparison_posthoc_baselines.csv"
    interval_path = args.output_dir / "comparison_cqr.csv"
    _write_metric_rows(metric_path, metric_rows)
    _write_interval_rows(interval_path, interval_rows, args.interval_levels)
    config = vars(args).copy()
    config["checkpoint"] = str(checkpoint)
    config["data_dir"] = str(args.data_dir)
    config["output_dir"] = str(args.output_dir)
    config["points_seen"] = reservoir.seen_points
    config["points_retained"] = reservoir.arrays()[0].shape[0]
    config["affine_bias"] = affine.bias
    config["affine_scale"] = affine.scale
    config["conditional_maps"] = len(calibrators["ConditionalPIT"].bin_warps)
    config["distribution_pools"] = len(calibrators["DistributionCalibration"].bin_residuals)
    (args.output_dir / "posthoc_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"[DONE] {metric_path}", flush=True)
    for row in metric_rows:
        print(f"{row['method']}: CRPS={row['CRPS']:.6f} QICE={row['QICE']:.6f}", flush=True)
    for row in interval_rows:
        print(
            f"{row['method']}: coverage_error={row['coverage_error']:.6f} "
            f"mean_interval_score={row['mean_interval_score']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
