#!/usr/bin/env python3
"""Evaluate sample-level post-hoc calibration baselines on frozen PPM forecasts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
K2VAE_ROOT = Path("/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf")
for path in (REPO_ROOT, K2VAE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_provider.data_factory import data_provider
from model_ppm.exp.exp_main import Exp_Main
from run_sparc import batch_target, batch_to_samples, set_seed
from sparc_adapters.pit_warp import PITWarpCalibrator
from sparc_adapters.posthoc_baselines import (
    CQRCalibrator,
    ConditionalPITCalibrator,
    DistributionCalibrationCalibrator,
    GlobalAffineCalibrator,
    flatten_forecast,
    interval_metrics,
)
from sparc_ppm.posthoc import MeanMetricAccumulator, PPMMetricAccumulator


class CalibrationReservoir:
    def __init__(self, max_points: int, seed: int):
        self.max_points = int(max_points)
        self.rng = np.random.default_rng(seed)
        self.priorities = np.empty(0)
        self.rows: np.ndarray | None = None
        self.targets = np.empty(0)
        self.horizon_ids = np.empty(0, dtype=np.int64)
        self.channel_ids = np.empty(0, dtype=np.int64)
        self.horizon_length = 0
        self.channel_count = 0
        self.seen_points = 0

    def update(self, samples: np.ndarray, targets: np.ndarray) -> None:
        rows, values, horizon_ids, channel_ids = flatten_forecast(samples, targets)
        self.horizon_length = samples.shape[2]
        self.channel_count = samples.shape[3]
        self.seen_points += rows.shape[0]
        priorities = self.rng.random(rows.shape[0])
        if rows.shape[0] > self.max_points:
            keep = np.argpartition(priorities, -self.max_points)[-self.max_points :]
            priorities, rows, values = priorities[keep], rows[keep], values[keep]
            horizon_ids, channel_ids = horizon_ids[keep], channel_ids[keep]
        all_rows = rows if self.rows is None else np.concatenate([self.rows, rows])
        all_priorities = np.concatenate([self.priorities, priorities])
        all_targets = np.concatenate([self.targets, values])
        all_horizon = np.concatenate([self.horizon_ids, horizon_ids])
        all_channel = np.concatenate([self.channel_ids, channel_ids])
        if all_priorities.size > self.max_points:
            keep = np.argpartition(all_priorities, -self.max_points)[-self.max_points :]
            all_priorities, all_rows, all_targets = (
                all_priorities[keep],
                all_rows[keep],
                all_targets[keep],
            )
            all_horizon, all_channel = all_horizon[keep], all_channel[keep]
        self.priorities, self.rows, self.targets = all_priorities, all_rows, all_targets
        self.horizon_ids, self.channel_ids = all_horizon, all_channel

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.rows is None or not len(self.rows):
            raise ValueError("Calibration reservoir is empty")
        return self.rows, self.targets, self.horizon_ids, self.channel_ids


class GlobalAffineAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum_inverse_variance = 0.0
        self.sum_residual_inverse_variance = 0.0
        self.sum_squared_standardized_residual = 0.0

    def update(self, samples: np.ndarray, targets: np.ndarray) -> None:
        rows, values, _, _ = flatten_forecast(samples, targets)
        means = rows.mean(1)
        scales = rows.std(1, ddof=rows.shape[1] > 1).clip(1e-6)
        residual = values - means
        inverse_variance = 1.0 / np.square(scales)
        self.count += residual.size
        self.sum_inverse_variance += float(inverse_variance.sum())
        self.sum_residual_inverse_variance += float((residual * inverse_variance).sum())
        self.sum_squared_standardized_residual += float(np.square(residual / scales).sum())

    def calibrator(self) -> GlobalAffineCalibrator:
        bias = self.sum_residual_inverse_variance / self.sum_inverse_variance
        scale_squared = (
            self.sum_squared_standardized_residual
            - 2.0 * bias * self.sum_residual_inverse_variance
            + bias * bias * self.sum_inverse_variance
        ) / self.count
        return GlobalAffineCalibrator(
            bias=float(bias),
            scale=float(np.clip(np.sqrt(max(scale_squared, 0.0)), 0.05, 20.0)),
        )


def fit_calibrators(args, model, loader, device):
    reservoir = CalibrationReservoir(args.max_calibration_points, args.seed)
    affine_stats = GlobalAffineAccumulator()
    model.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y, _x_mark, _y_mark) in enumerate(loader, start=1):
            if args.fit_limit_batches > 0 and step > args.fit_limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            samples = batch_to_samples(model, batch_x, args.num_samples, args).cpu().numpy()
            targets = batch_target(batch_y, args).cpu().numpy()
            reservoir.update(samples, targets)
            affine_stats.update(samples, targets)
            if step % 20 == 0:
                print(f"[fit] batches={step} retained={len(reservoir.arrays()[0])}", flush=True)
    rows, targets, horizon_ids, channel_ids = reservoir.arrays()
    means = rows.mean(1)
    scales = rows.std(1, ddof=rows.shape[1] > 1).clip(1e-6)
    pit = np.mean(rows <= targets[:, None], axis=1)
    calibrators = {
        "GlobalAffine": affine_stats.calibrator(),
        "IsotonicPIT": PITWarpCalibrator.from_pit(pit, n_knots=args.pit_warp_knots),
        "ConditionalPIT": ConditionalPITCalibrator.fit_summaries(
            pit,
            scales,
            horizon_ids,
            channel_ids,
            reservoir.horizon_length,
            reservoir.channel_count,
            n_horizon_bins=args.conditional_bins,
            n_scale_bins=args.conditional_bins,
            n_channel_bins=args.conditional_bins,
            min_bin_size=args.min_bin_size,
            n_knots=args.pit_warp_knots,
        ),
        "DistributionCalibration": DistributionCalibrationCalibrator.fit_summaries(
            (targets - means) / scales,
            means,
            scales,
            horizon_ids,
            reservoir.horizon_length,
            n_mean_bins=args.distribution_bins,
            n_scale_bins=args.distribution_bins,
            n_horizon_bins=args.distribution_bins,
            min_bin_size=args.min_bin_size,
        ),
        "CQR": CQRCalibrator.fit(
            rows[:, :, None, None], targets[:, None, None], levels=args.interval_levels
        ),
    }
    return calibrators, reservoir


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def add_model_defaults(args) -> None:
    defaults = {
        "model": "MLP", "features": "M", "target": "OT", "freq": "h",
        "label_len": 0, "dec_in": args.enc_in, "c_out": args.enc_in,
        "n_heads": 8, "d_layers": 1, "d_model_f": 256, "d_ff_f": 512,
        "dropout": 0.05, "embed": "timeF", "activation": "gelu",
        "output_attention": False, "do_predict": False, "itr": 1,
        "train_epochs": 30, "patience": 5, "learning_rate": 1e-4,
        "des": "Exp", "loss": "mse", "lradj": "type1", "use_amp": False,
        "cond_learning_rate": 1e-4, "std_learning_rate": 5e-4,
        "use_gpu": torch.cuda.is_available(), "use_multi_gpu": False,
        "devices": "0", "K": args.num_samples, "tau": 0.1, "var": 0.3,
        "nu": 5, "checkpoints": "./checkpoints/", "pretrain_checkpoints": "./pretrain_checkpoints/",
        "sample_chunk_size": args.num_samples,
    }
    for key, value in defaults.items():
        setattr(args, key, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=192)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--e_layers", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--ppm_checkpoint", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--fit_split", choices=["train", "val"], default="train")
    parser.add_argument("--fit_limit_batches", type=int, default=100)
    parser.add_argument("--test_limit_batches", type=int, default=0)
    parser.add_argument("--max_calibration_points", type=int, default=50000)
    parser.add_argument("--conditional_bins", type=int, default=4)
    parser.add_argument("--distribution_bins", type=int, default=4)
    parser.add_argument("--min_bin_size", type=int, default=128)
    parser.add_argument("--pit_warp_knots", type=int, default=41)
    parser.add_argument("--interval_levels", type=float, nargs="+", default=[0.5, 0.8, 0.9])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    add_model_defaults(args)
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    exp = Exp_Main(args)
    exp.model.load_state_dict(torch.load(args.ppm_checkpoint, map_location=exp.device))
    train_data, train_loader = data_provider(args, "train")
    val_data, val_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")
    fit_loader = train_loader if args.fit_split == "train" else val_loader
    calibrators, reservoir = fit_calibrators(args, exp.model, fit_loader, exp.device)
    affine = calibrators["GlobalAffine"]
    print(
        f"[fit] global_affine bias={affine.bias:.8g} scale={affine.scale:.8g}",
        flush=True,
    )

    metric_accumulators = {
        method: PPMMetricAccumulator()
        for method in ("PPM", "GlobalAffine", "IsotonicPIT", "ConditionalPIT", "DistributionCalibration")
    }
    interval_accumulators = {method: MeanMetricAccumulator() for method in ("PPM", "CQR")}
    exp.model.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y, _x_mark, _y_mark) in enumerate(test_loader, start=1):
            if args.test_limit_batches > 0 and step > args.test_limit_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            native = batch_to_samples(exp.model, batch_x, args.num_samples, args).cpu().numpy()
            targets = batch_target(batch_y, args).cpu().numpy()
            transforms = {
                "PPM": native,
                "GlobalAffine": calibrators["GlobalAffine"].transform(native),
                "IsotonicPIT": calibrators["IsotonicPIT"].warp_samples(native),
                "ConditionalPIT": calibrators["ConditionalPIT"].transform(native),
                "DistributionCalibration": calibrators["DistributionCalibration"].transform(native),
            }
            for method, samples in transforms.items():
                metric_accumulators[method].update(samples, targets)
            interval_accumulators["PPM"].update(
                interval_metrics(native, targets, args.interval_levels), native.shape[0]
            )
            interval_accumulators["CQR"].update(
                interval_metrics(native, targets, args.interval_levels, calibrators["CQR"]), native.shape[0]
            )
            if step % 5 == 0:
                print(f"[eval] batches={step}/{len(test_loader)}", flush=True)

    metric_rows = [{"method": method, **acc.result()} for method, acc in metric_accumulators.items()]
    baseline = metric_rows[0]
    for row in metric_rows:
        row["crps_delta"] = row["CRPS"] - baseline["CRPS"]
        row["qice_delta"] = row["QICE"] - baseline["QICE"]
    interval_rows = [{"method": method, **acc.result()} for method, acc in interval_accumulators.items()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(
        args.output_dir / "comparison_posthoc_baselines.csv",
        metric_rows,
        ["method", "CRPS", "QICE", "MSE", "MAE", "crps_delta", "qice_delta"],
    )
    write_rows(
        args.output_dir / "comparison_cqr.csv",
        interval_rows,
        ["method", "coverage_error", "mean_interval_score", "mean_interval_width"],
    )
    config = vars(args).copy()
    config.update(
        ppm_checkpoint=str(args.ppm_checkpoint),
        output_dir=str(args.output_dir),
        points_seen=reservoir.seen_points,
        points_retained=len(reservoir.arrays()[0]),
        global_affine_bias=affine.bias,
        global_affine_scale=affine.scale,
    )
    (args.output_dir / "posthoc_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"[DONE] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
