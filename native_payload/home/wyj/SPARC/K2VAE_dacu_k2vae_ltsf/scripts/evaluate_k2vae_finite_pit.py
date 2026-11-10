#!/usr/bin/env python3
"""Evaluate a PPM-style PIT/rank warp on native K2VAE or SPARC samples."""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probts.data import ProbTSBatchData, ProbTSDataModule
from probts.data.data_manager import DataManager
from probts.model.forecast_module import ProbTSForecastModule
from sparc_adapters.metrics import evaluate_samples_like_k2vae
from sparc_adapters.pit_warp import PITWarpCalibrator, qice_from_pit
from finite_pit_utils import (
    PIT_METHODS,
    RankPITAccumulator,
    ensemble_pit,
    qice_from_values,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_checkpoint(root: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in root.rglob("*.ckpt"):
        if path.name == "last.ckpt":
            continue
        match = re.search(r"val_CRPS=([0-9]+(?:\.[0-9]+)?)", path.name)
        score = float(match.group(1)) if match else float("inf")
        candidates.append((score, path))
    if not candidates:
        candidates.extend((float("inf"), path) for path in root.rglob("last.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    return sorted(candidates, key=lambda item: item[0])[0][1]


def load_module(checkpoint: Path, datamodule: ProbTSDataModule, device: torch.device) -> ProbTSForecastModule:
    manager = datamodule.data_manager
    module = ProbTSForecastModule.load_from_checkpoint(
        str(checkpoint),
        scaler=manager.scaler,
        context_length=manager.context_length,
        target_dim=manager.target_dim,
        freq=manager.freq,
        prediction_length=manager.prediction_length,
        train_pred_len_list=manager.train_pred_len_list,
        lags_list=manager.lags_list,
        time_feat_dim=manager.time_feat_dim,
        sampling_weight_scheme="none",
    )
    module = module.to(device).eval()
    module.forecaster = module.forecaster.to(device).eval()
    if hasattr(module.forecaster, "backbone"):
        module.forecaster.backbone = module.forecaster.backbone.to(device).eval()
        if hasattr(module.forecaster.backbone, "model"):
            module.forecaster.backbone.model = module.forecaster.backbone.model.to(device).eval()
    return module


class PITHistogramAccumulator(RankPITAccumulator):
    """Exact PIT accumulator for sample forecasts with finite sample ranks."""

    def __init__(self, sample_count: int):
        if sample_count <= 0:
            raise ValueError(f"sample_count must be positive, got {sample_count}")
        super().__init__(int(sample_count))

    def update(self, samples: np.ndarray, targets: np.ndarray) -> None:
        samples = np.asarray(samples)
        targets = np.asarray(targets)
        if samples.ndim != 4:
            raise ValueError(f"Expected samples [B, N, H, C], got {samples.shape}")
        if targets.ndim != 3:
            raise ValueError(f"Expected targets [B, H, C], got {targets.shape}")
        if samples.shape[1] != self.sample_count:
            raise ValueError(f"Expected {self.sample_count} samples, got {samples.shape[1]}")
        if samples.shape[0] != targets.shape[0] or samples.shape[2:] != targets.shape[1:]:
            raise ValueError(f"Sample shape {samples.shape} is incompatible with target shape {targets.shape}")
        super().update(samples, targets)

    def to_calibrator(
        self,
        method: str,
        rng: np.random.Generator | None,
        n_knots: int = 41,
        strength: float = 1.0,
    ) -> PITWarpCalibrator:
        uniform_knots = np.linspace(0.0, 1.0, n_knots)
        pit_knots = np.quantile(self.values(method, rng), uniform_knots)
        pit_knots[0] = 0.0
        pit_knots[-1] = 1.0
        pit_knots = np.maximum.accumulate(pit_knots)
        eps = 1e-6
        for index in range(1, len(pit_knots)):
            if pit_knots[index] <= pit_knots[index - 1]:
                pit_knots[index] = min(1.0, pit_knots[index - 1] + eps)
        pit_knots[-1] = 1.0
        return PITWarpCalibrator(pit_knots=pit_knots, uniform_knots=uniform_knots, strength=float(strength))


class MetricAverager:
    def __init__(self) -> None:
        self.weighted_sums: dict[str, float] = {}
        self.total_weight = 0

    def update(self, metrics: dict[str, float], weight: int) -> None:
        if weight <= 0:
            return
        for key, value in metrics.items():
            self.weighted_sums[key] = self.weighted_sums.get(key, 0.0) + float(value) * weight
        self.total_weight += weight

    def result(self) -> dict[str, float]:
        if self.total_weight == 0:
            raise ValueError("No metric batches were evaluated")
        return {key: value / self.total_weight for key, value in self.weighted_sums.items()}


def normalized_batch(module: ProbTSForecastModule, batch, device: torch.device) -> ProbTSBatchData:
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    batch_data = ProbTSBatchData(batch, device)
    batch_data.past_target_cdf = module.scaler.transform(batch_data.past_target_cdf)
    batch_data.future_target_cdf = module.scaler.transform(batch_data.future_target_cdf)
    return batch_data


def collect_pit_histogram(
    module: ProbTSForecastModule,
    loader,
    device: torch.device,
    num_samples: int,
    forecast_fn: Callable[[ProbTSBatchData, int], torch.Tensor],
    max_batches: int = 0,
) -> PITHistogramAccumulator:
    accumulator = PITHistogramAccumulator(sample_count=num_samples)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            batch_data = normalized_batch(module, batch, device)
            pred_len = batch_data.future_target_cdf.shape[1]
            samples = forecast_fn(batch_data, num_samples)[:, :, :pred_len]
            accumulator.update(
                samples.detach().cpu().numpy(),
                batch_data.future_target_cdf.detach().cpu().numpy(),
            )
            del samples, batch_data
            if batch_idx % 20 == 0:
                print(f"[fit PIT] collected {batch_idx} batches", flush=True)
    return accumulator


def tensor_to_denorm_numpy(module: ProbTSForecastModule, tensor: torch.Tensor) -> np.ndarray:
    return module.scaler.inverse_transform(tensor).detach().cpu().numpy()


def evaluate_streaming(
    module: ProbTSForecastModule,
    loader,
    device: torch.device,
    num_samples: int,
    quantiles_num: int,
    pit_warp: PITWarpCalibrator,
    k2vae_forecast: Callable[[ProbTSBatchData, int], torch.Tensor],
    sparc_forecast: Callable[[ProbTSBatchData, int], torch.Tensor],
    pit_source: str = "sparc",
    max_batches: int = 0,
    pit_method: str = "empirical",
    pit_seed: int = 1,
) -> tuple[dict[str, float], dict[str, float] | None, dict[str, float]]:
    if pit_source not in {"native", "sparc"}:
        raise ValueError(f"Unsupported PIT source: {pit_source}")

    k2vae = MetricAverager()
    sparc = MetricAverager() if pit_source == "sparc" else None
    pit = MetricAverager()
    k2vae_qice = []
    sparc_qice = []
    warped_qice = []
    k2vae_pit_rng = np.random.default_rng(pit_seed)
    sparc_pit_rng = np.random.default_rng(pit_seed)
    warped_pit_rng = np.random.default_rng(pit_seed)
    warped_ties = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            batch_data = normalized_batch(module, batch, device)
            pred_len = batch_data.future_target_cdf.shape[1]
            targets_denorm = tensor_to_denorm_numpy(module, batch_data.future_target_cdf)
            weight = targets_denorm.shape[0]

            k2vae_samples = k2vae_forecast(batch_data, num_samples)[:, :, :pred_len]
            target_norm = batch_data.future_target_cdf.detach().cpu().numpy()
            k2vae_norm = k2vae_samples.detach().cpu().numpy()
            k2vae_values = ensemble_pit(k2vae_norm, target_norm, pit_method, k2vae_pit_rng).reshape(
                target_norm.shape[0], -1
            )
            k2vae_qice.extend(qice_from_values(row, quantiles_num) for row in k2vae_values)
            k2vae_denorm = tensor_to_denorm_numpy(module, k2vae_samples)
            k2vae.update(evaluate_samples_like_k2vae(targets_denorm, k2vae_denorm, quantiles_num=quantiles_num), weight)

            if pit_source == "sparc":
                sparc_samples = sparc_forecast(batch_data, num_samples)[:, :, :pred_len]
                sparc_norm = sparc_samples.detach().cpu().numpy()
                sparc_values = ensemble_pit(sparc_norm, target_norm, pit_method, sparc_pit_rng).reshape(
                    target_norm.shape[0], -1
                )
                sparc_qice.extend(qice_from_values(row, quantiles_num) for row in sparc_values)
                sparc_denorm = tensor_to_denorm_numpy(module, sparc_samples)
                assert sparc is not None
                sparc.update(
                    evaluate_samples_like_k2vae(targets_denorm, sparc_denorm, quantiles_num=quantiles_num),
                    weight,
                )
                source_samples = sparc_samples
            else:
                sparc_samples = None
                sparc_denorm = None
                source_samples = k2vae_samples
            warped_norm = pit_warp.warp_samples(source_samples.detach().cpu().numpy())
            warped_values = ensemble_pit(
                warped_norm,
                target_norm,
                pit_method,
                warped_pit_rng,
            ).reshape(target_norm.shape[0], -1)
            warped_qice.extend(qice_from_values(row, quantiles_num) for row in warped_values)
            warped_ties += int(np.count_nonzero(warped_norm == target_norm[:, None, :, :]))
            warped_tensor = torch.as_tensor(warped_norm, device=device, dtype=source_samples.dtype)
            warped_denorm = tensor_to_denorm_numpy(module, warped_tensor)
            pit.update(evaluate_samples_like_k2vae(targets_denorm, warped_denorm, quantiles_num=quantiles_num), weight)

            del (
                batch_data,
                targets_denorm,
                k2vae_samples,
                k2vae_denorm,
                sparc_samples,
                sparc_denorm,
                source_samples,
                warped_norm,
                warped_tensor,
                warped_denorm,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_idx % 5 == 0:
                print(f"[eval] {batch_idx} batches", flush=True)

    k2vae_result = k2vae.result()
    k2vae_result["QICE"] = float(np.mean(k2vae_qice))
    sparc_result = sparc.result() if sparc is not None else None
    if sparc_result is not None:
        sparc_result["QICE"] = float(np.mean(sparc_qice))
    pit_result = pit.result()
    pit_result["QICE"] = float(np.mean(warped_qice))
    pit_result["PIT_ties"] = float(warped_ties)
    return k2vae_result, sparc_result, pit_result


def write_comparison(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "CRPS",
        "QICE",
        "ND",
        "NRMSE",
        "MSE",
        "delta_CRPS_vs_k2vae",
        "delta_QICE_vs_k2vae",
        "delta_CRPS_vs_sparc",
        "delta_QICE_vs_sparc",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def metrics_row(method: str, metrics: dict[str, float]) -> dict[str, float | str]:
    return {
        "method": method,
        "CRPS": metrics["CRPS"],
        "QICE": metrics["QICE"],
        "ND": metrics["ND"],
        "NRMSE": metrics["NRMSE"],
        "MSE": metrics["MSE"],
    }


def add_deltas(rows: list[dict[str, float | str]]) -> None:
    k2vae = next(row for row in rows if row["method"] == "K2VAE")
    sparc = next((row for row in rows if row["method"] == "K2VAE-SPARC"), None)
    for row in rows:
        row["delta_CRPS_vs_k2vae"] = float(row["CRPS"]) - float(k2vae["CRPS"])
        row["delta_QICE_vs_k2vae"] = float(row["QICE"]) - float(k2vae["QICE"])
        if sparc is not None:
            row["delta_CRPS_vs_sparc"] = float(row["CRPS"]) - float(sparc["CRPS"])
            row["delta_QICE_vs_sparc"] = float(row["QICE"]) - float(sparc["QICE"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument("--dataset", default="etth1")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--quantiles-num", type=int, default=20)
    parser.add_argument("--fit-split", choices=["train", "val"], default="train")
    parser.add_argument("--fit-limit-batches", type=int, default=100)
    parser.add_argument("--test-limit-batches", type=int, default=0)
    parser.add_argument("--pit-warp-knots", type=int, default=41)
    parser.add_argument("--pit-warp-strength", type=float, default=1.0)
    parser.add_argument("--pit-method", choices=PIT_METHODS, default="empirical")
    parser.add_argument("--pit-random-seed", type=int, default=2027)
    parser.add_argument("--pit-source", choices=["native", "sparc"], default="sparc")
    parser.add_argument(
        "--sparc-reconstruction-mode",
        choices=["residual", "gaussian"],
        default=None,
        help="Override the checkpoint's SPARC sample reconstruction mode.",
    )
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
    if args.sparc_reconstruction_mode is not None:
        module.forecaster.reconstruction_mode = args.sparc_reconstruction_mode
        print(f"[SPARC reconstruction] {args.sparc_reconstruction_mode}", flush=True)

    sparc_forecast = lambda batch_data, n: module.forecaster.forecast(batch_data, n)
    k2vae_forecast = lambda batch_data, n: module.forecaster.backbone.forecast(batch_data, n)
    pit_fit_forecast = k2vae_forecast if args.pit_source == "native" else sparc_forecast
    print(f"[PIT source] {args.pit_source}", flush=True)

    fit_loader = datamodule.train_dataloader() if args.fit_split == "train" else datamodule.val_dataloader()
    pit_histogram = collect_pit_histogram(
        module,
        fit_loader,
        device,
        args.num_samples,
        pit_fit_forecast,
        max_batches=args.fit_limit_batches,
    )
    fit_rng = np.random.default_rng(args.pit_random_seed)
    fit_values = pit_histogram.values(args.pit_method, fit_rng)
    before = qice_from_pit(fit_values, n_bins=args.quantiles_num)
    pit_warp = pit_histogram.to_calibrator(
        args.pit_method,
        np.random.default_rng(args.pit_random_seed),
        n_knots=args.pit_warp_knots,
        strength=args.pit_warp_strength,
    )
    after = qice_from_pit(pit_warp.transform_pit(fit_values), n_bins=args.quantiles_num)
    print(f"[fit PIT] QICE before={before:.6f} after={after:.6f}", flush=True)

    test_loader = datamodule.test_dataloader()
    k2vae_metrics, sparc_metrics, pit_metrics = evaluate_streaming(
        module,
        test_loader,
        device,
        args.num_samples,
        args.quantiles_num,
        pit_warp,
        k2vae_forecast,
        sparc_forecast,
        pit_source=args.pit_source,
        max_batches=args.test_limit_batches,
        pit_method=args.pit_method,
        pit_seed=args.pit_random_seed + 1,
    )

    pit_method = "K2VAE+PIT" if args.pit_source == "native" else "K2VAE-SPARC+PIT"
    rows = [metrics_row("K2VAE", k2vae_metrics)]
    if sparc_metrics is not None:
        rows.append(metrics_row("K2VAE-SPARC", sparc_metrics))
    rows.append(metrics_row(pit_method, pit_metrics))
    add_deltas(rows)
    output_name = (
        f"comparison_k2vae_pit_only_{args.pit_method}.csv"
        if args.pit_source == "native"
        else f"comparison_k2vae_sparc_pit_{args.pit_method}.csv"
    )
    out = args.output_dir / output_name
    write_comparison(out, rows)
    print(
        f"[finite PIT] method={args.pit_method} fit_ties={pit_histogram.tied} "
        f"test_ties={int(pit_metrics['PIT_ties'])}",
        flush=True,
    )
    print(f"[DONE] {out}", flush=True)
    for row in rows:
        print(
            "{method}: CRPS={CRPS:.6f} QICE={QICE:.6f} ND={ND:.6f}".format(
                method=row["method"],
                CRPS=float(row["CRPS"]),
                QICE=float(row["QICE"]),
                ND=float(row["ND"]),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
