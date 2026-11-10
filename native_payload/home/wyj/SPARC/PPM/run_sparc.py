#!/usr/bin/env python
"""Train SPARC on a frozen official PPM checkpoint and evaluate with PPM metrics."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch

from data_provider.data_factory import data_provider
from model_ppm.exp.exp_main import Exp_Main
from sparc_ppm.calibration import SampleSPARCCalibrator, augment_context
from sparc_ppm.pit_warp import PITWarpCalibrator, compute_sample_pit, qice_from_pit
from sparc_ppm.ppm_metrics import evaluate_ppm_samples


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ppm_setting(args: argparse.Namespace) -> str:
    return "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_{}".format(
        args.model_id,
        args.model,
        args.data_name,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.d_ff,
        args.des,
    )


def batch_to_samples(model: torch.nn.Module, batch_x: torch.Tensor, num_samples: int, args: argparse.Namespace) -> torch.Tensor:
    chunk_size = int(getattr(args, "sample_chunk_size", 0) or 0)
    if chunk_size > 0 and num_samples > chunk_size:
        chunks = []
        remaining = num_samples
        while remaining > 0:
            current = min(chunk_size, remaining)
            chunks.append(model(batch_x, current)[:, :, -args.pred_len :, :])
            remaining -= current
        outputs = torch.cat(chunks, dim=1)
    else:
        outputs = model(batch_x, num_samples)
        outputs = outputs[:, :, -args.pred_len :, :]
    f_dim = -1 if args.features == "MS" else 0
    return outputs[:, :, :, f_dim:]


def batch_target(batch_y: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    f_dim = -1 if args.features == "MS" else 0
    return batch_y[:, -args.pred_len :, f_dim:]


def augment_for_args(batch_x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return augment_context(
        batch_x,
        jitter_std=args.jitter_std,
        scaling_std=args.scaling_std,
        time_mask_ratio=args.time_mask_ratio,
    )


def train_sparc(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    best_path = output_dir / "sparc_ckpt.pt"
    if args.reuse_sparc_ckpt and best_path.exists():
        print(f"[SPARC] reuse existing checkpoint: {best_path}", flush=True)
        return best_path

    calibrator = SampleSPARCCalibrator(
        prediction_length=args.pred_len,
        target_dim=args.c_out,
        hidden_dim=args.sparc_hidden_dim,
        mean_correction=args.mean_correction,
        max_mean_delta=args.max_mean_delta,
        max_log_scale_delta=args.max_log_scale_delta,
        calibration_strength=args.calibration_strength,
        nll_weight=args.sparc_nll_weight,
        quantile_weight=args.sparc_quantile_weight,
        pit_weight=args.sparc_pit_weight,
        pit_temperature=args.sparc_pit_temperature,
        use_perturbation_features=args.use_perturbation_features,
        bounded_updates=args.bounded_updates,
        reconstruction_mode=args.reconstruction_mode,
    ).to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.sparc_learning_rate)
    best_loss = float("inf")
    bad_epochs = 0
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    for epoch in range(1, args.sparc_epochs + 1):
        calibrator.train()
        train_losses = []
        for step, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(train_loader, start=1):
            if args.limit_train_batches > 0 and step > args.limit_train_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            target = batch_target(batch_y, args)
            with torch.no_grad():
                base_stats = batch_to_samples(model, batch_x, args.stats_num_samples, args)
                aug_stats = batch_to_samples(model, augment_for_args(batch_x, args), args.stats_num_samples, args)
            loss = calibrator.loss(base_stats, aug_stats, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = validate_sparc(calibrator, model, val_loader, device, args)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"[SPARC] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": calibrator.state_dict(), "best_val_loss": best_loss, "args": vars(args)}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.sparc_patience:
                print(f"[SPARC] early stopping at epoch={epoch}", flush=True)
                break
    return best_path


def validate_sparc(
    calibrator: SampleSPARCCalibrator,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    calibrator.eval()
    losses = []
    with torch.no_grad():
        for step, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader, start=1):
            if args.limit_val_batches > 0 and step > args.limit_val_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            target = batch_target(batch_y, args)
            base_stats = batch_to_samples(model, batch_x, args.stats_num_samples, args)
            aug_stats = batch_to_samples(model, augment_for_args(batch_x, args), args.stats_num_samples, args)
            losses.append(float(calibrator.loss(base_stats, aug_stats, target).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def collect_sparc_samples(
    calibrator: SampleSPARCCalibrator,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    calibrator.eval()
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output_samples = batch_to_samples(model, batch_x, args.K, args)
            base_stats = batch_to_samples(model, batch_x, args.stats_num_samples, args)
            aug_stats = batch_to_samples(model, augment_for_args(batch_x, args), args.stats_num_samples, args)
            samples = calibrator.calibrate_samples_with_stats(output_samples, base_stats, aug_stats)
            preds.append(samples.detach().cpu().numpy())
            trues.append(batch_target(batch_y, args).detach().cpu().numpy())
            if i % 10 == 0 and i != 0:
                print(f"[SPARC] testing {i}/{len(loader)}", flush=True)
    return preds, trues


def collect_native_samples(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    num_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            samples = batch_to_samples(model, batch_x, num_samples, args)
            preds.append(samples.detach().cpu().numpy())
            trues.append(batch_target(batch_y, args).detach().cpu().numpy())
            if i % 10 == 0 and i != 0:
                print(f"[PIT-WARP] collect {i}/{len(loader)}", flush=True)
    return preds, trues


def fit_pit_warp(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> PITWarpCalibrator:
    preds, trues = collect_native_samples(model, loader, device, args, args.stats_num_samples)
    pit = np.concatenate([compute_sample_pit(pred, true) for pred, true in zip(preds, trues)], axis=0)
    before = qice_from_pit(pit, n_bins=10)
    calibrator = PITWarpCalibrator.from_pit(pit, n_knots=args.pit_warp_knots, strength=args.pit_warp_strength)
    after = qice_from_pit(calibrator.transform_pit(pit), n_bins=10)
    print(f"[PIT-WARP] calibration_qice_before={before * 100:.4f}% after={after * 100:.4f}%", flush=True)
    return calibrator


def collect_pit_warp_samples(
    calibrator: PITWarpCalibrator,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    preds, trues = collect_native_samples(model, loader, device, args, args.K)
    warped = [calibrator.warp_samples(pred) for pred in preds]
    return warped, trues


def fit_pit_warp_from_samples(
    preds: list[np.ndarray],
    trues: list[np.ndarray],
    args: argparse.Namespace,
    label: str,
) -> PITWarpCalibrator:
    pit = np.concatenate([compute_sample_pit(pred, true) for pred, true in zip(preds, trues)], axis=0)
    before = qice_from_pit(pit, n_bins=10)
    calibrator = PITWarpCalibrator.from_pit(pit, n_knots=args.pit_warp_knots, strength=args.pit_warp_strength)
    after = qice_from_pit(calibrator.transform_pit(pit), n_bins=10)
    print(f"[{label}] calibration_qice_before={before * 100:.4f}% after={after * 100:.4f}%", flush=True)
    return calibrator


def fit_pit_warp_from_pit(pit: np.ndarray, args: argparse.Namespace, label: str) -> PITWarpCalibrator:
    before = qice_from_pit(pit, n_bins=10)
    calibrator = PITWarpCalibrator.from_pit(pit, n_knots=args.pit_warp_knots, strength=args.pit_warp_strength)
    after = qice_from_pit(calibrator.transform_pit(pit), n_bins=10)
    print(
        f"[{label}] fit_points={pit.size} calibration_qice_before={before * 100:.4f}% "
        f"after={after * 100:.4f}%",
        flush=True,
    )
    return calibrator


def sample_pit_values(pit: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    pit = np.asarray(pit, dtype=np.float64).reshape(-1)
    if max_points <= 0 or pit.size <= max_points:
        return pit
    index = rng.choice(pit.size, size=max_points, replace=False)
    return pit[index]


def collect_sparc_pit_values(
    calibrator: SampleSPARCCalibrator,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    """Collect bounded PIT values for fitting PIT warp without storing full sample tensors."""
    calibrator.eval()
    model.eval()
    pit_values = []
    total_points = 0
    rng = np.random.default_rng(args.seed + 2027)
    max_points = int(args.max_pit_fit_points)
    with torch.no_grad():
        for i, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader, start=1):
            if args.limit_pit_fit_batches > 0 and i > args.limit_pit_fit_batches:
                break
            if max_points > 0 and total_points >= max_points:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            output_samples = batch_to_samples(model, batch_x, args.K, args)
            base_stats = batch_to_samples(model, batch_x, args.stats_num_samples, args)
            aug_stats = batch_to_samples(model, augment_for_args(batch_x, args), args.stats_num_samples, args)
            samples = calibrator.calibrate_samples_with_stats(output_samples, base_stats, aug_stats)
            pit = compute_sample_pit(samples.detach().cpu().numpy(), batch_target(batch_y, args).detach().cpu().numpy())
            remaining = max_points - total_points if max_points > 0 else 0
            pit = sample_pit_values(pit, remaining, rng) if max_points > 0 else pit
            pit_values.append(pit)
            total_points += pit.size
            if i % 10 == 0 or (max_points > 0 and total_points >= max_points):
                print(f"[SPARC+PIT-WARP] fit {i}/{len(loader)} pit_points={total_points}", flush=True)
    if not pit_values:
        raise RuntimeError("No PIT values collected for PIT warp fitting")
    return np.concatenate(pit_values, axis=0)


def write_comparison(output_dir: Path, ppm_metrics: dict[str, float], sparc_metrics: dict[str, float]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison_ppm_sparc.csv"
    fields = ["method", "CRPS", "QICE", "MSE", "MAE", "crps_delta", "qice_delta"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"method": "PPM", **{k: ppm_metrics[k] for k in ["CRPS", "QICE", "MSE", "MAE"]}})
        writer.writerow(
            {
                "method": "SPARC-PPM",
                **{k: sparc_metrics[k] for k in ["CRPS", "QICE", "MSE", "MAE"]},
                "crps_delta": sparc_metrics["CRPS"] - ppm_metrics["CRPS"],
                "qice_delta": sparc_metrics["QICE"] - ppm_metrics["QICE"],
            }
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--model_id", type=str, default="ETTh1_MLP_K100")
    parser.add_argument("--model", type=str, default="MLP")
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--data_name", type=str, default="ETTh1")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--pretrain_checkpoints", type=str, default="./pretrain_checkpoints/")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=192)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_model_f", type=int, default=256)
    parser.add_argument("--d_ff_f", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--des", type=str, default="Exp")
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--cond_learning_rate", type=float, default=0.0001)
    parser.add_argument("--std_learning_rate", type=float, default=0.0005)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", type=str2bool, default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--var", type=float, default=0.3)
    parser.add_argument("--nu", type=float, default=5)
    parser.add_argument("--ppm_checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=Path, default=Path("./results_sparc/ETTh1_96_192"))
    parser.add_argument("--calibration_mode", choices=["sparc_head", "pit_warp", "sparc_head_pit_warp"], default="sparc_head")
    parser.add_argument("--pit_warp_fit_split", choices=["train", "val"], default="val")
    parser.add_argument("--pit_warp_knots", type=int, default=21)
    parser.add_argument("--pit_warp_strength", type=float, default=1.0)
    parser.add_argument("--stats_num_samples", type=int, default=100)
    parser.add_argument("--sample_chunk_size", type=int, default=100)
    parser.add_argument("--sparc_epochs", type=int, default=20)
    parser.add_argument("--sparc_patience", type=int, default=5)
    parser.add_argument("--sparc_learning_rate", type=float, default=0.001)
    parser.add_argument("--sparc_hidden_dim", type=int, default=128)
    parser.add_argument("--mean_correction", type=str2bool, default=True)
    parser.add_argument("--calibration_strength", type=float, default=0.5)
    parser.add_argument("--max_mean_delta", type=float, default=0.05)
    parser.add_argument("--max_log_scale_delta", type=float, default=0.5)
    parser.add_argument("--sparc_nll_weight", type=float, default=0.5)
    parser.add_argument("--sparc_quantile_weight", type=float, default=0.5)
    parser.add_argument("--sparc_pit_weight", type=float, default=30.0)
    parser.add_argument("--sparc_pit_temperature", type=float, default=0.01)
    parser.add_argument("--use_perturbation_features", type=str2bool, default=True)
    parser.add_argument("--jitter_std", type=float, default=0.01)
    parser.add_argument("--scaling_std", type=float, default=0.05)
    parser.add_argument("--time_mask_ratio", type=float, default=0.05)
    parser.add_argument("--bounded_updates", type=str2bool, default=True)
    parser.add_argument("--reconstruction_mode", choices=["residual", "gaussian"], default="residual")
    parser.add_argument("--limit_train_batches", type=int, default=100)
    parser.add_argument("--limit_val_batches", type=int, default=0)
    parser.add_argument("--limit_pit_fit_batches", type=int, default=0)
    parser.add_argument("--max_pit_fit_points", type=int, default=2_000_000)
    parser.add_argument("--reuse_sparc_ckpt", type=str2bool, default=False)
    parser.add_argument("--ppm_crps", type=float, default=0.3368)
    parser.add_argument("--ppm_qice", type=float, default=1.4745)
    parser.add_argument("--ppm_mse", type=float, default=0.4472)
    parser.add_argument("--ppm_mae", type=float, default=0.4325)
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.label_len = 0
    set_seed(args.seed)
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)

    setting = ppm_setting(args)
    checkpoint = Path(args.ppm_checkpoint) if args.ppm_checkpoint else Path(args.checkpoints) / setting / "checkpoint.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing PPM checkpoint: {checkpoint}")

    exp = Exp_Main(args)
    exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device))
    train_data, train_loader = data_provider(args, "train")
    val_data, val_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.calibration_mode == "pit_warp":
        fit_loader = train_loader if args.pit_warp_fit_split == "train" else val_loader
        calibrator = fit_pit_warp(exp.model, fit_loader, exp.device, args)
        preds, trues = collect_pit_warp_samples(calibrator, exp.model, test_loader, exp.device, args)
    else:
        sparc_ckpt = train_sparc(exp.model, train_loader, val_loader, exp.device, args, args.output_dir)
        calibrator = SampleSPARCCalibrator(
            prediction_length=args.pred_len,
            target_dim=args.c_out,
            hidden_dim=args.sparc_hidden_dim,
            mean_correction=args.mean_correction,
            max_mean_delta=args.max_mean_delta,
            max_log_scale_delta=args.max_log_scale_delta,
            calibration_strength=args.calibration_strength,
            nll_weight=args.sparc_nll_weight,
            quantile_weight=args.sparc_quantile_weight,
            pit_weight=args.sparc_pit_weight,
            pit_temperature=args.sparc_pit_temperature,
            use_perturbation_features=args.use_perturbation_features,
            bounded_updates=args.bounded_updates,
            reconstruction_mode=args.reconstruction_mode,
        ).to(exp.device)
        calibrator.load_state_dict(torch.load(sparc_ckpt, map_location=exp.device)["state_dict"])
        if args.calibration_mode == "sparc_head_pit_warp":
            fit_loader = train_loader if args.pit_warp_fit_split == "train" else val_loader
            fit_pit = collect_sparc_pit_values(calibrator, exp.model, fit_loader, exp.device, args)
            pit_warp = fit_pit_warp_from_pit(fit_pit, args, "SPARC+PIT-WARP")
            preds, trues = collect_sparc_samples(calibrator, exp.model, test_loader, exp.device, args)
            preds = [pit_warp.warp_samples(pred) for pred in preds]
        else:
            preds, trues = collect_sparc_samples(calibrator, exp.model, test_loader, exp.device, args)
    sparc_metrics = evaluate_ppm_samples(preds, trues)
    ppm_metrics = {"CRPS": args.ppm_crps, "QICE": args.ppm_qice, "MSE": args.ppm_mse, "MAE": args.ppm_mae}
    comparison = write_comparison(args.output_dir, ppm_metrics, sparc_metrics)

    print("total_samples", sparc_metrics["total_samples"])
    print("NT metrc: CRPS:{:.4f}".format(sparc_metrics["CRPS"]))
    print("NT metrc: mse:{:.4f}, mae:{:.4f} ".format(sparc_metrics["MSE"], sparc_metrics["MAE"]))
    print("QICE bin count:", sparc_metrics["QICE_bins"])
    print("CARD metrc: QICE:{:.4f}%".format(sparc_metrics["QICE"]))
    print(f"[DONE] comparison={comparison}")


if __name__ == "__main__":
    main()
