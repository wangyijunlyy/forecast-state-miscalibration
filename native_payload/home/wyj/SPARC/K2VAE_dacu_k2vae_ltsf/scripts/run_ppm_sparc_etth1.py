#!/usr/bin/env python3
"""Run PPM and SPARC-PPM ETTh1 experiments with K2VAE-compatible metrics."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparc_adapters.calibration import SampleSPARCCalibrator
from sparc_adapters.metrics import evaluate_samples_like_k2vae, write_horizons_results_csv


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


def effective_stats_num_samples(args: argparse.Namespace) -> int:
    """Return SPARC's stats sample count without changing evaluation sample count."""

    stats_num_samples = getattr(args, "stats_num_samples", 0)
    return int(stats_num_samples) if int(stats_num_samples) > 0 else int(args.num_samples)


def build_ppm_args(args: argparse.Namespace, ppm_root: Path, result_root: Path) -> Namespace:
    return Namespace(
        is_training=1,
        model_id=f"ETTh1_MLP_K{args.num_samples}",
        model="MLP",
        data="ETTh1",
        data_name="ETTh1",
        root_path=str(args.data_root),
        data_path="ETTh1.csv",
        features="M",
        target="OT",
        freq="h",
        checkpoints=str(result_root / "ppm_checkpoints"),
        pretrain_checkpoints=str(result_root / "ppm_pretrain_checkpoints"),
        seq_len=args.seq_len,
        label_len=0,
        pred_len=args.pred_len,
        enc_in=args.target_dim,
        dec_in=args.target_dim,
        c_out=args.target_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_model_f=args.d_model,
        d_ff_f=args.d_ff,
        dropout=0.05,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        do_predict=False,
        num_workers=0,
        itr=1,
        train_epochs=args.max_epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        des="Exp",
        loss="mse",
        lradj="type1",
        use_amp=False,
        cond_learning_rate=0.0001,
        std_learning_rate=0.0005,
        use_gpu=torch.cuda.is_available(),
        gpu=args.gpu,
        use_multi_gpu=False,
        devices=str(args.gpu),
        seed=args.seed,
        K=args.num_samples,
        tau=args.tau,
        var=args.var,
        nu=5,
        ppm_root=str(ppm_root),
    )


def ppm_setting(ppm_args: Namespace) -> str:
    return "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_df{}_{}_{}".format(
        ppm_args.model_id,
        ppm_args.model,
        ppm_args.data_name,
        ppm_args.features,
        ppm_args.seq_len,
        ppm_args.label_len,
        ppm_args.pred_len,
        ppm_args.d_model,
        ppm_args.d_ff,
        ppm_args.des,
        0,
    )


def load_ppm_modules(ppm_root: Path):
    if str(ppm_root) not in sys.path:
        sys.path.insert(0, str(ppm_root))
    from data_provider.data_factory import data_provider
    from model_ppm.exp.exp_main import Exp_Main

    return data_provider, Exp_Main


def batch_to_samples(model: torch.nn.Module, batch_x: torch.Tensor, num_samples: int, pred_len: int, features: str) -> torch.Tensor:
    samples = model(batch_x, num_samples)
    samples = samples[:, :, -pred_len:, :]
    f_dim = -1 if features == "MS" else 0
    return samples[:, :, :, f_dim:]


def collect_samples(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    ppm_args: Namespace,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_samples = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            samples = batch_to_samples(model, batch_x, num_samples, ppm_args.pred_len, ppm_args.features)
            f_dim = -1 if ppm_args.features == "MS" else 0
            target = batch_y[:, -ppm_args.pred_len :, f_dim:]
            all_samples.append(samples.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
    return np.concatenate(all_targets, axis=0), np.concatenate(all_samples, axis=0)


def write_npz(output_dir: Path, targets: np.ndarray, samples: np.ndarray, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    np.savez_compressed(path, targets=targets, samples=samples)
    return path


def evaluate_and_save(
    output_dir: Path,
    targets: np.ndarray,
    samples: np.ndarray,
    model_args: str,
    horizon: int,
    save_samples_npz: bool = False,
) -> dict[str, float]:
    metrics = evaluate_samples_like_k2vae(targets, samples, quantiles_num=10)
    write_horizons_results_csv(output_dir, metrics, model_args=model_args, horizon=horizon)
    if save_samples_npz:
        write_npz(output_dir, targets, samples, "samples_targets.npz")
    return metrics


def augment_context(x: torch.Tensor, jitter_std: float = 0.01, scaling_std: float = 0.05, time_mask_ratio: float = 0.05) -> torch.Tensor:
    y = x.clone()
    if jitter_std > 0:
        y = y + torch.randn_like(y) * jitter_std
    if scaling_std > 0:
        scale = 1.0 + torch.randn(y.shape[0], 1, y.shape[2], device=y.device, dtype=y.dtype) * scaling_std
        y = y * scale.clamp_min(1e-3)
    if time_mask_ratio > 0:
        mask = torch.rand(y.shape[0], y.shape[1], 1, device=y.device) < time_mask_ratio
        fill = y.mean(dim=1, keepdim=True).detach()
        y = torch.where(mask, fill.expand_as(y), y)
    return y


def train_sparc_ppm(
    ppm_model: torch.nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    ppm_args: Namespace,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    calibrator = SampleSPARCCalibrator(
        prediction_length=ppm_args.pred_len,
        target_dim=ppm_args.c_out,
        hidden_dim=args.sparc_hidden_dim,
        calibration_strength=args.calibration_strength,
        mean_correction=args.mean_correction,
        max_mean_delta=args.max_mean_delta,
        max_log_scale_delta=args.max_log_scale_delta,
        nll_weight=args.nll_weight,
        quantile_weight=args.quantile_weight,
        coverage_weight=args.coverage_weight,
        ranking_weight=args.ranking_weight,
        pit_weight=args.pit_weight,
        pit_temperature=args.pit_temperature,
    ).to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=args.sparc_learning_rate)
    best_loss = float("inf")
    bad_epochs = 0
    best_path = output_dir / "sparc_ckpt.pt"

    ppm_model.eval()
    for parameter in ppm_model.parameters():
        parameter.requires_grad = False

    stat_k = effective_stats_num_samples(args)
    for epoch in range(1, args.max_epochs + 1):
        calibrator.train()
        train_losses = []
        for step, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(train_loader, start=1):
            if args.limit_train_batches > 0 and step > args.limit_train_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            f_dim = -1 if ppm_args.features == "MS" else 0
            target = batch_y[:, -ppm_args.pred_len :, f_dim:]
            with torch.no_grad():
                base = batch_to_samples(ppm_model, batch_x, stat_k, ppm_args.pred_len, ppm_args.features)
                aug = batch_to_samples(
                    ppm_model,
                    augment_context(batch_x),
                    stat_k,
                    ppm_args.pred_len,
                    ppm_args.features,
                )
            loss = calibrator.loss(base, aug, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrator.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = validate_sparc_ppm(calibrator, ppm_model, val_loader, device, ppm_args, args)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"[SPARC] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": calibrator.state_dict(),
                    "prediction_length": ppm_args.pred_len,
                    "target_dim": ppm_args.c_out,
                    "best_val_loss": best_loss,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[SPARC] early stopping at epoch={epoch}", flush=True)
                break
    return best_path


def validate_sparc_ppm(
    calibrator: SampleSPARCCalibrator,
    ppm_model: torch.nn.Module,
    loader,
    device: torch.device,
    ppm_args: Namespace,
    args: argparse.Namespace,
) -> float:
    calibrator.eval()
    losses = []
    stat_k = effective_stats_num_samples(args)
    with torch.no_grad():
        for step, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader, start=1):
            if args.limit_val_batches > 0 and step > args.limit_val_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            f_dim = -1 if ppm_args.features == "MS" else 0
            target = batch_y[:, -ppm_args.pred_len :, f_dim:]
            base = batch_to_samples(ppm_model, batch_x, stat_k, ppm_args.pred_len, ppm_args.features)
            aug = batch_to_samples(ppm_model, augment_context(batch_x), stat_k, ppm_args.pred_len, ppm_args.features)
            losses.append(float(calibrator.loss(base, aug, target).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def collect_sparc_samples(
    calibrator: SampleSPARCCalibrator,
    ppm_model: torch.nn.Module,
    loader,
    device: torch.device,
    ppm_args: Namespace,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    calibrator.eval()
    ppm_model.eval()
    all_samples = []
    all_targets = []
    stat_k = effective_stats_num_samples(args)
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            f_dim = -1 if ppm_args.features == "MS" else 0
            target = batch_y[:, -ppm_args.pred_len :, f_dim:]
            output_samples = batch_to_samples(ppm_model, batch_x, args.num_samples, ppm_args.pred_len, ppm_args.features)
            if stat_k == args.num_samples:
                base_stats = output_samples
            else:
                base_stats = batch_to_samples(ppm_model, batch_x, stat_k, ppm_args.pred_len, ppm_args.features)
            aug_stats = batch_to_samples(ppm_model, augment_context(batch_x), stat_k, ppm_args.pred_len, ppm_args.features)
            samples = calibrator.calibrate_samples_with_stats(output_samples, base_stats, aug_stats)
            all_samples.append(samples.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
    return np.concatenate(all_targets, axis=0), np.concatenate(all_samples, axis=0)


def write_comparison(
    result_root: Path,
    pred_len: int,
    ppm_metrics: dict[str, float],
    sparc_metrics: dict[str, float],
    sparc_method_label: str = "SPARC-PPM",
    comparison_tag: str = "",
) -> Path:
    tag = f"_{comparison_tag}" if comparison_tag else ""
    output = result_root / f"comparison{tag}_pred_{pred_len}.csv"
    fields = ["method", "horizon", "CRPS", "QICE", "ND", "NRMSE", "CRPS-Sum", "QICE-Sum", "ND-Sum", "NRMSE-Sum"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, metrics in [("PPM", ppm_metrics), (sparc_method_label, sparc_metrics)]:
            row = {"method": method, "horizon": pred_len}
            for field in fields[2:]:
                row[field] = metrics.get(field, "")
            writer.writerow(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppm-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--target-dim", type=int, default=7)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--stats-num-samples",
        type=int,
        default=0,
        help="SPARC-only sample count for estimating mean/std; 0 reuses --num-samples.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sparc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--var", type=float, default=0.3)
    parser.add_argument("--calibration-strength", type=float, default=0.8)
    parser.add_argument("--mean-correction", type=str2bool, default=True)
    parser.add_argument("--max-mean-delta", type=float, default=0.25)
    parser.add_argument("--max-log-scale-delta", type=float, default=1.0)
    parser.add_argument("--sparc-hidden-dim", type=int, default=128)
    parser.add_argument("--nll-weight", type=float, default=1.0)
    parser.add_argument("--quantile-weight", type=float, default=0.2)
    parser.add_argument("--coverage-weight", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=0.05)
    parser.add_argument("--pit-weight", type=float, default=0.0)
    parser.add_argument("--pit-temperature", type=float, default=0.02)
    parser.add_argument("--limit-train-batches", type=int, default=100)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--skip-ppm-train", action="store_true")
    parser.add_argument("--ppm-checkpoint-path", type=Path, default=None)
    parser.add_argument("--sparc-dir-name", default="sparc_ppm")
    parser.add_argument("--sparc-method-label", default="SPARC-PPM")
    parser.add_argument("--save-samples-npz", action="store_true", help="Save large targets/samples NPZ files")
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    result_root = args.result_root / f"pred_{args.pred_len}"
    result_root.mkdir(parents=True, exist_ok=True)
    data_provider, Exp_Main = load_ppm_modules(args.ppm_root)
    ppm_args = build_ppm_args(args, args.ppm_root, result_root)
    setting = ppm_setting(ppm_args)

    metadata = {"setting": setting, "args": vars(args), "ppm_args": vars(ppm_args)}
    with (result_root / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    exp = Exp_Main(ppm_args)
    if not args.skip_ppm_train:
        print(f"[PPM] training setting={setting}", flush=True)
        exp.train(setting)
    ckpt_path = args.ppm_checkpoint_path or (Path(ppm_args.checkpoints) / setting / "checkpoint.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing PPM checkpoint: {ckpt_path}")
    print(f"[PPM] loading checkpoint {ckpt_path}", flush=True)
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))

    _test_data, test_loader = data_provider(ppm_args, "test")
    targets, ppm_samples = collect_samples(exp.model, test_loader, exp.device, ppm_args, args.num_samples)
    ppm_dir = result_root / "ppm"
    ppm_metrics = evaluate_and_save(
        ppm_dir,
        targets,
        ppm_samples,
        "PPM",
        args.pred_len,
        save_samples_npz=args.save_samples_npz,
    )
    print(f"[PPM] CRPS={ppm_metrics.get('CRPS'):.6f} QICE={ppm_metrics.get('QICE'):.6f}", flush=True)

    _train_data, train_loader = data_provider(ppm_args, "train")
    _val_data, val_loader = data_provider(ppm_args, "val")
    sparc_dir = result_root / args.sparc_dir_name
    sparc_ckpt = train_sparc_ppm(exp.model, train_loader, val_loader, exp.device, ppm_args, sparc_dir, args)
    calibrator = SampleSPARCCalibrator(
        prediction_length=ppm_args.pred_len,
        target_dim=ppm_args.c_out,
        hidden_dim=args.sparc_hidden_dim,
        calibration_strength=args.calibration_strength,
        mean_correction=args.mean_correction,
        max_mean_delta=args.max_mean_delta,
        max_log_scale_delta=args.max_log_scale_delta,
        nll_weight=args.nll_weight,
        quantile_weight=args.quantile_weight,
        coverage_weight=args.coverage_weight,
        ranking_weight=args.ranking_weight,
        pit_weight=args.pit_weight,
        pit_temperature=args.pit_temperature,
    ).to(exp.device)
    checkpoint = torch.load(sparc_ckpt, map_location=exp.device)
    calibrator.load_state_dict(checkpoint["state_dict"])
    targets, sparc_samples = collect_sparc_samples(calibrator, exp.model, test_loader, exp.device, ppm_args, args)
    sparc_metrics = evaluate_and_save(
        sparc_dir,
        targets,
        sparc_samples,
        args.sparc_method_label,
        args.pred_len,
        save_samples_npz=args.save_samples_npz,
    )
    print(f"[{args.sparc_method_label}] CRPS={sparc_metrics.get('CRPS'):.6f} QICE={sparc_metrics.get('QICE'):.6f}", flush=True)

    comparison = write_comparison(
        result_root,
        args.pred_len,
        ppm_metrics,
        sparc_metrics,
        args.sparc_method_label,
        args.sparc_dir_name,
    )
    print(f"[DONE] comparison={comparison}", flush=True)


if __name__ == "__main__":
    main()
