#!/usr/bin/env python
import argparse
import csv
import math
import os
import re
import traceback
from pathlib import Path

import torch

from probts.data import ProbTSDataModule
from probts.data.data_manager import DataManager
from probts.data.data_wrapper import ProbTSBatchData
from probts.model.forecast_module import ProbTSForecastModule


ORDER = {
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


def patch_revin_device_safe():
    try:
        from probts.model.nn.prob.k2VAE.RevIN import RevIN
    except Exception:
        return

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.to(x.device)
            x = x + self.affine_bias.to(x.device)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.to(x.device)
            x = x / (self.affine_weight.to(x.device) + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

    RevIN._normalize = _normalize
    RevIN._denormalize = _denormalize


def infer_key_from_run(run_dir: Path):
    match = re.match(r"(.+?)_FastPosthoc.*?_CTX(\d+)_PRED(\d+)_", run_dir.name)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def is_exact_fast_posthoc_v2(path: Path) -> bool:
    parts = path.parts
    return "fast_posthoc_v2" in parts and not any(
        part.startswith("fast_posthoc_v2_") for part in parts
    )


def discover_runs(search_roots):
    latest = {}
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for ckpt_dir in root.rglob("ckpt"):
            if not ckpt_dir.is_dir() or not is_exact_fast_posthoc_v2(ckpt_dir):
                continue
            key = infer_key_from_run(ckpt_dir.parent)
            if key is None:
                continue
            dataset, context_length, prediction_length = key
            run_mtime = max((p.stat().st_mtime for p in ckpt_dir.glob("*.ckpt")), default=ckpt_dir.stat().st_mtime)
            latest_key = (dataset, prediction_length)
            if latest_key not in latest or run_mtime > latest[latest_key]["run_mtime"]:
                latest[latest_key] = {
                    "dataset": dataset,
                    "context_length": context_length,
                    "prediction_length": prediction_length,
                    "ckpt_dir": ckpt_dir,
                    "run_dir": ckpt_dir.parent,
                    "run_mtime": run_mtime,
                }
    return latest


def best_ckpt(ckpt_dir: Path):
    best_path = None
    best_val = None
    best_epoch = None
    for ckpt in ckpt_dir.glob("epoch=*-val_CRPS=*.ckpt"):
        match = re.match(r"epoch=(\d+)-val_CRPS=([0-9.]+)\.ckpt$", ckpt.name)
        if not match:
            continue
        epoch = int(match.group(1))
        val = float(match.group(2))
        if best_val is None or val < best_val:
            best_path = ckpt
            best_val = val
            best_epoch = epoch
    return best_path, best_epoch, best_val


def load_k2_summary(path: Path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["dataset"], int(row["horizon"]))] = row
    return rows


def qice_per_sequence(targets: torch.Tensor, forecasts: torch.Tensor, quantiles_num: int):
    quantiles = torch.arange(quantiles_num, device=targets.device, dtype=targets.dtype) / quantiles_num
    quantiles = quantiles[1:]
    reduce_dims = tuple(range(1, targets.dim()))
    coverages = []
    for q in quantiles:
        q_forecast = torch.quantile(forecasts, float(q), dim=1)
        coverage = torch.mean((targets < q_forecast).to(targets.dtype), dim=reduce_dims)
        coverages.append(coverage)
    coverages = torch.stack(coverages, dim=1)
    zeros = torch.zeros((targets.shape[0], 1), device=targets.device, dtype=targets.dtype)
    ones = torch.ones((targets.shape[0], 1), device=targets.device, dtype=targets.dtype)
    observed = torch.diff(torch.cat([zeros, coverages, ones], dim=1), dim=1)
    optimal = 1.0 / (len(quantiles) + 1)
    return torch.mean(torch.abs(observed - optimal), dim=1)


def force_tensor_attrs_to_device(module, device):
    """Move non-registered tensor attributes used by some backbone helpers."""
    for submodule in module.modules():
        for name, param in list(submodule._parameters.items()):
            if param is not None:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
        for name, buffer in list(submodule._buffers.items()):
            if buffer is not None:
                submodule._buffers[name] = buffer.to(device)
        for name, value in list(vars(submodule).items()):
            if isinstance(value, torch.Tensor):
                setattr(submodule, name, value.to(device))


def evaluate_qice(task, args, device):
    dataset = task["dataset"]
    context_length = task["context_length"]
    prediction_length = task["prediction_length"]
    ckpt_path = task["best_ckpt"]

    data_manager = DataManager(
        dataset=dataset,
        path=args.data_path,
        context_length=context_length,
        prediction_length=prediction_length,
        split_val=True,
        scaler="standard",
    )
    datamodule = ProbTSDataModule(
        data_manager=data_manager,
        batch_size=args.test_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    model = ProbTSForecastModule.load_from_checkpoint(
        str(ckpt_path),
        scaler=data_manager.scaler,
        context_length=data_manager.context_length,
        target_dim=data_manager.target_dim,
        freq=data_manager.freq,
        prediction_length=data_manager.prediction_length,
        lags_list=data_manager.lags_list,
        time_feat_dim=data_manager.time_feat_dim,
        sampling_weight_scheme="none",
    )
    model.to(device)
    model.eval()
    force_tensor_attrs_to_device(model, device)

    qice_values = []
    qice_sum_values = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(datamodule.test_dataloader()):
            force_tensor_attrs_to_device(model, device)
            batch_data = ProbTSBatchData(batch, device)
            targets = batch_data.future_target_cdf
            past = batch_data.past_target_cdf
            pred_len = targets.shape[1]

            batch_data.past_target_cdf = data_manager.scaler.transform(past)
            forecasts = model.forecaster.forecast(batch_data, model.num_samples)[:, :, :pred_len]
            forecasts = data_manager.scaler.inverse_transform(forecasts)

            qice_values.append(qice_per_sequence(targets, forecasts, args.quantiles_num).detach().cpu())
            qice_sum_values.append(
                qice_per_sequence(
                    targets.sum(dim=-1),
                    forecasts.sum(dim=-1),
                    args.quantiles_num,
                ).detach().cpu()
            )

            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break

    qice = torch.cat(qice_values).mean().item()
    qice_sum = torch.cat(qice_sum_values).mean().item()
    return qice, qice_sum


def write_rows(path: Path, rows):
    fields = [
        "dataset",
        "horizon",
        "context_length",
        "fast_posthoc_qice",
        "fast_posthoc_qice_sum",
        "k2vae_qice",
        "delta_qice",
        "rel_qice_pct",
        "best_epoch",
        "best_val_CRPS",
        "best_ckpt",
        "run_dir",
        "status",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    patch_revin_device_safe()
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--k2-summary", default="/home/wyj/K2VAE/results_k2vae_ltsf_reproduce_es20_p5_20260617/summary_partial.csv")
    parser.add_argument("--data-path", default="./datasets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantiles-num", type=int, default=20)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    if not args.search_root:
        args.search_root = [
            "/export/wyj/K2VAE_dacu_k2vae_ltsf",
            "/home/wyj/K2VAE_dacu_k2vae_ltsf",
        ]

    latest = discover_runs(args.search_root)
    k2 = load_k2_summary(Path(args.k2_summary))
    tasks = []
    for key, run in latest.items():
        ckpt, epoch, val = best_ckpt(run["ckpt_dir"])
        if ckpt is None:
            continue
        dataset, horizon = key
        task = dict(run)
        task.update({"best_ckpt": ckpt, "best_epoch": epoch, "best_val_CRPS": val})
        tasks.append(task)

    tasks.sort(key=lambda t: (ORDER.get(t["dataset"], 99), t["dataset"], t["prediction_length"]))
    tasks = [task for idx, task in enumerate(tasks) if idx % args.num_shards == args.shard_index]

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for task in tasks:
        dataset = task["dataset"]
        horizon = task["prediction_length"]
        k2_row = k2.get((dataset, horizon), {})
        k2_qice = k2_row.get("QICE", "")
        print(f"[qice] start dataset={dataset} horizon={horizon} ckpt={task['best_ckpt']}", flush=True)
        row = {
            "dataset": dataset,
            "horizon": horizon,
            "context_length": task["context_length"],
            "fast_posthoc_qice": "",
            "fast_posthoc_qice_sum": "",
            "k2vae_qice": k2_qice,
            "delta_qice": "",
            "rel_qice_pct": "",
            "best_epoch": task["best_epoch"],
            "best_val_CRPS": task["best_val_CRPS"],
            "best_ckpt": str(task["best_ckpt"]),
            "run_dir": str(task["run_dir"]),
            "status": "error",
            "error": "",
        }
        try:
            qice, qice_sum = evaluate_qice(task, args, device)
            row["fast_posthoc_qice"] = qice
            row["fast_posthoc_qice_sum"] = qice_sum
            if k2_qice not in ("", None):
                k2_value = float(k2_qice)
                row["delta_qice"] = qice - k2_value
                row["rel_qice_pct"] = (qice - k2_value) / k2_value * 100 if k2_value else math.nan
            row["status"] = "ok"
            print(f"[qice] done dataset={dataset} horizon={horizon} qice={qice:.6f}", flush=True)
        except Exception as exc:
            row["error"] = repr(exc)
            traceback.print_exc()
            print(f"[qice] error dataset={dataset} horizon={horizon}: {exc!r}", flush=True)
        rows.append(row)
        write_rows(Path(args.output), rows)

    write_rows(Path(args.output), rows)


if __name__ == "__main__":
    main()
