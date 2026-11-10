#!/usr/bin/env python3
"""Evaluate PPM PIT-only and SPARC-only from existing paper checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time


REPO = Path("/home/wyj/ppm_original_repro_20260707/PPM")
PYTHON = "/home/wyj/miniconda3/envs/k2vae/bin/python"
SOURCE_BASE = Path("/export/wyj/ppm_original_repro_20260707")
OUTPUT = SOURCE_BASE / "results_ppm_component_ablation_3seed_7datasets_20260719"
SEEDS = (2021, 2, 3)
DATASETS = ("ECL", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "traffic", "weather")
VARIANTS = ("pit_only", "sparc_only")


CONFIGS = {
    "ETTh1": dict(root_path="./dataset/ETT-small/", data_path="ETTh1.csv", model_id="ETTh1_MLP_K100", data="ETTh1", enc_in=7, d_model=256, e_layers=2, d_ff=512, num_workers=0, batch_size=64, test_batch_size=32, learning_rate=0.0001),
    "ETTh2": dict(root_path="./dataset/ETT-small/", data_path="ETTh2.csv", model_id="ETTh2_MLP_K100", data="ETTh2", enc_in=7, d_model=256, e_layers=2, d_ff=512, num_workers=0, batch_size=64, test_batch_size=32, learning_rate=0.0001),
    "ETTm1": dict(root_path="./dataset/ETT-small/", data_path="ETTm1.csv", model_id="ETTm1_MLP_K100", data="ETTm1", enc_in=7, d_model=256, e_layers=2, d_ff=512, num_workers=4, batch_size=128, test_batch_size=32, learning_rate=0.0001),
    "ETTm2": dict(root_path="./dataset/ETT-small/", data_path="ETTm2.csv", model_id="ETTm2_MLP_K100", data="ETTm2", enc_in=7, d_model=256, e_layers=2, d_ff=512, num_workers=0, batch_size=128, test_batch_size=64, learning_rate=0.0001),
    "ECL": dict(root_path="./dataset/", data_path="electricity.csv", model_id="ECL_MLP_K100", data="custom", enc_in=321, d_model=256, e_layers=2, d_ff=256, num_workers=0, batch_size=16, test_batch_size=2, learning_rate=0.0005),
    "traffic": dict(root_path="./dataset/", data_path="traffic.csv", model_id="traffic_MLP_K100", data="custom", enc_in=862, d_model=256, e_layers=2, d_ff=256, num_workers=0, batch_size=16, test_batch_size=8, learning_rate=0.001),
    "weather": dict(root_path="./dataset/", data_path="weather.csv", model_id="weather_MLP_K100", data="custom", enc_in=21, d_model=512, e_layers=3, d_ff=1024, num_workers=4, batch_size=128, test_batch_size=64, learning_rate=0.0001),
}


def source_dir(seed: int, dataset: str) -> Path:
    run_root = (
        "results_ppm_sparc_pit_stats100_seed2021_20260719"
        if seed == 2021
        else "results_ppm_sparc_pit_stats100_seed23_20260719"
    )
    return SOURCE_BASE / run_root / f"seed{seed}_{dataset}_96_192_high_s060_l060_p80_ws100_k81"


def baseline_metrics(source: Path) -> dict[str, str]:
    with (source / "comparison_ppm_sparc.csv").open() as handle:
        row = next(row for row in csv.DictReader(handle) if row["method"] == "PPM")
    return row


def build_tasks() -> list[tuple[int, str, str]]:
    return [
        (seed, dataset, variant)
        for seed in SEEDS
        for dataset in DATASETS
        for variant in VARIANTS
    ]


def shard_tasks(
    tasks: list[tuple[int, str, str]], worker_index: int, worker_count: int
) -> list[tuple[int, str, str]]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    return tasks[worker_index::worker_count]


def gpu_memory_free_mb(gpu: str) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    if not values:
        raise RuntimeError(f"could not read free memory for GPU {gpu}")
    return values[0]


def wait_for_gpu(gpu: str, min_free_memory_mb: int, poll_seconds: int) -> None:
    while True:
        free_mb = gpu_memory_free_mb(gpu)
        if free_mb >= min_free_memory_mb:
            return
        print(
            f"[WAIT] GPU {gpu} has {free_mb} MiB free; "
            f"need {min_free_memory_mb} MiB",
            flush=True,
        )
        time.sleep(poll_seconds)


def command(seed: int, dataset: str, variant: str, source: Path, out_dir: Path) -> list[str]:
    cfg = CONFIGS[dataset]
    metrics = baseline_metrics(source)
    calibration_mode = "pit_warp" if variant == "pit_only" else "sparc_head"
    reuse = "false" if variant == "pit_only" else "true"
    args = [
        PYTHON,
        "-u",
        "run_sparc_components.py",
        "--gpu", "0",
        "--root_path", str(cfg["root_path"]),
        "--data_path", str(cfg["data_path"]),
        "--model_id", str(cfg["model_id"]),
        "--model", "MLP",
        "--data", str(cfg["data"]),
        "--data_name", dataset,
        "--features", "M",
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", "192",
        "--enc_in", str(cfg["enc_in"]),
        "--dec_in", str(cfg["enc_in"]),
        "--c_out", str(cfg["enc_in"]),
        "--d_model", str(cfg["d_model"]),
        "--e_layers", str(cfg["e_layers"]),
        "--d_ff", str(cfg["d_ff"]),
        "--num_workers", str(cfg["num_workers"]),
        "--train_epochs", "30",
        "--batch_size", str(cfg["batch_size"]),
        "--test_batch_size", str(cfg["test_batch_size"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--des", "Exp",
        "--var", "0.3",
        "--tau", "0.1",
        "--K", "100",
        "--seed", str(seed),
        "--ppm_checkpoint", str(source / "ppm_checkpoint.pth"),
        "--stats_num_samples", "100",
        "--sample_chunk_size", "100",
        "--calibration_mode", calibration_mode,
        "--pit_warp_fit_split", "train",
        "--pit_warp_strength", "1.0",
        "--pit_warp_knots", "81",
        "--sparc_epochs", "20",
        "--sparc_patience", "5",
        "--sparc_learning_rate", "0.001",
        "--limit_train_batches", "100",
        "--limit_val_batches", "0",
        "--mean_correction", "false",
        "--calibration_strength", "0.6",
        "--max_mean_delta", "0.0",
        "--max_log_scale_delta", "0.6",
        "--sparc_nll_weight", "0.5",
        "--sparc_quantile_weight", "0.5",
        "--sparc_pit_weight", "80.0",
        "--sparc_pit_temperature", "0.01",
        "--ppm_crps", metrics["CRPS"],
        "--ppm_qice", metrics["QICE"],
        "--ppm_mse", metrics["MSE"],
        "--ppm_mae", metrics["MAE"],
        "--limit_pit_fit_batches", "100",
        "--max_pit_fit_points", "2000000",
        "--reuse_sparc_ckpt", reuse,
        "--output_dir", str(out_dir),
    ]
    return args


def write_status(records: list[dict[str, object]], status_path: Path) -> None:
    status_path.write_text(json.dumps(records, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    status_path = OUTPUT / f"status_worker_{args.worker_index}_of_{args.worker_count}.json"
    records: list[dict[str, object]] = []
    tasks = shard_tasks(build_tasks(), args.worker_index, args.worker_count)
    print(
        f"[WORKER] index={args.worker_index}/{args.worker_count} gpu={args.gpu} "
        f"tasks={len(tasks)} min_free_memory_mb={args.min_free_memory_mb}",
        flush=True,
    )
    for seed, dataset, variant in tasks:
        source = source_dir(seed, dataset)
        for required in ("comparison_ppm_sparc.csv", "ppm_checkpoint.pth", "sparc_ckpt.pt"):
            if not (source / required).is_file():
                raise FileNotFoundError(source / required)
        out_dir = OUTPUT / variant / f"seed{seed}_{dataset}_96_192_{variant}"
        comparison = out_dir / "comparison_ppm_sparc.csv"
        record: dict[str, object] = {"seed": seed, "dataset": dataset, "variant": variant}
        if comparison.is_file() and comparison.stat().st_size > 0:
            record["status"] = "complete"
            records.append(record)
            write_status(records, status_path)
            print(f"[SKIP] seed={seed} dataset={dataset} variant={variant}", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        if variant == "sparc_only":
            link = out_dir / "sparc_ckpt.pt"
            if not link.exists():
                link.symlink_to(source / "sparc_ckpt.pt")

        wait_for_gpu(args.gpu, args.min_free_memory_mb, args.poll_seconds)
        log_path = OUTPUT / "logs" / f"seed{seed}_{dataset}_{variant}.log"
        print(f"[START] seed={seed} dataset={dataset} variant={variant}", flush=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["PYTHONUNBUFFERED"] = "1"
        started = time.time()
        with log_path.open("a") as log:
            result = subprocess.run(
                command(seed, dataset, variant, source, out_dir),
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["returncode"] = result.returncode
        record["elapsed_seconds"] = round(time.time() - started, 1)
        record["status"] = "complete" if result.returncode == 0 and comparison.is_file() else "failed"
        records.append(record)
        write_status(records, status_path)
        print(
            f"[DONE] seed={seed} dataset={dataset} variant={variant} "
            f"status={record['status']} elapsed={record['elapsed_seconds']}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
