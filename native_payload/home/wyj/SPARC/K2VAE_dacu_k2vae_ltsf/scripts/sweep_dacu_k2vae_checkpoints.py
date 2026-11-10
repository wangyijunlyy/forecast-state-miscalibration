import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


PAPER_ETTH1 = {
    96: (0.264, 0.336),
    192: (0.290, 0.372),
    336: (0.308, 0.394),
    720: (0.314, 0.396),
}


def parse_epoch(name: str):
    match = re.search(r"epoch[=_](\d+)", name)
    return int(match.group(1)) if match else None


def parse_checkpoint(path: Path):
    match = re.match(r"epoch=(\d+)-val_CRPS=([0-9]+(?:\.[0-9]+)?)\.ckpt$", path.name)
    if not match:
        return None
    return int(match.group(1)), float(match.group(2))


def parse_candidate(candidate: str):
    if "=" not in candidate:
        raise argparse.ArgumentTypeError("candidate must be LABEL=/path/to/run_dir")
    label, run_dir = candidate.split("=", 1)
    if not label or not run_dir:
        raise argparse.ArgumentTypeError("candidate must be LABEL=/path/to/run_dir")
    return label, Path(run_dir)


def read_result(path: Path):
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def infer_from_result_path(path: Path):
    parts = path.parts
    horizon = None
    epoch = None
    variant = None
    for index, part in enumerate(parts):
        if part.startswith("pred_"):
            horizon = int(part.replace("pred_", ""))
            if index > 0:
                variant = parts[index - 1]
        elif part.startswith("epoch_"):
            epoch = parse_epoch(part)
    if horizon is None or epoch is None or variant is None:
        raise ValueError(f"Cannot infer variant/horizon/epoch from {path}")
    return variant, horizon, epoch


def collect_sweep_rows(root: Path):
    rows = []
    for result_path in sorted(root.rglob("*horizons_results.csv")):
        variant, horizon, epoch = infer_from_result_path(result_path)
        result = read_result(result_path)
        paper_crps, paper_nd = PAPER_ETTH1[horizon]
        crps = float(result["test_CRPS"])
        nd = float(result["test_ND"])
        rows.append(
            {
                "variant": variant,
                "horizon": horizon,
                "epoch": epoch,
                "test_CRPS": crps,
                "paper_CRPS": paper_crps,
                "delta_CRPS": crps - paper_crps,
                "test_ND": nd,
                "paper_NMAE": paper_nd,
                "delta_ND": nd - paper_nd,
                "beats_paper_k2vae": crps < paper_crps and nd < paper_nd,
                "source": str(result_path),
            }
        )
    rows.sort(key=lambda row: (row["variant"], row["horizon"], row["epoch"]))
    return rows


def write_summary(rows, out_path: Path):
    fields = [
        "variant",
        "horizon",
        "epoch",
        "test_CRPS",
        "paper_CRPS",
        "delta_CRPS",
        "test_ND",
        "paper_NMAE",
        "delta_ND",
        "beats_paper_k2vae",
        "source",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def existing_result(out_dir: Path):
    return next(out_dir.rglob("*horizons_results.csv"), None) if out_dir.exists() else None


def checkpoint_list(run_dir: Path):
    ckpt_dir = run_dir / "ckpt"
    checkpoints = []
    for path in ckpt_dir.glob("epoch=*-val_CRPS=*.ckpt"):
        parsed = parse_checkpoint(path)
        if parsed:
            checkpoints.append((*parsed, path))
    return sorted(checkpoints)


def run_checkpoint(args, variant: str, run_dir: Path, epoch: int, ckpt: Path):
    out_dir = args.result_root / variant / f"pred_{args.horizon}" / f"epoch_{epoch}"
    if args.skip_existing and existing_result(out_dir):
        print(f"[skip] {variant} horizon={args.horizon} epoch={epoch}")
        return

    command = [
        str(args.python_bin),
        "run.py",
        "--config",
        str(args.config),
        "--seed_everything",
        str(args.seed),
        "--data.data_manager.init_args.path",
        str(args.data_dir),
        "--data.data_manager.init_args.split_val",
        "true",
        "--data.data_manager.init_args.dataset",
        "etth1",
        "--data.data_manager.init_args.context_length",
        str(args.context_length),
        "--data.data_manager.init_args.prediction_length",
        str(args.horizon),
        "--trainer.default_root_dir",
        str(out_dir),
        "--trainer.max_epochs",
        "1",
        "--trainer.num_sanity_val_steps",
        "0",
        "--model.load_from_ckpt",
        str(ckpt),
        "--model.num_samples",
        str(args.num_samples),
        "--model.val_num_samples",
        str(args.num_samples),
        "--model.forecaster.init_args.no_training",
        "true",
    ]
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"[run] {variant} horizon={args.horizon} epoch={epoch} ckpt={ckpt}")
    subprocess.run(command, check=True, env=env)


def run_sweep(args):
    for variant, run_dir in args.candidate:
        for epoch, _, ckpt in checkpoint_list(run_dir):
            run_checkpoint(args, variant, run_dir, epoch, ckpt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_candidate, default=[])
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--data-dir", type=Path, default=Path("./datasets"))
    parser.add_argument("--config", type=Path, default=Path("config/default/dacu_k2vae.yaml"))
    parser.add_argument("--python-bin", type=Path, default=Path("python"))
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if not args.summarize_only:
        if not args.candidate:
            parser.error("--candidate is required unless --summarize-only is set")
        run_sweep(args)

    rows = collect_sweep_rows(args.result_root)
    out_path = args.summary_out or args.result_root / "summary.csv"
    write_summary(rows, out_path)
    print(f"wrote {len(rows)} rows to {out_path}")
    print("variant,horizon,epoch,test_CRPS,delta_CRPS,test_ND,delta_ND,beats_paper_k2vae")
    for row in rows:
        print(
            f"{row['variant']},{row['horizon']},{row['epoch']},"
            f"{row['test_CRPS']:.6f},{row['delta_CRPS']:.6f},"
            f"{row['test_ND']:.6f},{row['delta_ND']:.6f},{row['beats_paper_k2vae']}"
        )


if __name__ == "__main__":
    main()
