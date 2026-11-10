import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_dacu_k2vae_checkpoints.py"


def load_sweep_module():
    spec = importlib.util.spec_from_file_location("sweep_dacu_k2vae_checkpoints", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_result(path, crps, nd):
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "test_CRPS", "test_ND"])
        writer.writeheader()
        writer.writerow({"horizon": "720", "test_CRPS": str(crps), "test_ND": str(nd)})


def test_checkpoint_sweep_summary_marks_joint_paper_wins(tmp_path):
    write_result(
        tmp_path / "weak_aug_weak_cal" / "pred_720" / "epoch_1" / "run" / "testctx_96_horizons_results.csv",
        0.312,
        0.401,
    )
    write_result(
        tmp_path / "weak_aug_weak_cal" / "pred_720" / "epoch_2" / "run" / "testctx_96_horizons_results.csv",
        0.313,
        0.395,
    )

    sweep = load_sweep_module()
    rows = sweep.collect_sweep_rows(tmp_path)

    assert rows[0]["variant"] == "weak_aug_weak_cal"
    assert rows[0]["horizon"] == 720
    assert rows[0]["epoch"] == 1
    assert rows[0]["beats_paper_k2vae"] is False
    assert rows[1]["epoch"] == 2
    assert rows[1]["delta_CRPS"] == -0.0010000000000000009
    assert rows[1]["delta_ND"] == -0.0010000000000000009
    assert rows[1]["beats_paper_k2vae"] is True
