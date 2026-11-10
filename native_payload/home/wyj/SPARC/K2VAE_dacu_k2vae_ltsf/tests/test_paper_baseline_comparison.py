import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compare_dacu_to_paper_baselines(tmp_path):
    result_dir = tmp_path / "dacu" / "patchtst" / "pred_96" / "ETTh1_CTX96_PRED96" / "version_0"
    result_dir.mkdir(parents=True)
    result_file = result_dir / "horizons_results.csv"
    with result_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "test_CRPS", "test_ND"])
        writer.writeheader()
        writer.writerow({"horizon": "96", "test_CRPS": "0.300", "test_ND": "0.390"})

    out_file = tmp_path / "summary.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_dacu_to_paper_baselines.py"),
            str(tmp_path),
            "--out",
            str(out_file),
        ],
        check=True,
    )

    rows = list(csv.DictReader(out_file.open()))
    patchtst_96 = next(row for row in rows if row["model"] == "patchtst" and row["horizon"] == "96")

    assert patchtst_96["paper_CRPS"] == "0.312"
    assert patchtst_96["paper_NMAE"] == "0.407"
    assert patchtst_96["beats_paper_CRPS"] == "True"
    assert patchtst_96["beats_paper_NMAE"] == "True"
    assert patchtst_96["beats_paper_baseline"] == "True"
