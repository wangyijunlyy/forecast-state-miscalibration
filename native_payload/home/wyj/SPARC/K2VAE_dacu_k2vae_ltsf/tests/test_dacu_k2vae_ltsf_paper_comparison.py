import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_k2vae_ltsf_baselines_cover_all_long_term_horizons():
    baseline_path = ROOT / "scripts" / "paper_k2vae_ltsf_baselines.csv"

    rows = list(csv.DictReader(baseline_path.open()))
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(int(row["horizon"]))

    assert len(rows) == 36
    assert set(by_dataset) == {
        "etth1",
        "etth2",
        "ettm1",
        "ettm2",
        "electricity_ltsf",
        "traffic_ltsf",
        "weather_ltsf",
        "exchange_ltsf",
        "illness_ltsf",
    }

    for dataset, horizons in by_dataset.items():
        if dataset == "illness_ltsf":
            assert sorted(horizons) == [24, 36, 48, 60]
        else:
            assert sorted(horizons) == [96, 192, 336, 720]

    etth1_96 = next(
        row for row in rows if row["dataset"] == "etth1" and row["horizon"] == "96"
    )
    illness_36 = next(
        row for row in rows if row["dataset"] == "illness_ltsf" and row["horizon"] == "36"
    )
    assert etth1_96["paper_CRPS"] == "0.264"
    assert etth1_96["paper_NMAE"] == "0.336"
    assert illness_36["paper_CRPS"] == "0.113"
    assert illness_36["paper_NMAE"] == "0.142"


def test_compare_dacu_k2vae_ltsf_to_paper(tmp_path):
    result_dir = (
        tmp_path
        / "dacu_k2vae"
        / "etth1"
        / "pred_96"
        / "etth1_DACU-k2VAEModel_CTX96_PRED96"
        / "version_0"
    )
    result_dir.mkdir(parents=True)
    result_file = result_dir / "horizons_results.csv"
    with result_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "test_CRPS", "test_ND"])
        writer.writeheader()
        writer.writerow({"horizon": "96", "test_CRPS": "0.250", "test_ND": "0.330"})

    out_file = tmp_path / "summary.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_dacu_k2vae_ltsf_to_paper.py"),
            str(tmp_path),
            "--out",
            str(out_file),
        ],
        check=True,
    )

    rows = list(csv.DictReader(out_file.open()))
    etth1_96 = next(row for row in rows if row["dataset"] == "etth1" and row["horizon"] == "96")

    assert len(rows) == 36
    assert etth1_96["paper_CRPS"] == "0.264"
    assert etth1_96["dacu_test_CRPS"] == "0.250"
    assert etth1_96["delta_CRPS"] == "-0.014000000000000012"
    assert etth1_96["paper_NMAE"] == "0.336"
    assert etth1_96["dacu_test_ND"] == "0.330"
    assert etth1_96["beats_paper_CRPS"] == "True"
    assert etth1_96["beats_paper_NMAE"] == "True"
    assert etth1_96["beats_paper_k2vae"] == "True"


def test_render_dacu_k2vae_ltsf_table(tmp_path):
    summary_file = tmp_path / "summary.csv"
    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "horizon",
                "paper_CRPS",
                "paper_CRPS_std",
                "dacu_test_CRPS",
                "delta_CRPS",
                "beats_paper_CRPS",
                "paper_NMAE",
                "paper_NMAE_std",
                "dacu_test_ND",
                "delta_ND",
                "beats_paper_NMAE",
                "beats_paper_k2vae",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "etth1",
                "horizon": "96",
                "paper_CRPS": "0.264",
                "paper_CRPS_std": "0.020",
                "dacu_test_CRPS": "0.254",
                "delta_CRPS": "-0.010",
                "beats_paper_CRPS": "True",
                "paper_NMAE": "0.336",
                "paper_NMAE_std": "0.041",
                "dacu_test_ND": "0.330",
                "delta_ND": "-0.006",
                "beats_paper_NMAE": "True",
                "beats_paper_k2vae": "True",
                "source": "/tmp/result.csv",
            }
        )

    out_file = tmp_path / "table.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_dacu_k2vae_ltsf_table.py"),
            str(summary_file),
            "--out",
            str(out_file),
        ],
        check=True,
    )

    table = out_file.read_text()
    assert "| Dataset | Horizon | Paper CRPS | DACU CRPS | Delta CRPS | CRPS Lower? |" in table
    assert "| etth1 | 96 | 0.264 +/- 0.020 | 0.254 | -0.010 | yes |" in table
    assert "Wins: 1/1 completed rows" in table
