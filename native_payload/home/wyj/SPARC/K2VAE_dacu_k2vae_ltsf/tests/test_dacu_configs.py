from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_all_default_dacu_configs_point_to_dacu_forecaster():
    expected = {
        "dacu_k2vae.yaml",
        "dacu_dlinear.yaml",
        "dacu_patchtst.yaml",
        "dacu_itransformer.yaml",
        "dacu_timegrad.yaml",
        "dacu_tsdiff.yaml",
        "dacu_csdi.yaml",
    }
    found = {path.name for path in (ROOT / "config" / "default").glob("dacu_*.yaml")}

    assert expected <= found

    for name in expected:
        data = yaml.safe_load((ROOT / "config" / "default" / name).read_text())
        init_args = data["model"]["forecaster"]["init_args"]
        assert data["model"]["forecaster"]["class_path"].endswith("DACUForecaster")
        assert init_args["backbone_class_path"].startswith("probts.model.forecaster.")
        assert init_args["likelihood"] in {"laplace", "gaussian"}
        assert init_args["adapter_mode"] in {"full", "sample", "auto"}
        assert init_args["forecast_mode"] in {"calibrated_normal", "backbone", "rescale"}
        assert 0.0 <= init_args["calibration_strength"] <= 1.0
        assert isinstance(init_args["backbone_init_args"], dict)


def test_dacu_experiment_scripts_exist():
    assert (ROOT / "scripts" / "run_dacu_grid.sh").exists()
    assert (ROOT / "scripts" / "collect_dacu_results.py").exists()
    assert (ROOT / "scripts" / "compare_dacu_to_paper_baselines.py").exists()
    assert (ROOT / "scripts" / "run_dacu_etth1_vs_paper_baselines.sh").exists()
    assert (ROOT / "scripts" / "run_dacu_etth1_itransformer_vs_paper.sh").exists()
    assert (ROOT / "scripts" / "run_dacu_etth1_timegrad_vs_paper.sh").exists()
    assert (ROOT / "scripts" / "run_dacu_etth1_csdi_vs_paper.sh").exists()
    assert (ROOT / "scripts" / "run_dacu_etth1_remaining_vs_paper_baselines.sh").exists()
    assert (ROOT / "scripts" / "check_no_placeholder_results.py").exists()
    assert (ROOT / "scripts" / "run_dacu_k2vae_ltsf_all.sh").exists()
    runner = (ROOT / "scripts" / "run_dacu_etth1_four_baselines.sh").read_text()
    assert "--trainer.num_sanity_val_steps" in runner
    assert "--model.val_num_samples" in runner
    assert "SKIP_EXISTING" in runner
    assert "PYTHON_BIN" in runner
    assert '"${PYTHON_BIN}" run.py' in runner
    k2vae_runner = (ROOT / "scripts" / "run_dacu_k2vae_ltsf_all.sh").read_text()
    assert "nvidia-smi" in k2vae_runner
    assert "tmux" in k2vae_runner
    assert "SKIP_EXISTING" in k2vae_runner
    assert "VAL_NUM_SAMPLES=${VAL_NUM_SAMPLES:-100}" in k2vae_runner
    assert "EARLY_STOPPING_ENABLED=${EARLY_STOPPING_ENABLED:-true}" in k2vae_runner
    assert '--early_stopping.enabled "${EARLY_STOPPING_ENABLED}"' in k2vae_runner
    assert "illness_ltsf:36:24,36,48,60" in k2vae_runner
    assert "compare_dacu_k2vae_ltsf_to_paper.py" in k2vae_runner
    assert "render_dacu_k2vae_ltsf_table.py" in k2vae_runner
    assert "failed=0" in k2vae_runner
    assert 'wait "${pid}"' in k2vae_runner


def test_etth1_dacu_configs_enable_early_stopping():
    expected = {
        "dacu_csdi.yaml",
        "dacu_timegrad.yaml",
        "dacu_dlinear.yaml",
        "dacu_patchtst.yaml",
        "dacu_itransformer.yaml",
    }
    config_dir = ROOT / "config" / "ltsf" / "etth1"

    for name in expected:
        data = yaml.safe_load((config_dir / name).read_text())
        init_args = data["model"]["forecaster"]["init_args"]

        assert data["model"]["forecaster"]["class_path"].endswith("DACUForecaster")
        assert init_args["forecast_mode"] == "rescale"
        assert data["early_stopping"]["enabled"] is True
        assert data["early_stopping"]["monitor"] == "val_CRPS"


def test_four_baselines_runner_uses_configurable_python():
    runner = (ROOT / "scripts" / "run_dacu_etth1_four_baselines.sh").read_text()

    assert "PYTHON_BIN" in runner
    assert '"${PYTHON_BIN}" run.py' in runner
