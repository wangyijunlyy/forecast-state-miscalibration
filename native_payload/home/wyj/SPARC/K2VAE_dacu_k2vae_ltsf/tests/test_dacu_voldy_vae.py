from types import SimpleNamespace
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probts.model.forecaster.prob_forecaster.voldy_vae import VolDy_VAEModel
from probts.model.forecaster.prob_forecaster.dacu import DACUForecaster


def _batch(batch_size=2, context_length=24, prediction_length=12, target_dim=3):
    return SimpleNamespace(
        past_target_cdf=torch.randn(batch_size, context_length, target_dim),
        future_target_cdf=torch.randn(batch_size, prediction_length, target_dim),
    )


def _model(context_length=24, prediction_length=12, target_dim=3):
    return VolDy_VAEModel(
        d_model=16,
        d_ff=32,
        e_layers=1,
        dropout=0.0,
        activation="relu",
        n_heads=2,
        factor=1,
        patch_len=6,
        multistep=True,
        dynamic_dim=8,
        hidden_layers=1,
        hidden_dim=16,
        vol_hidden_dim=8,
        weight_beta=0.01,
        sample_schedule=2,
        init_koopman="dynamic",
        context_length=context_length,
        prediction_length=prediction_length,
        target_dim=target_dim,
        freq="H",
        lags_list=[],
    )


def test_voldy_vae_loss_is_scalar_and_finite():
    torch.manual_seed(0)
    model = _model()
    loss = model.loss(_batch())
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_voldy_vae_forecast_returns_batch_first_samples():
    torch.manual_seed(0)
    model = _model()
    forecasts = model.forecast(_batch(), num_samples=3)
    assert forecasts.shape == (2, 3, 12, 3)
    assert torch.isfinite(forecasts).all()


def _dacu_wrapper(forecast_mode="rescale"):
    return DACUForecaster(
        backbone_class_path="probts.model.forecaster.prob_forecaster.voldy_vae.VolDy_VAEModel",
        adapter_mode="auto",
        likelihood="laplace",
        uncertainty_weight=0.005,
        consistency_weight=0.02,
        consistency_samples=2,
        forecast_mode=forecast_mode,
        calibration_strength=0.2,
        jitter_std=0.01,
        scaling_std=0.01,
        time_mask_ratio=0.0,
        channel_mask_ratio=0.0,
        backbone_init_args={
            "d_model": 16,
            "d_ff": 32,
            "e_layers": 1,
            "dropout": 0.0,
            "activation": "relu",
            "n_heads": 2,
            "factor": 1,
            "patch_len": 6,
            "multistep": True,
            "dynamic_dim": 8,
            "hidden_layers": 1,
            "hidden_dim": 16,
            "vol_hidden_dim": 8,
            "weight_beta": 0.01,
            "sample_schedule": 2,
            "init_koopman": "dynamic",
        },
        context_length=24,
        prediction_length=12,
        target_dim=3,
        freq="H",
        lags_list=[],
    )


def test_dacu_auto_mode_treats_voldy_vae_as_full_adapter():
    wrapper = _dacu_wrapper()
    assert wrapper.full_adapter is True


def test_dacu_voldy_vae_loss_is_scalar_and_finite():
    torch.manual_seed(0)
    wrapper = _dacu_wrapper()
    loss = wrapper.loss(_batch())
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dacu_voldy_vae_rescale_forecast_keeps_sample_shape():
    torch.manual_seed(0)
    wrapper = _dacu_wrapper(forecast_mode="rescale")
    forecasts = wrapper.forecast(_batch(), num_samples=3)
    assert forecasts.shape == (2, 3, 12, 3)
    assert torch.isfinite(forecasts).all()


def test_voldy_vae_etth1_configs_and_runner_exist():
    vanilla = ROOT / "config" / "ltsf" / "etth1" / "voldy_vae.yaml"
    dacu = ROOT / "config" / "ltsf" / "etth1" / "dacu_voldy_vae.yaml"
    runner = ROOT / "scripts" / "run_dacu_voldy_vae_etth1.sh"

    assert vanilla.exists()
    assert dacu.exists()
    assert runner.exists()

    vanilla_data = yaml.safe_load(vanilla.read_text())
    dacu_data = yaml.safe_load(dacu.read_text())
    assert vanilla_data["model"]["forecaster"]["class_path"].endswith("VolDy_VAEModel")
    assert dacu_data["model"]["forecaster"]["class_path"].endswith("DACUForecaster")
    assert dacu_data["model"]["forecaster"]["init_args"]["forecast_mode"] == "rescale"
    runner_text = runner.read_text()
    assert "96 192 336 720" in runner_text
    assert 'PYTHON="${PYTHON:-python}"' in runner_text
