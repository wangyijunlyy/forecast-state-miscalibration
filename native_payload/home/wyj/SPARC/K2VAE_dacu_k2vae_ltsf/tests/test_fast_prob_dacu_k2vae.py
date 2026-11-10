import sys
from types import SimpleNamespace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probts.model.forecaster.prob_forecaster.fast_dacu import (
    AffineProbCalibrator,
    FastJointProbDACUForecaster,
    FastPosthocProbDACUCalibrator,
    FastPosthocProbDACUv2,
)


def _batch(batch_size=2, context_length=24, prediction_length=12, target_dim=3):
    return SimpleNamespace(
        past_target_cdf=torch.randn(batch_size, context_length, target_dim),
        future_target_cdf=torch.randn(batch_size, prediction_length, target_dim),
    )


def _backbone_args():
    return {
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
        "weight_beta": 0.01,
        "sample_schedule": 2,
        "init_koopman": "dynamic",
    }


def _shared_args():
    return {
        "context_length": 24,
        "prediction_length": 12,
        "target_dim": 3,
        "freq": "H",
        "lags_list": [],
    }


def _wrapper(cls, **kwargs):
    init_args = {
        "backbone_class_path": "probts.model.forecaster.prob_forecaster.k2vae.k2VAEModel",
        "backbone_init_args": _backbone_args(),
        **_shared_args(),
    }
    init_args.update(kwargs)
    return cls(**init_args)


def test_fast_posthoc_dacu_freezes_backbone_and_trains_calibrator():
    wrapper = _wrapper(FastPosthocProbDACUCalibrator)

    backbone_params = list(wrapper.backbone.parameters())
    calibrator_params = list(wrapper.calibrator.parameters())

    assert backbone_params
    assert calibrator_params
    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in calibrator_params)


def test_fast_posthoc_dacu_loss_and_forecast_shape():
    torch.manual_seed(0)
    wrapper = _wrapper(FastPosthocProbDACUCalibrator, calibration_strength=0.5)
    batch = _batch()

    loss = wrapper.loss(batch)
    forecasts = wrapper.forecast(batch, num_samples=5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert forecasts.shape == (2, 5, 12, 3)
    assert torch.isfinite(forecasts).all()


def test_affine_prob_calibrator_is_frozen_backbone_posthoc_wrapper():
    wrapper = _wrapper(AffineProbCalibrator, mean_correction=True)

    assert all(not p.requires_grad for p in wrapper.backbone.parameters())
    assert wrapper.mean_bias.requires_grad
    assert wrapper.log_scale_bias.requires_grad


def test_affine_prob_calibrator_loss_and_forecast_shape():
    torch.manual_seed(1)
    wrapper = _wrapper(AffineProbCalibrator, mean_correction=True)
    batch = _batch()

    loss = wrapper.loss(batch)
    forecasts = wrapper.forecast(batch, num_samples=4)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert forecasts.shape == (2, 4, 12, 3)
    assert torch.isfinite(forecasts).all()


def test_fast_joint_prob_dacu_keeps_backbone_trainable():
    wrapper = _wrapper(FastJointProbDACUForecaster)

    assert any(p.requires_grad for p in wrapper.backbone.parameters())
    assert any(p.requires_grad for p in wrapper.calibrator.parameters())


def test_fast_joint_prob_dacu_uses_scalar_loss_and_sample_preserving_forecast():
    torch.manual_seed(2)
    wrapper = _wrapper(FastJointProbDACUForecaster, calibration_strength=0.5)
    batch = _batch()

    loss = wrapper.loss(batch)
    forecasts = wrapper.forecast(batch, num_samples=3)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert forecasts.shape == (2, 3, 12, 3)
    assert torch.isfinite(forecasts).all()


def test_fast_posthoc_v2_freezes_backbone_and_has_augmented_feature_head():
    wrapper = _wrapper(FastPosthocProbDACUv2, mean_correction=True)

    assert all(not p.requires_grad for p in wrapper.backbone.parameters())
    assert hasattr(wrapper.calibrator, "horizon_embedding")
    assert wrapper.calibrator.mean_correction is True


def test_fast_posthoc_v2_loss_and_forecast_shape():
    torch.manual_seed(3)
    wrapper = _wrapper(
        FastPosthocProbDACUv2,
        mean_correction=True,
        calibration_strength=0.8,
        crps_weight=0.2,
        coverage_weight=0.1,
        ranking_weight=0.05,
    )
    batch = _batch()

    loss = wrapper.loss(batch)
    forecasts = wrapper.forecast(batch, num_samples=4)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert forecasts.shape == (2, 4, 12, 3)
    assert torch.isfinite(forecasts).all()


def test_fast_posthoc_v2_augmented_features_include_loc_and_scale_residuals():
    torch.manual_seed(4)
    wrapper = _wrapper(FastPosthocProbDACUv2)
    batch = _batch()

    mean, base_scale, loc_residual, scale_residual = wrapper._frozen_features(batch)

    assert mean.shape == (2, 12, 3)
    assert base_scale.shape == mean.shape
    assert loc_residual.shape == mean.shape
    assert scale_residual.shape == mean.shape
    assert torch.isfinite(loc_residual).all()
    assert torch.isfinite(scale_residual).all()
