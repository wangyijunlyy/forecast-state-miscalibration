from types import SimpleNamespace

import torch

from probts.model.forecaster.prob_forecaster.fast_dacu import (
    AugmentedFeatureCalibrator,
    FastPosthocProbDACUv2,
    GlobalAffineProbCalibrator,
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


def _wrapper(cls=FastPosthocProbDACUv2, **kwargs):
    args = {
        "backbone_class_path": "probts.model.forecaster.prob_forecaster.k2vae.k2VAEModel",
        "backbone_init_args": _backbone_args(),
        "context_length": 24,
        "prediction_length": 12,
        "target_dim": 3,
        "freq": "H",
        "lags_list": [],
    }
    args.update(kwargs)
    return cls(**args)


def _batch():
    return SimpleNamespace(
        past_target_cdf=torch.randn(2, 24, 3),
        future_target_cdf=torch.randn(2, 12, 3),
    )


class ConstantRaw(torch.nn.Module):
    def __init__(self, mean_raw: float, scale_raw: float):
        super().__init__()
        self.register_buffer("raw", torch.tensor([mean_raw, scale_raw]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.raw.to(features).expand(*features.shape[:-1], 2)


def test_disabling_perturbation_features_zeroes_disagreement_inputs():
    torch.manual_seed(0)
    wrapper = _wrapper(use_perturbation_features=False)

    _, _, loc_residual, scale_residual = wrapper._frozen_features(_batch())

    assert torch.count_nonzero(loc_residual) == 0
    assert torch.count_nonzero(scale_residual) == 0


def test_unbounded_update_skips_tanh_and_max_delta_limits():
    mean = torch.zeros(1, 2, 1)
    scale = torch.ones_like(mean)
    residual = torch.zeros_like(mean)
    bounded = AugmentedFeatureCalibrator(
        prediction_length=2,
        target_dim=1,
        mean_correction=True,
        max_mean_delta=0.1,
        max_log_scale_delta=0.2,
        bounded_updates=True,
    )
    unbounded = AugmentedFeatureCalibrator(
        prediction_length=2,
        target_dim=1,
        mean_correction=True,
        max_mean_delta=0.1,
        max_log_scale_delta=0.2,
        bounded_updates=False,
    )
    bounded.net = ConstantRaw(2.0, 2.0)
    unbounded.net = ConstantRaw(2.0, 2.0)

    bounded_mean, bounded_log_var = bounded(mean, scale, residual, residual)
    unbounded_mean, unbounded_log_var = unbounded(mean, scale, residual, residual)

    assert torch.all(bounded_mean.abs() <= 0.1 + 1e-7)
    assert torch.all(bounded_log_var.abs() <= 0.4 + 1e-7)
    assert torch.allclose(unbounded_mean, torch.full_like(mean, 2.0))
    assert torch.allclose(unbounded_log_var, torch.full_like(mean, 4.0))


def test_gaussian_redraw_replaces_native_residual_geometry():
    sample_count = 5000
    base = torch.linspace(-2.0, 2.0, sample_count).view(1, sample_count, 1, 1)
    batch = _batch()
    refined_mean = torch.full((1, 1, 1), 3.0)
    log_var = torch.zeros_like(refined_mean)
    residual = _wrapper(reconstruction_mode="residual")
    gaussian = _wrapper(reconstruction_mode="gaussian")
    residual._backbone_samples = lambda _batch, _count: base
    gaussian._backbone_samples = lambda _batch, _count: base

    preserved = residual._calibrated_samples(batch, sample_count, refined_mean, log_var, 1.0)
    torch.manual_seed(7)
    redrawn = gaussian._calibrated_samples(batch, sample_count, refined_mean, log_var, 1.0)

    assert preserved.shape == redrawn.shape == base.shape
    assert not torch.allclose(redrawn, preserved)
    assert torch.allclose(preserved.mean(dim=1), refined_mean, atol=1e-5)
    assert torch.allclose(redrawn.mean(dim=1), refined_mean, atol=0.05)
    assert torch.allclose(redrawn.std(dim=1), torch.ones_like(refined_mean), atol=0.05)


def test_global_affine_uses_one_location_and_scale_parameter():
    wrapper = _wrapper(GlobalAffineProbCalibrator, mean_correction=True)

    assert wrapper.mean_bias.shape == torch.Size([])
    assert wrapper.log_scale_bias.shape == torch.Size([])
    assert all(not parameter.requires_grad for parameter in wrapper.backbone.parameters())
    loss = wrapper.loss(_batch())

    assert loss.ndim == 0
    assert torch.isfinite(loss)
