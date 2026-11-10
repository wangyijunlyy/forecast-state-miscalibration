import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probts.model.forecaster.prob_forecaster.dacu import DACUForecaster
from probts.model.forecaster.point_forecaster.dlinear import DLinear

MODULE_PATH = ROOT / "probts" / "model" / "nn" / "prob" / "dacu.py"
spec = importlib.util.spec_from_file_location("dacu_utils", MODULE_PATH)
dacu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dacu)


def test_augmentations_preserve_shape_and_return_metadata():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 3)
    augmenter = dacu.TimeSeriesAugmenter(
        jitter_std=0.01,
        scaling_std=0.05,
        time_mask_ratio=0.25,
        channel_mask_ratio=0.33,
        freq_dropout_ratio=0.2,
    )

    y, meta = augmenter(x)

    assert y.shape == x.shape
    assert set(meta) >= {"jitter", "scaling", "time_mask", "channel_mask", "freq_dropout"}
    assert torch.isfinite(y).all()


def test_uncertainty_nll_is_finite_for_laplace_and_gaussian():
    residual = torch.tensor([[[0.1, 1.0], [2.0, 0.5]]])
    log_var = torch.zeros_like(residual)

    laplace = dacu.uncertainty_nll(residual, log_var, likelihood="laplace")
    gaussian = dacu.uncertainty_nll(residual, log_var, likelihood="gaussian")

    assert torch.isfinite(laplace).all()
    assert torch.isfinite(gaussian).all()
    assert laplace.mean() > 0
    assert gaussian.mean() > 0


def test_detached_residual_blocks_gradient_to_predictions():
    pred_a = torch.randn(2, 8, 3, requires_grad=True)
    pred_b = torch.randn(2, 8, 3, requires_grad=True)
    log_var = torch.zeros_like(pred_a, requires_grad=True)

    residual = dacu.consistency_residual(pred_a, pred_b, detach=True)
    loss = dacu.uncertainty_nll(residual, log_var, likelihood="laplace").mean()
    loss.backward()

    assert pred_a.grad is None
    assert pred_b.grad is None
    assert log_var.grad is not None
    assert torch.isfinite(log_var.grad).all()


def test_rescale_samples_preserves_standardized_backbone_residuals():
    base_samples = torch.tensor(
        [
            [
                [[0.0], [2.0]],
                [[2.0], [4.0]],
                [[4.0], [6.0]],
            ]
        ]
    )
    refined_mean = torch.tensor([[[10.0], [20.0]]])
    calibrated_scale = torch.tensor([[[2.0], [0.5]]])

    rescaled = dacu.rescale_samples_by_calibration(base_samples, refined_mean, calibrated_scale)

    original_mean = base_samples.mean(dim=1, keepdim=True)
    original_scale = base_samples.std(dim=1, keepdim=True).clamp_min(1e-6)
    original_residuals = (base_samples - original_mean) / original_scale
    rescaled_residuals = (rescaled - refined_mean.unsqueeze(1)) / calibrated_scale.unsqueeze(1)

    assert rescaled.shape == base_samples.shape
    assert torch.allclose(rescaled.mean(dim=1), refined_mean)
    assert torch.allclose(rescaled.std(dim=1), calibrated_scale)
    assert torch.allclose(rescaled_residuals, original_residuals)


def test_blend_log_variance_uses_configured_strength():
    base_scale = torch.full((1, 2, 1), 2.0)
    calibrated_log_var = torch.zeros_like(base_scale)

    blended = dacu.blend_log_variance(base_scale, calibrated_log_var, calibration_strength=0.25)

    expected = 0.75 * (2.0 * torch.log(base_scale)) + 0.25 * calibrated_log_var
    assert torch.allclose(blended, expected)


class LagSensitiveBackbone(torch.nn.Module):
    def __init__(self, context_length, prediction_length, target_dim, lags_list, **kwargs):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.required_history_length = context_length + max(lags_list)
        self.seen_past_lengths = []

    def loss(self, batch_data):
        return batch_data.past_target_cdf.sum() * 0.0

    def forecast(self, batch_data, num_samples):
        past_length = batch_data.past_target_cdf.shape[1]
        self.seen_past_lengths.append(past_length)
        assert past_length >= self.required_history_length
        batch_size = batch_data.past_target_cdf.shape[0]
        return torch.zeros(batch_size, num_samples, self.prediction_length, self.target_dim)


class EncoderDistributionBackbone(torch.nn.Module):
    def __init__(self, prediction_length, target_dim, **kwargs):
        super().__init__()
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.encoder_calls = 0
        self.forecast_calls = 0

    def encoder(self, inputs):
        self.encoder_calls += 1
        batch_size = inputs.shape[0]
        loc = torch.zeros(batch_size, self.prediction_length, self.target_dim)
        scale = torch.ones_like(loc)
        return torch.distributions.Normal(loc, scale)

    def get_inputs(self, batch_data, mode):
        assert mode == "encode"
        return batch_data.past_target_cdf

    def loss(self, batch_data):
        return batch_data.past_target_cdf.sum() * 0.0

    def forecast(self, batch_data, num_samples):
        self.forecast_calls += 1
        raise AssertionError("full adapter should use encoder statistics instead of forecast samples")


def test_sample_adapter_preserves_lag_history_when_augmenting_context():
    wrapper = DACUForecaster(
        backbone_class_path=f"{__name__}.LagSensitiveBackbone",
        adapter_mode="sample",
        consistency_samples=2,
        context_length=96,
        prediction_length=24,
        target_dim=2,
        freq="H",
        lags_list=[168],
    )
    batch = SimpleNamespace(
        past_target_cdf=torch.randn(3, 264, 2),
    )

    loss = wrapper.loss(batch)

    assert torch.isfinite(loss)
    assert min(wrapper.backbone.seen_past_lengths) == 264


def test_full_adapter_can_use_encoder_distribution_statistics():
    wrapper = DACUForecaster(
        backbone_class_path=f"{__name__}.EncoderDistributionBackbone",
        adapter_mode="full",
        context_length=96,
        prediction_length=24,
        target_dim=2,
        freq="H",
        lags_list=[],
    )
    batch = SimpleNamespace(
        past_target_cdf=torch.randn(3, 96, 2),
        future_target_cdf=torch.randn(3, 24, 2),
    )

    loss = wrapper.loss(batch)

    assert torch.isfinite(loss)
    assert wrapper.backbone.encoder_calls == 2
    assert wrapper.backbone.forecast_calls == 0


def test_dlinear_forecast_returns_batch_first_samples():
    model = DLinear(
        kernel_size=25,
        individual=True,
        context_length=96,
        prediction_length=24,
        target_dim=2,
        freq="H",
        use_lags=False,
        use_feat_idx_emb=False,
        use_time_feat=False,
        lags_list=[],
    )
    batch = SimpleNamespace(
        past_target_cdf=torch.randn(3, 96, 2),
        future_target_cdf=torch.randn(3, 24, 2),
    )

    forecasts = model.forecast(batch, num_samples=5)

    assert forecasts.shape == (3, 5, 24, 2)


def test_zero_dacu_weights_skip_extra_dacu_terms():
    wrapper = DACUForecaster(
        backbone_class_path=f"{__name__}.LagSensitiveBackbone",
        adapter_mode="sample",
        uncertainty_weight=0.0,
        consistency_weight=0.0,
        context_length=96,
        prediction_length=24,
        target_dim=2,
        freq="H",
        lags_list=[168],
    )
    batch = SimpleNamespace(
        past_target_cdf=torch.randn(3, 264, 2),
    )

    def fail_if_called(_batch):
        raise AssertionError("zero-weight DACU terms should not be evaluated")

    wrapper._dacu_terms = fail_if_called

    loss = wrapper.loss(batch)

    assert torch.isfinite(loss)
