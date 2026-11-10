import torch

from sparc_ppm.calibration import SampleSPARCCalibrator


def _inputs():
    torch.manual_seed(7)
    samples = torch.randn(2, 20, 4, 3)
    augmented = samples + 0.25
    return samples, augmented


def test_disabling_perturbation_features_ignores_disagreement_values():
    samples, augmented = _inputs()
    calibrator = SampleSPARCCalibrator(
        prediction_length=4,
        target_dim=3,
        use_perturbation_features=False,
    )

    dist_a, _ = calibrator.distribution_from_samples(samples, augmented)
    dist_b, _ = calibrator.distribution_from_samples(samples, augmented * 5.0)

    torch.testing.assert_close(dist_a.loc, dist_b.loc)
    torch.testing.assert_close(dist_a.scale, dist_b.scale)


def test_gaussian_redraw_preserves_shape_but_not_native_residuals():
    samples, augmented = _inputs()
    calibrator = SampleSPARCCalibrator(
        prediction_length=4,
        target_dim=3,
        reconstruction_mode="gaussian",
    )

    torch.manual_seed(11)
    redrawn = calibrator.calibrate_samples_with_stats(samples, samples, augmented)

    assert redrawn.shape == samples.shape
    native_standardized = (samples - samples.mean(1, keepdim=True)) / samples.std(
        1, keepdim=True
    )
    redraw_standardized = (redrawn - redrawn.mean(1, keepdim=True)) / redrawn.std(
        1, keepdim=True
    )
    assert not torch.allclose(native_standardized, redraw_standardized)


def test_unbounded_update_is_not_clipped_by_configured_limits():
    mean = torch.zeros(1, 2, 1)
    scale = torch.ones_like(mean)
    residual = torch.zeros_like(mean)
    bounded = SampleSPARCCalibrator(
        prediction_length=2,
        target_dim=1,
        max_mean_delta=0.1,
        max_log_scale_delta=0.1,
        bounded_updates=True,
    )
    unbounded = SampleSPARCCalibrator(
        prediction_length=2,
        target_dim=1,
        max_mean_delta=0.1,
        max_log_scale_delta=0.1,
        bounded_updates=False,
    )
    unbounded.load_state_dict(bounded.state_dict())
    for module in (bounded, unbounded):
        final = module.net[-1]
        torch.nn.init.zeros_(final.weight)
        final.bias.data.fill_(2.0)

    bounded_mean, bounded_log_var = bounded(mean, scale, residual, residual)
    unbounded_mean, unbounded_log_var = unbounded(mean, scale, residual, residual)

    assert bounded_mean.abs().max() <= 0.1
    assert unbounded_mean.abs().max() > 0.1
    assert bounded_log_var.abs().max() <= 0.2
    assert unbounded_log_var.abs().max() > 0.2
