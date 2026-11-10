import numpy as np

from sparc_adapters.pit_warp import compute_sample_pit, qice_from_pit
from sparc_adapters.posthoc_baselines import (
    CQRCalibrator,
    ConditionalPITCalibrator,
    DistributionCalibrationCalibrator,
    GlobalAffineCalibrator,
    interval_metrics,
)


def test_global_affine_corrects_shared_location_and_scale_error():
    rng = np.random.default_rng(3)
    latent_mean = rng.normal(size=(256, 4, 2))
    targets = latent_mean + rng.normal(size=(256, 4, 2))
    samples = latent_mean[:, None] + 1.25 + rng.normal(scale=1.8, size=(256, 100, 4, 2))
    calibrator = GlobalAffineCalibrator.fit(samples, targets)

    calibrated = calibrator.transform(samples)
    before = np.mean(np.abs(samples.mean(axis=1) - targets))
    after = np.mean(np.abs(calibrated.mean(axis=1) - targets))

    assert -1.5 < calibrator.bias < -1.0
    assert 0.4 < calibrator.scale < 0.8
    assert after < before * 0.8


def test_conditional_pit_corrects_opposite_horizon_biases():
    rng = np.random.default_rng(7)
    train_targets = rng.normal(size=(320, 2, 1))
    test_targets = rng.normal(size=(160, 2, 1))
    train_samples = train_targets[:, None] + rng.normal(size=(320, 100, 2, 1))
    test_samples = test_targets[:, None] + rng.normal(size=(160, 100, 2, 1))
    train_samples[:, :, 0] += 0.9
    train_samples[:, :, 1] -= 0.9
    test_samples[:, :, 0] += 0.9
    test_samples[:, :, 1] -= 0.9

    calibrator = ConditionalPITCalibrator.fit(
        train_samples,
        train_targets,
        n_horizon_bins=2,
        n_scale_bins=1,
        n_channel_bins=1,
        min_bin_size=50,
    )
    calibrated = calibrator.transform(test_samples)

    before = np.mean(
        [qice_from_pit(compute_sample_pit(test_samples[:, :, h : h + 1], test_targets[:, h : h + 1])) for h in range(2)]
    )
    after = np.mean(
        [qice_from_pit(compute_sample_pit(calibrated[:, :, h : h + 1], test_targets[:, h : h + 1])) for h in range(2)]
    )

    assert calibrated.shape == test_samples.shape
    assert after < before * 0.5


def test_distribution_calibration_resamples_conditional_residuals():
    rng = np.random.default_rng(11)
    train_targets = rng.normal(size=(300, 3, 1))
    test_targets = rng.normal(size=(150, 3, 1))
    train_samples = 1.1 + rng.normal(scale=0.7, size=(300, 100, 3, 1))
    test_samples = 1.1 + rng.normal(scale=0.7, size=(150, 100, 3, 1))

    calibrator = DistributionCalibrationCalibrator.fit(
        train_samples,
        train_targets,
        n_mean_bins=1,
        n_scale_bins=1,
        n_horizon_bins=1,
        min_bin_size=50,
    )
    calibrated = calibrator.transform(test_samples)

    before_error = np.mean(np.abs(np.median(test_samples, axis=1) - test_targets))
    after_error = np.mean(np.abs(np.median(calibrated, axis=1) - test_targets))
    before_qice = qice_from_pit(compute_sample_pit(test_samples, test_targets))
    after_qice = qice_from_pit(compute_sample_pit(calibrated, test_targets))

    assert calibrated.shape == test_samples.shape
    assert after_error < before_error
    assert after_qice < before_qice


def test_cqr_improves_coverage_for_underdispersed_intervals():
    rng = np.random.default_rng(17)
    train_targets = rng.normal(size=(400, 2, 1))
    test_targets = rng.normal(size=(200, 2, 1))
    train_samples = rng.normal(scale=0.3, size=(400, 100, 2, 1))
    test_samples = rng.normal(scale=0.3, size=(200, 100, 2, 1))
    levels = (0.5, 0.8, 0.9)

    calibrator = CQRCalibrator.fit(train_samples, train_targets, levels=levels)
    baseline = interval_metrics(test_samples, test_targets, levels=levels)
    calibrated = interval_metrics(test_samples, test_targets, levels=levels, calibrator=calibrator)

    assert calibrated["coverage_error"] < baseline["coverage_error"] * 0.4
    assert np.isfinite(calibrated["mean_interval_score"])
