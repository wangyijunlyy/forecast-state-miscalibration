import numpy as np

from sparc_ppm.pit_warp import PITWarpCalibrator, qice_from_pit


def test_pit_warp_reduces_histogram_calibration_error():
    rng = np.random.default_rng(7)
    calibration_pit = rng.beta(2.5, 2.5, size=2000)
    before = qice_from_pit(calibration_pit, n_bins=10)
    calibrator = PITWarpCalibrator.from_pit(calibration_pit, n_knots=11, strength=1.0)
    after = qice_from_pit(calibrator.transform_pit(calibration_pit), n_bins=10)

    assert after < before


def test_pit_warp_keeps_sample_shape_and_monotone_order():
    base_samples = np.tile(np.linspace(-2.0, 2.0, 100), (3, 2, 1)).transpose(0, 2, 1)
    base_samples = base_samples[:, :, :, None]
    calibration_pit = np.linspace(0.05, 0.95, 100)
    calibrator = PITWarpCalibrator.from_pit(calibration_pit, n_knots=11, strength=0.5)

    warped = calibrator.warp_samples(base_samples)

    assert warped.shape == base_samples.shape
    assert np.all(np.diff(warped, axis=1) >= -1e-8)
