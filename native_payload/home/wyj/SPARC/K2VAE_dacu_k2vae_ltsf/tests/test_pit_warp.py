import numpy as np

from sparc_adapters.pit_warp import PITWarpCalibrator, compute_sample_pit, qice_from_pit


def test_pit_warp_reduces_qice_on_biased_samples():
    rng = np.random.default_rng(7)
    targets = rng.normal(loc=0.0, scale=1.0, size=(128, 24, 2))
    samples = targets[:, None, :, :] + rng.normal(loc=-0.45, scale=1.15, size=(128, 100, 24, 2))

    pit = compute_sample_pit(samples, targets)
    before = qice_from_pit(pit)
    calibrator = PITWarpCalibrator.from_pit(pit, n_knots=41, strength=1.0)
    after = qice_from_pit(calibrator.transform_pit(pit))

    assert after < before * 0.4


def test_warp_samples_preserves_shape_and_monotone_order():
    rng = np.random.default_rng(11)
    samples = rng.normal(size=(4, 50, 8, 3))
    calibrator = PITWarpCalibrator.from_pit(rng.uniform(size=1000), n_knots=21, strength=0.8)

    warped = calibrator.warp_samples(samples)

    assert warped.shape == samples.shape
    sorted_warped = np.sort(warped, axis=1)
    assert np.all(np.diff(sorted_warped, axis=1) >= -1e-12)
