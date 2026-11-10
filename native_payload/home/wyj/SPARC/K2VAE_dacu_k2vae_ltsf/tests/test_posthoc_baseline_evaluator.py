import numpy as np

from scripts.evaluate_k2vae_posthoc_baselines import (
    CalibrationReservoir,
    GlobalAffineAccumulator,
)


def test_calibration_reservoir_is_bounded_and_retains_coordinate_metadata():
    rng = np.random.default_rng(23)
    reservoir = CalibrationReservoir(max_points=80, seed=5)
    for _ in range(3):
        samples = rng.normal(size=(5, 20, 6, 3))
        targets = rng.normal(size=(5, 6, 3))
        reservoir.update(samples, targets)

    rows, targets, horizon_ids, channel_ids = reservoir.arrays()

    assert rows.shape == (80, 20)
    assert targets.shape == horizon_ids.shape == channel_ids.shape == (80,)
    assert set(horizon_ids) <= set(range(6))
    assert set(channel_ids) <= set(range(3))
    assert reservoir.horizon_length == 6
    assert reservoir.channel_count == 3


def test_global_affine_accumulator_recovers_streamed_fit():
    rng = np.random.default_rng(29)
    accumulator = GlobalAffineAccumulator()
    for _ in range(4):
        latent_mean = rng.normal(size=(32, 3, 1))
        targets = latent_mean + rng.normal(size=(32, 3, 1))
        samples = latent_mean[:, None] + 0.8 + rng.normal(scale=1.5, size=(32, 100, 3, 1))
        accumulator.update(samples, targets)

    calibrator = accumulator.calibrator()

    assert -1.1 < calibrator.bias < -0.5
    assert 0.45 < calibrator.scale < 0.9
