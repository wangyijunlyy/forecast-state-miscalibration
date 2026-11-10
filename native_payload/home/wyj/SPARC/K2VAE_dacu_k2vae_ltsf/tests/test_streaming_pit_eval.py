import numpy as np

from scripts.evaluate_k2vae_sparc_pit_warp import PITHistogramAccumulator
from sparc_adapters.pit_warp import compute_sample_pit, qice_from_pit


def test_pit_histogram_accumulator_matches_materialized_pit():
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(3, 11, 4, 2))
    targets = rng.normal(size=(3, 4, 2))

    materialized = compute_sample_pit(samples, targets)
    acc = PITHistogramAccumulator(sample_count=samples.shape[1])
    acc.update(samples, targets)

    assert acc.total == materialized.size
    assert np.allclose(acc.values(), np.sort(materialized))
    assert acc.qice(n_bins=10) == qice_from_pit(materialized, n_bins=10)


def test_pit_histogram_accumulator_quantiles_match_materialized_pit():
    rng = np.random.default_rng(11)
    levels = np.linspace(0.0, 1.0, 13)
    acc = PITHistogramAccumulator(sample_count=17)
    all_pit = []

    for _ in range(4):
        samples = rng.normal(size=(2, 17, 5, 3))
        targets = rng.normal(size=(2, 5, 3))
        acc.update(samples, targets)
        all_pit.append(compute_sample_pit(samples, targets))

    materialized = np.concatenate(all_pit)
    assert np.allclose(acc.quantile(levels), np.quantile(materialized, levels))
