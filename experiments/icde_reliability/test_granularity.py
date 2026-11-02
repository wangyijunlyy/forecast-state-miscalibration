"""Catch partition fitting or fallback changes that would invalidate the sweep."""
import unittest
import numpy as np
from portable_dc import EmpiricalCalibrator
from granularity import fit_partition, reconstruct


class GranularityTests(unittest.TestCase):
    def test_existing_dc16_dc64_predictions_are_unchanged(self):
        rng = np.random.default_rng(724)
        rows = rng.normal(size=(11000, 100)) * rng.uniform(.1, 2, (11000, 1))
        rows += rng.normal(size=(11000, 1))
        rows[:20] = 0
        y = rng.normal(size=len(rows)); h = np.arange(len(rows)) % 12
        query = rows[37:114]; query_h = h[37:114]
        for g in [1, 4]:
            original = EmpiricalCalibrator.fit_rows(rows, y, h, 12, n_horizon_bins=g)
            actual, meta = fit_partition(rows, y, h, 12, 4, g)
            values, fallback = reconstruct(actual, query, query_h)
            np.testing.assert_array_equal(values, original.transform_rows(query, query_h))
            self.assertEqual(meta['excluded_sources'], 20)
            self.assertEqual(meta['nominal_states'], 16*g)
            self.assertTrue(0 <= fallback <= len(query))

    def test_global_fallback_uses_surviving_count_and_real_horizon_ids(self):
        # Two horizon groups have 128 and 127 valid sources respectively.
        rows = np.tile(np.linspace(-1., 1., 100), (256, 1))
        rows[-1] = 0
        h = np.r_[np.zeros(128, dtype=int), np.ones(128, dtype=int)]
        y = rows.mean(1)+np.where(h == 0, -2., 3.)*rows.std(1, ddof=1)
        cal, meta = fit_partition(rows, y, h, 2, 1, 2)
        self.assertEqual(meta['eligible_local_pools'], 1)
        self.assertEqual(meta['source_counts'], [128, 127])
        query = rows[[129, 0]]; query_h = np.array([1, 0])
        values, fallback = reconstruct(cal, query, query_h)
        self.assertEqual(fallback, 1)
        expected_global = np.quantile(np.r_[np.full(128, -2.), np.full(127, 3.)],
                                      (np.arange(100)+.5)/100)
        np.testing.assert_allclose(values[0], query[0].mean()+query[0].std(ddof=1)*expected_global)
        np.testing.assert_allclose(values[1], query[1].mean()-2*query[1].std(ddof=1))


if __name__ == '__main__':
    unittest.main()
