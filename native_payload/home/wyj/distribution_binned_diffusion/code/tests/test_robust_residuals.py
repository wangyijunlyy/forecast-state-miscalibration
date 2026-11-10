"""Regression checks for the observed near-zero-scale residual contamination."""
import unittest
import numpy as np
from robust_residuals import filter_residuals


class ResidualFilterTests(unittest.TestCase):
    def test_near_zero_scale_residuals_do_not_blow_up_reconstruction(self):
        # 5% of calibration coordinates have nearly collapsed ensembles.
        # Without filtering, the 99.5% residual quantile is 3 million.
        central = np.tile(np.arange(-4., 5.), 200)
        residuals = np.concatenate([central, np.full(100, 3_000_000.)])
        filtered, metadata = filter_residuals(residuals)
        reconstructed_tail = -2.7 + .14255 * np.quantile(filtered, .995)
        self.assertLess(abs(reconstructed_tail), 10.)
        np.testing.assert_array_equal(filtered[:len(central)], central)
        np.testing.assert_array_equal(residuals[-100:], np.full(100, 3_000_000.))
        self.assertEqual(metadata['clipped_upper'], 100)

    def test_extreme_negative_residuals_are_also_filtered(self):
        residuals = np.concatenate([np.tile(np.arange(-4., 5.), 200), np.full(100, -3_000_000.)])
        filtered, metadata = filter_residuals(residuals)
        self.assertGreater(np.quantile(filtered, .005), -30.)
        self.assertEqual(metadata['clipped_lower'], 100)

    def test_constant_pool_stays_finite_and_unchanged(self):
        filtered, _ = filter_residuals(np.full(200, 2.))
        np.testing.assert_array_equal(filtered, np.full(200, 2.))
        self.assertTrue(np.isfinite(filtered).all())

    def test_filter_is_equivariant_to_residual_units(self):
        residuals = np.concatenate([np.tile(np.arange(-4., 5.), 200), [1e6]])
        first, _ = filter_residuals(residuals)
        shifted, _ = filter_residuals(3 * residuals + 7)
        np.testing.assert_allclose(shifted, 3 * first + 7)


if __name__ == '__main__':
    unittest.main()
