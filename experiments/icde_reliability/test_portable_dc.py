"""Deployment must preserve source filtering and arbitrary retained row order."""
import tempfile
import unittest
from pathlib import Path
import numpy as np


class PortableCalibrationTests(unittest.TestCase):
    def test_compiled_queries_preserve_all_members_and_strict_threshold_ties(self):
        from portable_dc import EmpiricalCalibrator
        rng = np.random.default_rng(98)
        fitting = rng.normal(size=(3000, 100)); y = rng.normal(size=3000)
        cal = EmpiricalCalibrator.fit_rows(fitting, y, np.arange(3000)%12, horizon_length=12)
        compiled = cal.compile()
        rows = rng.normal(size=(43, 100)); h = np.arange(43)%12
        decoded = cal.transform_rows(rows, h)
        np.testing.assert_array_equal(compiled.transform_rows(rows, h), decoded)
        # Exact member values exercise <= versus >, including duplicated tails.
        thresholds = np.column_stack([decoded[:, 0], decoded[:, 49], decoded[:, 99]])
        expected = np.stack([(decoded > thresholds[:, j,None]).mean(1) for j in range(3)], axis=1)
        np.testing.assert_array_equal(compiled.probabilities_rows(rows, h, thresholds), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'compiled.npz'; compiled.save(path)
            restored = type(compiled).load(path)
            np.testing.assert_array_equal(restored.probabilities_rows(rows, h, thresholds), expected)

    def test_source_filter_removes_collapsed_sources_but_preserves_valid_extremes(self):
        from portable_dc import EmpiricalCalibrator
        rows = np.tile(np.linspace(-1., 1., 100), (257, 1))
        targets = np.linspace(-3., 3., 257); rows[0] = 0.; targets[0] = 2.
        targets[1] = 1000.
        cal = EmpiricalCalibrator.fit_rows(rows, targets, np.zeros(257, dtype=int), horizon_length=12)
        self.assertEqual(cal.excluded_sources, 1)
        self.assertEqual(cal.retained_sources, 256)
        self.assertLess(cal.global_pool.max(), 10000.)
        self.assertGreater(cal.global_pool.max(), 1000.)
        self.assertNotIn(2e6, cal.global_pool)
        np.testing.assert_allclose(cal.mean_edges[1:-1], [0., 0., 0.], atol=1e-15)

    def test_pool_export_and_import_preserve_predictions_and_actual_horizon_ids(self):
        from portable_dc import EmpiricalCalibrator
        rng = np.random.default_rng(71)
        rows = np.tile(rng.normal(size=(1, 100)), (2400, 1)); h = np.arange(len(rows)) % 12
        y = rows.mean(1)+(h//3-1.5)*rows.std(1, ddof=1)
        cal = EmpiricalCalibrator.fit_rows(rows, y, h, horizon_length=12, n_horizon_bins=4)
        take = np.array([100, 8, 93, 26, 12])
        expected = cal.transform_rows(rows[take], h[take])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'calibrator.npz'; cal.save(path)
            restored = EmpiricalCalibrator.load(path)
            np.testing.assert_array_equal(expected, restored.transform_rows(rows[take], h[take]))
        self.assertFalse(np.allclose(expected[0], cal.transform_rows(rows[take[:1]], np.array([0]))[0]))


if __name__ == '__main__':
    unittest.main()
