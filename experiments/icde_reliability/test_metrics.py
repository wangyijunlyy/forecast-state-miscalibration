"""Checks against hand-calculated scores and the historical full-stream metrics."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load_metrics():
    path = HERE / 'reliability_metrics.py'
    assert path.is_file(), 'Reliability metrics have not been implemented'
    spec = importlib.util.spec_from_file_location('icde_metrics', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReliabilityTests(unittest.TestCase):
    def test_complete_traversal_accepts_native_iterable_without_length(self):
        m = load_metrics()
        loader = iter([1, 2, 3])
        m.assert_traversal(loader, 3, 12)
        with self.assertRaises(ValueError):
            m.assert_traversal([1, 2, 3], 2, 8)

    def test_opposite_state_errors_cancel_but_brier_remains_nonzero(self):
        m = load_metrics()
        x = np.tile(np.array([-1., 0., 1., 2.])[None, :, None, None], (2, 1, 1, 1))
        y = np.array([-2., 3.]).reshape(2, 1, 1)
        acc = m.ReliabilityMetrics(np.array([[0., 1., 2.]]), min_state_count=1)
        acc.update(x, y, np.array([0, 1]))
        result = acc.result()
        self.assertAlmostEqual(result['CRPS'], 1.875)
        self.assertAlmostEqual(result['QICE_percent'], 16.)
        self.assertAlmostEqual(result['state_coverage']['state_error'], .5)
        self.assertAlmostEqual(result['state_coverage']['pooled_error'], 4/15)
        np.testing.assert_allclose(result['events']['brier'], [.25, .3125, .5])
        self.assertEqual(result['coordinate_count'], 2)

    def test_streaming_equals_historical_scores_and_batching(self):
        m = load_metrics()
        spec = importlib.util.spec_from_file_location('historical_metrics', HERE.parent/'three_seed_reproduction/core.py')
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        rng = np.random.default_rng(42)
        x = rng.normal(size=(5, 100, 7, 3)); y = rng.normal(size=(5, 7, 3))
        thresholds = np.array([[-1., 0., 1.]]*3)
        full = m.ReliabilityMetrics(thresholds, min_state_count=1)
        split = m.ReliabilityMetrics(thresholds, min_state_count=1)
        legacy = old.CommonMetrics()
        groups = np.arange(y.size) % 16
        full.update(x, y, groups); legacy.update(x, y)
        n = y[0].size
        for i in range(len(y)):
            split.update(x[i:i+1], y[i:i+1], groups[i*n:(i+1)*n])
        a, b, c = full.result(), split.result(), legacy.result()
        for key in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE', 'MedianMAE']:
            self.assertAlmostEqual(a[key], c[key], places=12)
            self.assertAlmostEqual(a[key], b[key], places=12)
        np.testing.assert_allclose(a['events']['brier'], b['events']['brier'], atol=1e-14)
        np.testing.assert_allclose(a['state_coverage']['coverage'], b['state_coverage']['coverage'])

    def test_sparse_groups_do_not_change_the_comparison_population(self):
        m = load_metrics()
        x = np.zeros((3, 100, 1, 1)); y = np.array([-1., 1., -1.]).reshape(3, 1, 1)
        acc = m.ReliabilityMetrics(np.zeros((1, 3)), min_state_count=2)
        acc.update(x, y, np.array([0, 0, 1]))
        r = acc.result()['state_coverage']
        self.assertEqual(r['eligible_coordinates'], 2)
        self.assertEqual(r['eligible_states'], 1)
        self.assertAlmostEqual(r['pooled_error'], r['state_error'])

    def test_nonfinite_forecasts_are_not_silently_removed(self):
        m = load_metrics()
        acc = m.ReliabilityMetrics(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            acc.update(np.full((1, 100, 1, 1), np.nan), np.zeros((1, 1, 1)), np.array([0]))


if __name__ == '__main__':
    unittest.main()
