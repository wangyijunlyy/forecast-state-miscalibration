"""Full-stream scoring must preserve channel thresholds and global eligibility."""
import unittest
import numpy as np
from reliability_metrics import ReliabilityMetrics


class FullTestEvaluationTests(unittest.TestCase):
    def test_coordinate_chunks_match_complete_native_tensor(self):
        from full_test_evaluation import flatten_batch, update_coordinates
        rng = np.random.default_rng(81)
        samples = rng.normal(size=(3, 100, 64, 3))
        target = rng.normal(size=(3, 64, 3))
        thresholds = np.array([[-1., .2, .8], [.1, 1., 2.], [-2., -.1, .4]])
        groups = np.tile(np.arange(64) % 2, (3, 3, 1)).transpose(0, 2, 1).reshape(-1)
        reference = ReliabilityMetrics(thresholds, n_states=36)
        reference.update(samples, target, groups)
        rows, y, horizon, channels = flatten_batch(samples, target)
        np.testing.assert_array_equal(horizon, np.tile(np.repeat(np.arange(64), 3), 3))
        np.testing.assert_array_equal(channels, np.tile(np.arange(3), 3*64))
        actual = ReliabilityMetrics(thresholds, n_states=36)
        for begin in range(0, len(y), 47):
            ix = slice(begin, begin+47)
            update_coordinates(actual, rows[ix], y[ix], channels[ix], groups[ix], thresholds, False)
        want, got = reference.result(), actual.result()
        for metric in ['CRPS', 'MSE', 'MAE', 'RMSE', 'QICE_percent']:
            self.assertAlmostEqual(got[metric], want[metric], places=12)
        np.testing.assert_array_equal(got['state_coverage']['counts'], want['state_coverage']['counts'])
        self.assertEqual(got['state_coverage']['eligible_states'], 2)
        self.assertAlmostEqual(got['state_coverage']['state_error'], want['state_coverage']['state_error'], places=12)
        np.testing.assert_allclose(got['events']['brier'], want['events']['brier'], atol=1e-14)

    def test_incomplete_coordinates_fail_full_population_check(self):
        from full_test_evaluation import verify_population
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            verify_population(50000, 51000)
        verify_population(51000, 51000)

    def test_granularity_crps_stream_matches_independent_pairwise_score(self):
        from full_test_evaluation import CRPSAccumulator
        rng = np.random.default_rng(12)
        samples = np.sort(rng.lognormal(size=(91, 100)), axis=1)
        target = rng.normal(size=91)
        expected = np.mean(np.abs(samples-target[:, None]).mean(1)
            -.5*np.abs(samples[:, :, None]-samples[:, None, :]).mean((1, 2)))
        acc = CRPSAccumulator()
        acc.update(samples[:7], target[:7])
        acc.update(samples[7:], target[7:])
        self.assertAlmostEqual(acc.result()['CRPS'], expected, places=12)
        self.assertEqual(acc.result()['coordinate_count'], 91)


if __name__ == '__main__':
    unittest.main()
