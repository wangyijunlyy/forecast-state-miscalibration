"""Protect the 36-state method and diagnostics from hidden 4x4 assumptions."""
import unittest
import numpy as np
from granularity import fit_partition
from reliability_metrics import ReliabilityMetrics, state_ids


class FSDC36Tests(unittest.TestCase):
    def test_portable_fitting_matches_the_actual_36_state_partition(self):
        from portable_dc import EmpiricalCalibrator
        rng = np.random.default_rng(3644)
        rows = rng.normal(size=(7000,100))*rng.uniform(.1,2,(7000,1))
        rows += rng.normal(size=(7000,1)); rows[:7] = 0
        h = np.arange(len(rows))%12; y = rng.normal(size=len(rows))
        actual,_ = fit_partition(rows,y,h,12,6,4)
        portable = EmpiricalCalibrator.fit_rows(rows,y,h,12,n_horizon_bins=4,summary_bins=6)
        np.testing.assert_array_equal(portable.mean_edges,actual.mean_edges)
        np.testing.assert_array_equal(portable.scale_edges,actual.scale_edges)
        np.testing.assert_array_equal(portable.transform_rows(rows[:80],h[:80]),actual.transform_rows(rows[:80],h[:80]))
        self.assertEqual(portable.compile().state_to_template.shape,(144,))

    def test_six_by_six_compilation_preserves_horizon_conditioned_predictions(self):
        rng = np.random.default_rng(3636)
        rows = rng.normal(size=(24000, 100))*rng.uniform(.1, 2, (24000, 1))
        rows += rng.normal(size=(24000, 1)); h = np.arange(len(rows)) % 12
        y = rows.mean(1)+rows.std(1, ddof=1)*(rng.normal(size=len(rows))+.1*h)
        cal, _ = fit_partition(rows, y, h, 12, 6, 4)
        compiled = cal.compile()
        query = rows[13:91]; qh = h[13:91]
        np.testing.assert_array_equal(compiled.transform_rows(query, qh), cal.transform_rows(query, qh))
        self.assertEqual(compiled.state_to_template.shape, (144,))

    def test_36_state_metrics_preserve_population_and_extreme_state_ids(self):
        edges = np.r_[-np.inf, np.arange(1, 6), np.inf]
        # Mean group 5 and scale group 5 must map to state 35.
        x = np.tile(np.linspace(-10., 10., 100), (2, 1))
        x = (x-x.mean(1)[:, None])/x.std(1, ddof=1)[:, None]*np.exp(5.5)+5.5
        samples = x[:, :, None, None]
        groups = state_ids(samples, edges, edges)
        np.testing.assert_array_equal(groups, [35, 35])
        metric = ReliabilityMetrics(np.array([[0., 1., 2.]]), n_states=36, min_state_count=1)
        metric.update(samples, np.zeros((2, 1, 1)), groups)
        result = metric.result()
        self.assertEqual(len(result['state_coverage']['counts']), 36)
        self.assertEqual(result['state_coverage']['counts'][35], 2)
        self.assertEqual(result['coordinate_count'], 2)


if __name__ == '__main__':
    unittest.main()
