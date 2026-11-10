"""Changed fitting split must leave the full FSPC reconstruction intact."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    from train_fspc import fit_training_variant
except ImportError:
    fit_training_variant = None
from portable_dc import EmpiricalCalibrator
from mean_anchor import centered_forecasts
try:
    from run_train_fspc import cohort_groups
except ImportError:
    cohort_groups = None
from portable_dc import CompiledCalibrator
try:
    from aggregate_train_fspc import audit_training_sources, template_record
except ImportError:
    audit_training_sources = None
try:
    from run_train_fspc import try_gpu_permit
except ImportError:
    try_gpu_permit = None


class TrainFSPCTests(unittest.TestCase):
    def test_template_proof_binds_horizon_semantics(self):
        cal = CompiledCalibrator(np.array([-np.inf,np.inf]),np.array([-np.inf,np.inf]),
            np.zeros((1,100)),np.array([0]),12,1)
        record = template_record(cal)
        self.assertEqual(record.get('horizon_length'),12)
        self.assertEqual(record.get('n_horizon_bins'),1)

    def test_gpu_permits_remain_bounded_across_controller_restarts(self):
        self.assertIsNotNone(try_gpu_permit, 'Persistent GPU permits are not implemented')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = try_gpu_permit(root,'0'); second = try_gpu_permit(root,'0')
            self.assertIsNotNone(first); self.assertIsNotNone(second)
            self.assertIsNone(try_gpu_permit(root,'0'))
            first.close()
            replacement = try_gpu_permit(root,'0')
            self.assertIsNotNone(replacement)
            second.close(); replacement.close()

    def test_cohort_diagnostics_use_the_frozen_comparison_partition(self):
        self.assertIsNotNone(cohort_groups, 'Paired cohort diagnostic partition is not implemented')
        rows = np.tile(np.linspace(-1, 1, 100), (3, 1)) + np.array([-2, 0, 2])[:, None]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'compiled_deployment/runs/example/compiled.npz'
            path.parent.mkdir(parents=True)
            cal = CompiledCalibrator(np.array([-np.inf,-3,-1,1,3,4,np.inf]),
                np.array([-np.inf,-5,-4,-3,0,1,np.inf]), np.zeros((1,100)), np.zeros(36,dtype=int),12,1)
            cal.save(path)
            np.testing.assert_array_equal(cohort_groups(rows, root, {'id':'example'}), [9,15,21])

    def test_training_templates_and_bias_are_isolated_from_validation(self):
        self.assertIsNotNone(fit_training_variant, 'Training-fit FSPC is not implemented')
        rng = np.random.default_rng(71)
        rows = rng.normal(size=(1800, 100)) * np.linspace(.2, 1.7, 1800)[:, None]
        rows += np.linspace(-2, 2, 1800)[:, None]
        rows[:200] = .5  # Collapsed sources are excluded from templates, not raw mean bias.
        y = rows.mean(1) + 2
        h = np.arange(len(y)) % 12
        validation = rows[200:].copy()
        val_h = h[200:]
        positive = validation.mean(1) + .3
        negative = validation.mean(1) - .3
        first = fit_training_variant(rows, y, h, validation, positive, val_h, 12)
        second = fit_training_variant(rows, y, h, validation, negative, val_h, 12)
        expected = EmpiricalCalibrator.fit_rows(rows, y, h, 12, summary_bins=6).compile()
        for cal, bias, strength, meta in [first, second]:
            np.testing.assert_array_equal(cal.mean_edges, expected.mean_edges)
            np.testing.assert_array_equal(cal.scale_edges, expected.scale_edges)
            np.testing.assert_array_equal(cal.quantiles, expected.quantiles)
            np.testing.assert_array_equal(cal.state_to_template, expected.state_to_template)
            self.assertEqual(sum(meta['training_state_counts']), len(y))
            self.assertEqual(meta['excluded_sources'], 200)
            self.assertEqual(meta['template_fit_split'], 'train')
            self.assertEqual(meta['selection']['selection_split'], 'validation')
            self.assertFalse(meta['selection']['test_targets_used'])
            out = centered_forecasts(cal, validation, val_h, bias, strength)
            np.testing.assert_allclose(np.diff(out, axis=1),
                np.diff(cal.transform_rows(validation, val_h), axis=1), atol=1e-12)
        np.testing.assert_array_equal(first[1], second[1])
        self.assertGreater(first[2], 0)
        self.assertEqual(second[2], 0)

    def test_independent_source_audit_detects_changed_parameters(self):
        self.assertIsNotNone(audit_training_sources, 'Independent training source audit is not implemented')
        rng = np.random.default_rng(43)
        rows = rng.normal(size=(1600, 100))
        y = rows.mean(1) + .8
        h = np.arange(len(y)) % 12
        cal, bias, strength, fit = fit_training_variant(rows, y, h, rows, y-.4, h, 12)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for part, targets in [('train_calibration', y), ('', y-.4)]:
                path = source/part/'runs/example/calibration.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, rows=rows, targets=targets, horizon=h)
            audit_training_sources({'id':'example', 'horizon':12}, fit, cal, source)
            changed = copy.deepcopy(fit); changed['state_bias'][0] += 1
            with self.assertRaises(AssertionError):
                audit_training_sources({'id':'example', 'horizon':12}, changed, cal, source)
            changed_cal = copy.deepcopy(cal); changed_cal.quantiles[0, 0] += 1
            with self.assertRaises(AssertionError):
                audit_training_sources({'id':'example', 'horizon':12}, fit, changed_cal, source)


if __name__ == '__main__':
    unittest.main()
