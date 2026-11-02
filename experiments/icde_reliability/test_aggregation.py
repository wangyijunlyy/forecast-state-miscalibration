import importlib.util
import sys
import unittest
from pathlib import Path


def module():
    path = Path(__file__).with_name('aggregation.py')
    assert path.exists(), 'Independent seed/configuration aggregation is missing'
    spec = importlib.util.spec_from_file_location('icde_aggregation', path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return m


class AggregationTests(unittest.TestCase):
    def test_configurations_average_within_seed_before_seed_statistics(self):
        m = module()
        rows = [dict(backbone='A', dataset=d, horizon=12, seed=s, method='Native', value=v)
                for d, values in [('x', [1, 2, 3]), ('y', [9, 12, 15])]
                for s, v in enumerate(values, 1)]
        result = m.backbone_statistics(rows, ['value'])[0]
        self.assertEqual(result['n_configurations'], 2)
        self.assertEqual(result['value_mean'], 7.)
        self.assertEqual(result['value_std'], 2.)
        self.assertEqual(result['n_training_seeds'], 3)

    def test_missing_seed_is_not_turned_into_a_three_seed_summary(self):
        m = module()
        rows = [dict(backbone='A', dataset='x', horizon=12, seed=s, method='Native', value=1.) for s in [1, 2]]
        self.assertEqual(m.backbone_statistics(rows, ['value']), [])

    def test_duplicate_task_method_identity_is_rejected(self):
        m = module()
        rows = [dict(backbone='A', dataset='x', horizon=12, seed=s, method='Native', value=1.) for s in [1, 2, 3]]
        with self.assertRaises(ValueError):
            m.backbone_statistics(rows+[rows[0]], ['value'])

    def test_state_summary_uses_the_same_fully_eligible_configurations_in_all_seeds(self):
        m = module()
        rows = [dict(backbone='A', dataset=d, horizon=12, seed=s, method='Native', value=v)
                for d, values in [('x', [1., None, 3.]), ('y', [9., 12., 15.])]
                for s, v in enumerate(values, 1)]
        result = m.backbone_statistics(rows, ['value'])[0]
        self.assertEqual(result['n_configurations'], 2)
        self.assertEqual(result['value_n_configurations'], 1)
        self.assertEqual(result['value_mean'], 12.)
        self.assertEqual(result['value_std'], 3.)


if __name__ == '__main__':
    unittest.main()
