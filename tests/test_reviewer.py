import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'experiments/icde_reliability'))
from reviewer_pipeline import fit_suite, retained_evaluation
from mean_improved_all import fit_mean_models


def fixture():
    rng = np.random.default_rng(17)
    def coordinates(n):
        mean = rng.normal(size=n)
        return dict(rows=mean[:, None]+rng.normal(size=(n, 100)),
            targets=.8*mean+.2+rng.normal(size=n)*.7,
            horizon=np.arange(n)%12, channel=np.zeros(n, dtype=int))
    train = coordinates(1800); val = coordinates(1800); test = coordinates(160)
    val.update(window=np.arange(1800)%300, n_windows=np.array(300), origin_stride=np.array(1),
        thresholds=np.array([[0., 1., 2.]]))
    return train, val, test


class ReviewerTests(unittest.TestCase):
    def test_relocation_preserves_relative_bundle_paths(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('reviewer_entry', REPO/'scripts/reproduce.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        home = Path('/home/wyj/review/work/payload/home/wyj')
        value = "absolute='/home/wyj/SPARC'; relative=work/'payload/home/wyj'"
        rewritten = module.relocate_text(value, home)
        self.assertEqual(rewritten, f"absolute='{home}/SPARC'; relative=work/'payload/home/wyj'")

    def test_fresh_fit_matches_paper_mean_and_full_controls_are_exact(self):
        train, val, test = fixture()
        suite = fit_suite(train, val, 12, 300, 1, 1, controls=True)
        old, _ = fit_mean_models({'horizon':12}, val['rows'], val['targets'], val['window'],
            train['rows'], train['targets'], 300, 1)
        np.testing.assert_allclose(suite['model'].predict(test['rows'], test['horizon']),
            old['FSPCMean'].predict(test['rows']), atol=1e-12, rtol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            result = retained_evaluation(suite, test, val['thresholds'], directory, 1)
        self.assertEqual(len(result['controls']), 16)
        self.assertEqual(len(result['state_assignment']), 7)
        self.assertEqual(result['parity']['maximum_value_difference'], 0.)
        self.assertTrue(result['parity']['exact_boundary_ties'])
        self.assertTrue(result['parity']['saved_state_reload'])
        self.assertFalse(suite['metadata']['test_targets_used'])
        self.assertEqual(set(result['methods']), {'Native', 'MeanOnly', 'ShapeOnly', 'FSPC'})

    def test_cli_computes_metrics_without_historical_scores_or_test_target_leakage(self):
        train, val, test = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in [('train',train), ('val',val), ('test',test)]:
                np.savez(root/(name+'.npz'), **data)
            command = [sys.executable, str(REPO/'scripts/fspc_arrays.py'),
                '--train', str(root/'train.npz'), '--validation', str(root/'val.npz'),
                '--test', str(root/'test.npz'), '--horizon', '12', '--output']
            subprocess.run(command+[str(root/'one')], check=True, capture_output=True, text=True)
            test['targets'] = test['targets']+100
            np.savez(root/'test.npz', **test)
            subprocess.run(command+[str(root/'two')], check=True, capture_output=True, text=True)
            self.assertEqual((root/'one/parameters.json').read_text(), (root/'two/parameters.json').read_text())
            one = json.loads((root/'one/retained_results.json').read_text())
            two = json.loads((root/'two/retained_results.json').read_text())
            self.assertGreater(two['methods']['FSPC']['MSE'], one['methods']['FSPC']['MSE']+1000)

    def test_all_task_configurations_and_no_published_outputs(self):
        tasks = json.loads((REPO/'configs/tasks.json').read_text())
        self.assertEqual(len(tasks), 249)
        self.assertEqual(len({(t['backbone'],t['dataset'],t['horizon']) for t in tasks}), 83)
        self.assertEqual(len({t['backbone'] for t in tasks}), 7)
        self.assertEqual(len({t['id'] for t in tasks}), 249)
        self.assertEqual({t['seed'] for t in tasks}, {1,2,3})
        for rel in ['native_payload/home/wyj/K2VAE/probts/data/__init__.py',
                'native_payload/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf/probts/data/__init__.py',
                'native_payload/home/wyj/miniconda3/envs/k2vae/lib/python3.10/site-packages/torch_timeseries/dataset/__init__.py']:
            self.assertTrue((REPO/rel).is_file(), rel)
        # Ignore generated workspaces; inspect only checked-in source roots.
        for top in ['configs','scripts','docs','tests','experiments','native_payload','upstream_source_reconstruction','provenance']:
            for path in (REPO/top).rglob('*'):
                if path.is_file():
                    self.assertNotIn(path.suffix, {'.npz','.npy','.csv','.pdf','.pt','.pth','.ckpt','.pkl'})
                    self.assertNotEqual(path.name, 'results.json')


if __name__ == '__main__':
    unittest.main()
