"""The public batch adapter must retain paired populations across files."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np


class StreamInterfaceTests(unittest.TestCase):
    def run_case(self, omit_method=False, summary_bins=4):
        script = Path(__file__).with_name('evaluate_stream.py')
        self.assertTrue(script.exists(), 'Public forecast-batch evaluator is missing')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rng = np.random.default_rng(11)
            np.savez(root/'history.npz', targets=rng.normal(size=(100, 2)))
            cuts = np.linspace(-.2,.2,summary_bins-1).tolist()
            (root/'states.json').write_text(json.dumps(dict(mean_cuts=cuts, log_scale_cuts=cuts)))
            for i, size in enumerate([3, 2]):
                x = rng.normal(size=(size, 100, 4, 2)); y = rng.normal(size=(size, 4, 2))
                arrays = dict(targets=y, Native=x)
                if not (omit_method and i == 1):
                    arrays['DC16'] = x+.1
                np.savez(root/f'batch{i}.npz', **arrays)
            result = subprocess.run([sys.executable, str(script), '--history', str(root/'history.npz'),
                '--states', str(root/'states.json'), '--output', str(root/'scores.json'),
                str(root/'batch0.npz'), str(root/'batch1.npz')], capture_output=True, text=True)
            output = json.loads((root/'scores.json').read_text()) if (root/'scores.json').exists() else None
            return result, output

    def test_every_supplied_batch_is_scored_with_shared_states(self):
        result, output = self.run_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output['evaluation'], 'all supplied forecast batches')
        self.assertEqual(output['batch_files'], 2)
        a, b = output['methods']['Native'], output['methods']['DC16']
        self.assertEqual(a['coordinate_count'], 40)
        self.assertEqual(a['state_coverage']['counts'], b['state_coverage']['counts'])
        self.assertEqual(a['events']['observed_frequencies'], b['events']['observed_frequencies'])

    def test_missing_method_in_one_batch_fails_the_whole_evaluation(self):
        result, output = self.run_case(omit_method=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(output)

    def test_36_state_boundaries_preserve_the_complete_paired_population(self):
        result,output = self.run_case(summary_bins=6)
        self.assertEqual(result.returncode,0,result.stderr)
        counts = output['methods']['Native']['state_coverage']['counts']
        self.assertEqual(len(counts),36)
        self.assertEqual(sum(counts),40)
        self.assertEqual(counts,output['methods']['DC16']['state_coverage']['counts'])


if __name__ == '__main__':
    unittest.main()
