"""Real-data parity and paired timing of exact compiled distribution queries."""
import argparse
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_dc import EmpiricalCalibrator, summaries

VERSION = 'compiled-dc16-queries-v1-20261102'


def atomic(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def run(task, root):
    directory = root/'runs'/task['id']; output = root/'compiled_deployment/runs'/task['id']
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        previous = json.loads((output/'results.json').read_text())
        if previous['version'] != VERSION:
            raise ValueError('Existing compiled benchmark version mismatch')
        return
    result = json.loads((directory/'results.json').read_text())
    if result['version'] != 'icde-reliability-v1-20261102' or not result['full_test_split']:
        raise ValueError('Primary result is incompatible or incomplete')
    with (directory/'calibrators.pkl').open('rb') as stream:
        original = pickle.load(stream)['DC16']
    meta = result['validation']['models']['DC16']
    portable = EmpiricalCalibrator(original.mean_edges, original.scale_edges, original.global_residuals,
        dict(original.bin_residuals), task['horizon'], 1, meta['retained_sources'], meta['excluded_sources'])
    start = time.perf_counter(); compiled = portable.compile(); compile_seconds = time.perf_counter()-start
    compiled.save(output/'compiled.npz')
    with np.load(directory/'test_reservoir.npz') as data:
        rows, target, horizon, channel = (data[k] for k in ['rows', 'targets', 'horizon', 'channel'])
    with np.load(directory/'calibration.npz') as data:
        thresholds = data['thresholds'][channel]
    mean, _, scale = summaries(rows)

    def reference_decode():
        return original.transform(rows[:, :, None, None])[:, :, 0, 0]

    def queries(values):
        return np.stack([(values > thresholds[:, j,None]).mean(1) for j in range(3)], axis=1)

    decoded = reference_decode(); new = compiled.transform_rows(rows, horizon)
    np.testing.assert_allclose(new, decoded, atol=1e-12, rtol=1e-12)
    maximum_difference = float(np.abs(new-decoded).max()); exact_values = bool(np.array_equal(new, decoded))
    reference_probabilities = queries(decoded)
    direct = compiled.probabilities_rows(rows, horizon, thresholds)
    cached = compiled.probabilities_summaries(mean, scale, horizon, thresholds)
    np.testing.assert_array_equal(direct, reference_probabilities)
    np.testing.assert_array_equal(cached, reference_probabilities)
    brier = ((direct-(target[:, None] > thresholds))**2).mean(0)
    control = json.loads((directory/'controls.json').read_text())
    np.testing.assert_allclose(brier, control['methods']['Budget50000']['events']['brier'], atol=1e-10, rtol=1e-10)
    del decoded, new, direct, cached
    operations = {
        'reference_decode': reference_decode,
        'compiled_decode': lambda: compiled.transform_rows(rows, horizon),
        'reference_decode_and_queries': lambda: queries(reference_decode()),
        'direct_queries_from_rows': lambda: compiled.probabilities_rows(rows, horizon, thresholds),
        'direct_queries_from_summaries': lambda: compiled.probabilities_summaries(mean, scale, horizon, thresholds),
    }
    times = {name:[] for name in operations}; rng = np.random.default_rng(20261102+task['seed'])
    for _ in range(3):
        for name in rng.permutation(list(operations)):
            start = time.perf_counter(); value = operations[name]()
            times[name].append(time.perf_counter()-start)
            del value
    atomic(output/'results.json', dict(version=VERSION, task=task,
        evaluation='saved uniform coordinate reservoir from the complete native test traversal',
        coordinates=len(rows), samples=100, thresholds_per_coordinate=3,
        complete_test_coordinates=result['methods']['Native']['coordinate_count'],
        exact_reconstructed_values=exact_values, maximum_value_difference=maximum_difference,
        exact_event_probabilities=True, brier=brier.tolist(), brier_matches_existing_control=True,
        warmup='parity evaluation of each implementation', repeats=3, times=times,
        median_seconds={k:float(np.median(v)) for k,v in times.items()},
        compilation_seconds=compile_seconds, original_pickle_bytes=len(pickle.dumps(original)),
        compiled_npz_bytes=(output/'compiled.npz').stat().st_size,
        compiled_template_count=len(compiled.quantiles),
        compiled_numeric_array_bytes=compiled.quantiles.nbytes+compiled.state_to_template.nbytes+compiled.mean_edges.nbytes+compiled.scale_edges.nbytes))
    print('COMPILED_COMPLETE', task['id'], len(rows), maximum_difference, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); tasks = json.loads((args.root/'manifest.json').read_text())
    tasks.sort(key=lambda t:(t['seed'], t['backbone'], t['dataset'], t['horizon']))
    pidfile = args.root/'compiled_deployment/controller.json'; pidfile.parent.mkdir(parents=True, exist_ok=True)
    for i, task in enumerate(tasks):
        atomic(pidfile, dict(pid=os.getpid(), task=task['id'], index=i, total=len(tasks)))
        run(task, args.root)
    atomic(pidfile, dict(pid=os.getpid(), stage='complete', total=len(tasks)))
