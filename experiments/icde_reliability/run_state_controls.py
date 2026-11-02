"""All-checkpoint state-assignment controls on the fixed retained test cohorts."""
import argparse
import copy
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_dc import EmpiricalCalibrator, bounded
from reliability_metrics import ReliabilityMetrics

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))
TRAIN = Path('/home/wyj/source_filtered_reproduction/runs')
VERSION = 'state-assignment-controls-v1-20261102'


def atomic(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def run(task):
    identity = task['id']; source = ROOT/'runs'/identity; output = ROOT/'state_controls/runs'/identity
    output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        existing = json.loads((output/'results.json').read_text())
        if existing['version'] != VERSION:
            raise ValueError('State-control version mismatch')
        return
    with (source/'calibrators.pkl').open('rb') as stream:
        models = pickle.load(stream)
    original = models['DC16']
    with (TRAIN/identity/'calibrators.pkl').open('rb') as stream:
        partition = pickle.load(stream)['DC16']
    with np.load(source/'calibration.npz') as data:
        fitting, target, horizon = (data[k] for k in ['rows', 'targets', 'horizon'])
        threshold_table = data['thresholds']
    with np.load(source/'test_reservoir.npz') as data:
        rows, y, test_h, channels = (data[k] for k in ['rows', 'targets', 'horizon', 'channel'])
    mean = fitting.mean(1); raw = fitting.std(1, ddof=1); scale = raw.clip(1e-6)
    residual = (target-mean)/scale; valid = raw > 1.01e-6
    np.testing.assert_array_equal(bounded(residual[valid]), original.global_residuals)
    base = EmpiricalCalibrator(original.mean_edges, original.scale_edges, original.global_residuals,
        dict(original.bin_residuals), task['horizon'], 1, int(valid.sum()), int((~valid).sum()))
    fit_keys = base.keys(mean, scale, horizon); unique, inverse = np.unique(fit_keys, axis=0, return_inverse=True)
    mu = rows.mean(1); sigma = rows.std(1, ddof=1).clip(1e-6)
    groups = 4*np.searchsorted(partition.mean_edges[1:-1], mu, side='right')+np.searchsorted(partition.scale_edges[1:-1], np.log(sigma), side='right')
    coordinate_thresholds = threshold_table[channels]

    def score(values, ordered):
        # Each retained scalar coordinate is a column of this scoring view.
        # Its threshold is still indexed by its original channel; horizon IDs
        # have already been used by the actual calibrator before scoring.
        accumulator = ReliabilityMetrics(coordinate_thresholds)
        accumulator.update(values.T[None, :, None, :], y[None, None, :], groups, ordered=ordered)
        return accumulator.result()

    scores = {'Native':score(rows, False)}
    compiled = base.compile(); compiled.save(output/'genuine.npz')
    scores['DC16'] = score(compiled.transform_rows(rows, test_h), True)
    global_q = np.quantile(models['GlobalDC'].global_residuals, (np.arange(100)+.5)/100)
    scores['GlobalDC'] = score(mu[:, None]+sigma[:, None]*global_q, True)
    control = json.loads((source/'controls.json').read_text())['methods']['Budget50000']
    for key in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE']:
        np.testing.assert_allclose(scores['DC16'][key], control[key], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(scores['DC16']['events']['brier'], control['events']['brier'], atol=1e-10, rtol=1e-10)
    permutation_seeds = []
    for repeat in range(5):
        value = task['seed']+80000+repeat; permutation_seeds.append(value)
        shuffled = residual.copy(); shuffled[valid] = residual[valid][np.random.default_rng(value).permutation(valid.sum())]
        pools = {tuple(key):bounded(shuffled[(inverse == i)&valid])
                 for i,key in enumerate(unique) if np.sum((inverse == i)&valid) >= 128}
        if pools.keys() != base.local_pools.keys():
            raise ValueError('Shuffling changed local source counts or fallback eligibility')
        altered = EmpiricalCalibrator(base.mean_edges, base.scale_edges, base.global_pool, pools,
            base.horizon_length, 1, base.retained_sources, base.excluded_sources)
        compiled = altered.compile(); compiled.save(output/f'shuffled{repeat}.npz')
        scores[f'Shuffled{repeat}'] = score(compiled.transform_rows(rows, test_h), True)
    atomic(output/'results.json', dict(version=VERSION, task=task,
        evaluation='shared uniform retained coordinates after complete native test traversal',
        coordinates=len(rows), fitting_coordinates=len(fitting), source_channel_count=len(threshold_table),
        retained_sources=int(valid.sum()), excluded_sources=int((~valid).sum()),
        fixed_source_mask=True, identical_global_pool=True, identical_state_counts=True,
        permutation_seeds=permutation_seeds, diagnostic_partition='training-fitted native mean/scale boundaries',
        genuine_matches_existing_control=True, methods=scores))
    print('STATE_CONTROL_COMPLETE',identity,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args(); tasks = json.loads((ROOT/'manifest.json').read_text())
    errors = []; controller = ROOT/'state_controls/controller.json'; controller.parent.mkdir(parents=True,exist_ok=True)
    atomic(controller,dict(pid=os.getpid(),stage='running',expected_tasks=len(tasks)))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run,t):t['id'] for t in tasks}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(dict(task=futures[future],error=repr(error))); print('STATE_CONTROL_FAILED',futures[future],repr(error),flush=True)
    atomic(controller,dict(pid=os.getpid(),stage='complete' if not errors else 'failed',expected_tasks=len(tasks),errors=errors))
    if errors:
        raise SystemExit(1)
