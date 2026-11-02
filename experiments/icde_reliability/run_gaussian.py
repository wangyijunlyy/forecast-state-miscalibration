"""Full-test Gaussian postprocessing controls on the actual frozen backbones."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from run_task import ROOT, SOURCE, prepare, seed, atomic, state_ids, assert_traversal
from reliability_metrics import ReliabilityMetrics
from gaussian_models import fit_gaussian, predict

VERSION = 'full-gaussian-controls-v1-20261102'


def run(task):
    output = ROOT/'gaussian_runs'/task['id']; output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        previous = json.loads((output/'results.json').read_text())
        if previous['version'] != VERSION:
            raise ValueError('Gaussian result version mismatch')
        return
    original = json.loads((ROOT/'runs'/task['id']/'results.json').read_text())
    fitting = ROOT/'runs'/task['id']/'calibration.npz'
    if (output/'fitting.json').exists():
        fit = json.loads((output/'fitting.json').read_text())
        if fit['version'] != VERSION:
            raise ValueError('Gaussian fitting version mismatch')
    else:
        with np.load(fitting) as data:
            rows = data['rows']; target = data['targets']
        mu = rows.mean(1); sigma = rows.std(1, ddof=1).clip(1e-6); models = {}; timings = {}
        for name, affine in [('GaussianVariance2P', False), ('GaussianAffine4P', True)]:
            start = time.monotonic(); models[name] = fit_gaussian(mu, sigma, target, affine)
            timings[name] = time.monotonic()-start
            if not models[name]['optimizer_success'] or not np.isfinite(models[name]['fit_analytic_gaussian_crps']):
                atomic(output/'failed_fit.json', dict(version=VERSION, task=task, models=models))
                raise ValueError('Unsuccessful Gaussian fitting: '+name)
        fit = dict(version=VERSION, task=task, fitting_split='validation', calibration_coordinates=len(target),
                   coordinate_source=str(fitting), models=models, fit_seconds=timings)
        atomic(output/'fitting.json', fit)
    with np.load(fitting) as data:
        thresholds = data['thresholds']
    import pickle
    with (SOURCE/'runs'/task['id']/'calibrators.pkl').open('rb') as stream:
        partition = pickle.load(stream)['DC16']
    interface = prepare(task, output)
    names = ['Native', 'GaussianVariance2P', 'GaussianAffine4P']
    metrics = {name:ReliabilityMetrics(thresholds) for name in names}
    seed(task['seed']+30000); windows = steps = 0
    for step, batch in enumerate(interface['test'], 1):
        with torch.no_grad():
            x, y, _ = interface['predict'](batch, 'test', step)
        groups = state_ids(x, partition.mean_edges, partition.scale_edges)
        metrics['Native'].update(x, y, groups)
        b, s, h, c = x.shape; rows = x.transpose(0, 2, 3, 1).reshape(-1, s)
        for name in names[1:]:
            values = predict(fit['models'][name], rows).reshape(b, h, c, s).transpose(0, 3, 1, 2)
            metrics[name].update(values, y, groups, ordered=True)
        windows += len(y); steps = step
        if step % 20 == 0:
            print('GAUSSIAN_TEST', task['id'], step, windows, flush=True)
    assert_traversal(interface['test'], steps, windows)
    if windows != original['test_windows'] or steps != original['test_batches']:
        raise ValueError('Gaussian full-test iterator differs from primary suite')
    scores = {name:metric.result() for name,metric in metrics.items()}
    deltas = {key:scores['Native'][key]-original['methods']['Native'][key] for key in ['CRPS','MSE','QICE_percent']}
    if max(abs(v) for v in deltas.values()) > 1e-9:
        raise ValueError('Native Gaussian-control replay differs from primary forecasts')
    atomic(output/'results.json', dict(version=VERSION, task=task, full_test_split=True,
        checkpoint=interface['checkpoint'], test_windows=windows, test_batches=steps,
        test_seed=task['seed']+30000, methods=scores, fitting=fit, native_score_deltas=deltas,
        diagnostic_partition=original['diagnostic_partition']))
    print('GAUSSIAN_COMPLETE', task['id'], {k:v['CRPS'] for k,v in scores.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2); run(json.loads(args.task.read_text()))
