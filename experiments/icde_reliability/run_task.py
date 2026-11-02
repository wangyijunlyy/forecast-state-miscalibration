"""Paired complete-iterator evaluation of forecast-state reliability."""
import argparse
import copy
import json
import os
import pickle
import random
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))
OLD = Path('/home/wyj/three_seed_reproduction_20261031')
SOURCE = Path('/home/wyj/source_filtered_reproduction')
sys.path.insert(0, str(OLD/'code'))
sys.path.insert(0, '/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf')
sys.path.insert(0, '/home/wyj/distribution_binned_diffusion/code')
from interfaces import prepare
from core import atomic
from reservoir import CalibrationReservoir
from sparc_adapters.posthoc_baselines import DistributionCalibrationCalibrator, ConditionalPITCalibrator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reliability_metrics import ReliabilityMetrics, TargetReservoir, state_ids, assert_traversal

VERSION = 'icde-reliability-v1-20261102'
METHODS = ['Native', 'TrainDC16', 'DC16', 'GlobalDC', 'GlobalPIT', 'ConditionalPIT']


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def bounded(a):
    a = np.sort(np.asarray(a, dtype=np.float64))
    return a if len(a) <= 20000 else a[np.linspace(0, len(a)-1, 20000).astype(int)]


def fit_dc(rows, y, h, horizon, n_mean=4, n_scale=4, n_horizon=1, source_filter=True):
    mu = rows.mean(1); raw = rows.std(1, ddof=1); sigma = raw.clip(1e-6)
    z = (y-mu)/sigma
    cal = DistributionCalibrationCalibrator.fit_summaries(z, mu, sigma, h, horizon,
        n_mean_bins=n_mean, n_scale_bins=n_scale, n_horizon_bins=n_horizon,
        min_bin_size=128, max_pool_size=20000)
    kept = raw > 1.01e-6 if source_filter else np.ones(len(z), dtype=bool)
    if not kept.any():
        raise ValueError('All fitting sources have collapsed scale')
    if not kept.all():
        keys = np.stack([np.minimum(h*n_horizon//horizon, n_horizon-1),
            np.searchsorted(cal.mean_edges[1:-1], mu, side='right'),
            np.searchsorted(cal.scale_edges[1:-1], np.log(sigma), side='right')], axis=1)
        unique, inv = np.unique(keys, axis=0, return_inverse=True)
        cal.global_residuals = bounded(z[kept])
        cal.bin_residuals = {tuple(int(v) for v in key): bounded(z[(inv == i)&kept])
            for i, key in enumerate(unique) if np.sum((inv == i)&kept) >= 128}
    return cal, dict(retained_sources=int(kept.sum()), excluded_sources=int((~kept).sum()),
        calibration_coordinates=len(y), local_pools=len(cal.bin_residuals))


def fit_models(task, rows, y, h, channels):
    models = {}; metadata = {}; start = time.monotonic()
    for name, options in [('DC16', {}), ('GlobalDC', dict(n_mean=1, n_scale=1))]:
        begin = time.monotonic()
        models[name], metadata[name] = fit_dc(rows, y, h, task['horizon'], **options)
        metadata[name]['fit_seconds'] = time.monotonic()-begin
    sigma = rows.std(1, ddof=1).clip(1e-6); pit = (rows <= y[:, None]).mean(1)
    for name, bins in [('GlobalPIT', 1), ('ConditionalPIT', 4)]:
        begin = time.monotonic()
        models[name] = ConditionalPITCalibrator.fit_summaries(pit, sigma, h, channels,
            horizon_length=task['horizon'], channel_count=int(channels.max())+1,
            n_horizon_bins=bins, n_scale_bins=bins, n_channel_bins=bins,
            min_bin_size=128, n_knots=41)
        metadata[name] = dict(fit_seconds=time.monotonic()-begin, calibration_coordinates=len(y))
    return models, metadata, time.monotonic()-start


def collect_validation(task, out, interface, train_cal):
    seed(task['seed']+20000)
    pool = CalibrationReservoir(50000, task['seed']+40000)
    targets = TargetReservoir(4096, task['seed']+50000)
    diagnostic = None; windows = steps = fit_windows = 0; native_seconds = 0.
    for step, batch in enumerate(interface['val'], 1):
        torch.cuda.synchronize(); begin = time.monotonic()
        with torch.no_grad():
            x, y, _ = interface['predict'](batch, 'validation', step)
        torch.cuda.synchronize(); native_seconds += time.monotonic()-begin
        targets.update(y)
        if step <= interface['fit_limit']:
            pool.update(x, y); fit_windows += len(y)
        if diagnostic is None:
            diagnostic = {name: ReliabilityMetrics(np.zeros((y.shape[-1], 3))) for name in ['Native', 'TrainDC16']}
        groups = state_ids(x, train_cal.mean_edges, train_cal.scale_edges)
        diagnostic['Native'].update(x, y, groups)
        diagnostic['TrainDC16'].update(train_cal.transform(x), y, groups, ordered=True)
        windows += len(y); steps = step
        if step % 20 == 0:
            print('VALIDATION', task['id'], step, windows, flush=True)
    assert_traversal(interface['val'], steps, windows)
    rows, y, h, c = pool.arrays()
    models, metadata, fit_seconds = fit_models(task, rows, y, h, c)
    thresholds = targets.thresholds()
    np.savez_compressed(out/'calibration.npz', rows=rows, targets=y, horizon=h, channel=c, thresholds=thresholds)
    with (out/'calibrators.pkl').open('wb') as stream:
        pickle.dump(models, stream)
    scores = {name: acc.result() for name, acc in diagnostic.items()}
    for result in scores.values():
        del result['events']  # Validation diagnostics do not evaluate retrospectively fitted event thresholds.
    record = dict(version=VERSION, validation_complete=True, full_validation_split=True,
        validation_windows=windows, validation_batches=steps, calibration_windows=fit_windows,
        retained_calibration_coordinates=len(y), calibration_seen_coordinates=pool.seen_points,
        calibration_fit_limit=interface['fit_limit'], calibration_split='validation',
        checkpoint_selection_uses_validation=True, models=metadata, fit_seconds=fit_seconds,
        native_generation_seconds=native_seconds, diagnostics=scores,
        event_threshold_quantiles=[.8, .9, .95], event_threshold_source='historical validation targets',
        threshold_rows_seen=targets.seen, threshold_rows_retained=len(targets.values),
        calibrator_pickle_bytes=(out/'calibrators.pkl').stat().st_size)
    atomic(out/'validation.json', record)
    return models, thresholds, record


def array_forecast(rows):
    return rows[:, :, None, None]


def controls(task, out, test_pool, train_cal, thresholds):
    fitting = np.load(out/'calibration.npz')
    rows, y, h, c = (fitting[k] for k in ['rows', 'targets', 'horizon', 'channel'])
    test_rows, test_y, test_h, test_c = test_pool.arrays()
    # Preserve each coordinate's real horizon/channel for transforms and event thresholds.
    # Original calibrator transforms infer h/c from tensor layout, so reconstruct by pool key below.
    results = {}; cost = {}
    variants = [(f'Budget{budget}', dict(budget=min(budget, len(y)))) for budget in [2000, 10000, 50000]]
    variants += [('UnfilteredDC16', dict(source_filter=False)), ('DC64', dict(n_horizon=4))]
    rng = np.random.default_rng(task['seed']+60000); order = rng.permutation(len(y))
    test_mu = test_rows.mean(1); test_sigma = test_rows.std(1, ddof=1).clip(1e-6)
    groups = 4*np.searchsorted(train_cal.mean_edges[1:-1], test_mu, side='right') + np.searchsorted(train_cal.scale_edges[1:-1], np.log(test_sigma), side='right')
    for name, options in variants:
        options = dict(options); budget = options.pop('budget', len(y)); take = order[:budget]
        begin = time.monotonic()
        cal, meta = fit_dc(rows[take], y[take], h[take], task['horizon'], **options)
        fit_time = time.monotonic()-begin
        keys = np.stack([np.minimum(test_h*cal.n_horizon_bins//task['horizon'], cal.n_horizon_bins-1),
            np.searchsorted(cal.mean_edges[1:-1], test_mu, side='right'),
            np.searchsorted(cal.scale_edges[1:-1], np.log(test_sigma), side='right')], axis=1)
        values = np.empty_like(test_rows); grid = (np.arange(test_rows.shape[1])+.5)/test_rows.shape[1]
        begin = time.monotonic()
        for key in np.unique(keys, axis=0):
            mask = np.all(keys == key, axis=1)
            q = np.quantile(cal.bin_residuals.get(tuple(key), cal.global_residuals), grid)
            values[mask] = test_mu[mask, None]+test_sigma[mask, None]*q
        predict_time = time.monotonic()-begin
        # Accumulate with actual per-channel thresholds despite arbitrary retained row ordering.
        acc = ReliabilityMetrics(thresholds)
        for channel in np.unique(test_c):
            ix = test_c == channel
            local = ReliabilityMetrics(thresholds[channel:channel+1])
            local.update(array_forecast(values[ix]), test_y[ix, None, None], groups[ix], ordered=True)
            if not acc.n:
                acc = local
            else:
                for attr in ['n', 'crps', 'se', 'ae', 'medae', 'qice_counts', 'state_counts', 'covered', 'brier',
                             'event_count', 'probability_bin_counts', 'probability_sums', 'observation_sums', 'state_brier_sums']:
                    setattr(acc, attr, getattr(acc, attr)+getattr(local, attr))
        results[name] = acc.result()
        cost[name] = dict(**meta, fit_seconds=fit_time, transform_seconds=predict_time,
            calibrator_bytes=len(pickle.dumps(cal)), calibration_budget=budget)
    atomic(out/'controls.json', dict(version=VERSION, evaluation='uniform reservoir after full test traversal',
        retained_test_coordinates=len(test_y), full_test_coordinates_seen=test_pool.seen_points,
        methods=results, resources=cost))


def run(task, out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        result = json.loads((out/'results.json').read_text())
        if result.get('version') == VERSION and (out/'controls.json').exists():
            return
        raise ValueError('Existing output has an incompatible or incomplete identity')
    interface = prepare(task, out)
    with (SOURCE/'runs'/task['id']/'calibrators.pkl').open('rb') as stream:
        train_cal = pickle.load(stream)['DC16']
    if (out/'validation.json').exists():
        validation = json.loads((out/'validation.json').read_text())
        assert validation['version'] == VERSION
        with (out/'calibrators.pkl').open('rb') as stream:
            models = pickle.load(stream)
        thresholds = np.load(out/'calibration.npz')['thresholds']
    else:
        models, thresholds, validation = collect_validation(task, out, interface, train_cal)
    models['TrainDC16'] = train_cal
    accumulators = {name: ReliabilityMetrics(thresholds) for name in METHODS}
    timings = {name: 0. for name in METHODS}; metric_seconds = {name: 0. for name in METHODS}
    pool = CalibrationReservoir(50000, task['seed']+70000)
    seed(task['seed']+30000); windows = steps = 0; start = time.monotonic()
    for step, batch in enumerate(interface['test'], 1):
        torch.cuda.synchronize(); begin = time.monotonic()
        with torch.no_grad():
            x, y, _ = interface['predict'](batch, 'test', step)
        torch.cuda.synchronize(); timings['Native'] += time.monotonic()-begin
        groups = state_ids(x, train_cal.mean_edges, train_cal.scale_edges)
        pool.update(x, y)
        for name in METHODS:
            begin = time.monotonic()
            values = x if name == 'Native' else models[name].transform(x)
            if name != 'Native':
                timings[name] += time.monotonic()-begin
            begin = time.monotonic()
            accumulators[name].update(values, y, groups, ordered=name != 'Native')
            metric_seconds[name] += time.monotonic()-begin
            del values
        windows += len(y); steps = step
        if step % 20 == 0:
            print('TEST', task['id'], step, windows, round(time.monotonic()-start, 1), flush=True)
    old = json.loads((SOURCE/'runs'/task['id']/'results.json').read_text())
    assert_traversal(interface['test'], steps, windows)
    if steps != old['test_batches'] or windows != old['test_windows']:
        raise ValueError('Incomplete or mismatched native test iterator')
    results = {name: acc.result() for name, acc in accumulators.items()}
    deltas = {key: results['Native'][key]-old['common_normalized']['Native'][key] for key in ['CRPS', 'MSE', 'QICE_percent']}
    rows, y, h, c = pool.arrays()
    np.savez_compressed(out/'test_reservoir.npz', rows=rows, targets=y, horizon=h, channel=c)
    controls(task, out, pool, train_cal, thresholds)
    result = dict(version=VERSION, task=task, checkpoint=interface['checkpoint'], full_test_split=True,
        test_windows=windows, test_batches=steps, test_seed=task['seed']+30000,
        methods=results, validation=validation, native_score_deltas=deltas,
        generation_or_transform_seconds=timings, metric_seconds=metric_seconds,
        native_generation_excludes_preparation=True, total_test_seconds=time.monotonic()-start,
        maximum_resident_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        diagnostic_partition='training-fitted DC16 mean/scale boundaries shared by all methods')
    atomic(out/'results.json', result)
    print('COMPLETE', task['id'], {name: r['CRPS'] for name, r in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--task', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    task = json.loads(args.task.read_text())
    run(task, ROOT/'runs'/task['id'])
