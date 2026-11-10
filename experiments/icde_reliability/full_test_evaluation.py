"""Complete-test FSPC component evaluation with shared native draws.

Frozen main-method parameters are reused. Other settings use the existing
historical fitting reservoirs and fitting protocol, before test traversal.
"""
import argparse
import fcntl
import gzip
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from reliability_metrics import ReliabilityMetrics

VERSION = 'fspc-complete-test-components-v1-20261110'
DEFAULT = 'b6_h1_n50000'
SUM_FIELDS = ('n', 'crps', 'se', 'ae', 'medae', 'qice_counts', 'state_counts',
    'covered', 'brier', 'event_count', 'probability_bin_counts', 'probability_sums',
    'observation_sums', 'state_brier_sums')


class CRPSAccumulator:
    """Granularity tables require only CRPS, over the complete population."""
    def __init__(self):
        self.count = 0
        self.total = 0.
        self.weights = (2*np.arange(1, 101)-101)/10000

    def update(self, ordered, target):
        self.total += float((np.abs(ordered-target[:, None]).mean(1)-ordered@self.weights).sum())
        self.count += len(target)

    def result(self):
        if not self.count:
            raise ValueError('Empty CRPS population')
        return dict(CRPS=self.total/self.count, coordinate_count=self.count)


def flatten_batch(samples, target):
    b, s, h, c = samples.shape
    if s != 100 or target.shape != (b, h, c):
        raise ValueError('Expected paired [B,100,H,C] forecasts and [B,H,C] targets')
    return (samples.transpose(0, 2, 3, 1).reshape(-1, s), target.reshape(-1),
        np.tile(np.repeat(np.arange(h), c), b), np.tile(np.arange(c), b*h))


def update_coordinates(acc, rows, target, channel, groups, thresholds, ordered):
    part = ReliabilityMetrics(thresholds[channel], n_states=36)
    part.update(rows.T[None, :, None, :], target[None, None, :], groups, ordered=ordered)
    for field in SUM_FIELDS:
        setattr(acc, field, getattr(acc, field) + getattr(part, field))


def verify_population(actual, expected):
    if actual != expected:
        raise ValueError(f'Incomplete full test: {actual} coordinates, expected {expected}')


def load(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def fit_settings(task, root, source, means, aux, prior):
    from fspc_unified import UnifiedMean, fit_unified
    from fspc_granularity import VARIANTS
    from granularity import method_name
    from location_scale import replay_windows
    out = root/'runs'/task['id']
    frozen = out/'models.pkl.gz'
    if frozen.exists():
        with gzip.open(frozen, 'rb') as stream:
            return pickle.load(stream)
    ident = task['id']
    old = load(means/'runs'/ident/'results.json')
    provenance = old['provenance']
    diff = task['backbone'] == 'DiffusionTS'
    val_path = prior/'unclipped/runs'/ident/'val_calibration.npz' if diff else source/'runs'/ident/'calibration.npz'
    train_path = prior/'unclipped/runs'/ident/'train_calibration.npz' if diff else source/'train_calibration/runs'/ident/'calibration.npz'
    with np.load(val_path) as data:
        val, vy, vh, vc = [data[k] for k in ['rows', 'targets', 'horizon', 'channel']]
        if 'window' in data:
            window = data['window']
        else:
            window, hh, cc = replay_windows(provenance['calibration_windows'], task['horizon'],
                int(vc.max())+1, 50000, task['seed']+40000, provenance['batch_sizes'])
            np.testing.assert_array_equal(hh, vh)
            np.testing.assert_array_equal(cc, vc)
    with np.load(train_path) as data:
        train, ty, th = [data[k] for k in ['rows', 'targets', 'horizon']]
    base = UnifiedMean.from_record(load(aux/'components/runs'/ident/'parameters.json'))
    previous = load(aux/'components/runs'/ident/'results.json')
    models, resources = {}, {}
    for bins, groups, budget in VARIANTS:
        key = method_name(bins, groups, budget)
        if key == DEFAULT:
            models[key] = base
            resources[key] = previous['resources'][key]
        else:
            models[key], resources[key] = fit_unified(train, ty, th, val, vy, vh, window,
                provenance['calibration_windows'], provenance['origin_stride'], task['horizon'],
                bins=bins, horizon_groups=groups, budget=budget, ordering_seed=task['seed']+60000)
            # Reproduce the historical fitting objective without requiring a
            # particular optimizer argument on a flat CRPS optimum.
            expected = previous['resources'][key]['selection']['selected_crps']
            np.testing.assert_allclose(resources[key]['selection']['selected_crps'], expected,
                rtol=1e-9, atol=1e-9)
        atomic(out/'progress.json', dict(stage='fit', settings=len(models), expected=14))
    models['TrainFSPC'], resources['TrainFSPC'] = fit_unified(train, ty, th, val, vy, vh, window,
        provenance['calibration_windows'], provenance['origin_stride'], task['horizon'], template_split='train')
    bundle = dict(models=models, resources=resources, base=base,
        sources=dict(validation=str(val_path), training=str(train_path),
            default_parameters=str(aux/'components/runs'/ident/'parameters.json')))
    with gzip.open(frozen, 'wb') as stream:
        pickle.dump(bundle, stream, protocol=4)
    atomic(out/'frozen.json', dict(version=VERSION, task=task, test_targets_used_for_fitting=False,
        parameters_frozen_before_test=True, default_parameters_reused=True,
        settings=list(models), sources=bundle['sources']))
    return bundle


def run(task, root, source, means, aux, prior):
    from portable_dc import summaries, edges
    from mean_anchor import state_ids
    from final_fspc import CompiledFSPC
    from portable_dc import EmpiricalCalibrator
    from run_mean_improved_all import check_parity
    out = root/'runs'/task['id']
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (out/'results.json').exists():
        record = load(out/'results.json')
        assert record['version'] == VERSION and record['full_test_split']
        return
    start = time.monotonic()
    old = load(means/'runs'/task['id']/'results.json')
    assert old['full_test_split']
    atomic(out/'task.json', task)
    bundle = fit_settings(task, root, source, means, aux, prior)
    base, models = bundle['base'], bundle['models']
    compiled = {key: CompiledFSPC.from_model(model) for key, model in models.items()}
    default_compiled = compiled[DEFAULT]
    with np.load(source/'runs'/task['id']/'calibration.npz') as data:
        thresholds = data['thresholds']
    # Keep the diagnostic partition identical to the main complete-test study.
    with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as data:
        mean, _, scale = summaries(data['rows'])
    me, se = edges(mean, 6), edges(np.log(scale), 6)
    names = ['Native', 'MeanOnly', 'ShapeOnly', 'FSPC', 'TrainFSPC']
    accum = {name: ReliabilityMetrics(thresholds, n_states=36) for name in names}
    grain_scores = {key: CRPSAccumulator() for key in models if key not in [DEFAULT, 'TrainFSPC']}
    fallback = {key: 0 for key in models if key != 'TrainFSPC'}
    seen = 0

    def consume(rows, y, h, c):
        nonlocal seen
        for begin in range(0, len(y), 8192):
            ix = slice(begin, begin+8192)
            a, b, hh, cc = np.asarray(rows[ix], dtype=np.float64), y[ix], h[ix], c[ix]
            mu, _, sig = summaries(a)
            groups = 6*np.searchsorted(me[1:-1], mu, side='right') + np.searchsorted(se[1:-1], np.log(sig), side='right')
            ids, reference, delta = default_compiled.parts(mu, sig, hh)
            location = reference if delta is None else reference+delta
            arms = dict(Native=a, MeanOnly=a+(location-mu)[:, None],
                ShapeOnly=mu[:, None]+sig[:, None]*default_compiled.cal.quantiles[ids],
                FSPC=default_compiled.transform_summaries(mu, sig, hh))
            for name, values in arms.items():
                native_ordered = task['backbone'] == 'DeepSTUQ'
                if native_ordered and name == 'Native':
                    assert np.all(np.diff(values, axis=1) >= 0)
                update_coordinates(accum[name], values, b, cc, groups, thresholds,
                    native_ordered or name not in ['Native', 'MeanOnly'])
            del arms
            for key, model in models.items():
                if key != DEFAULT:
                    values = compiled[key].transform_summaries(mu, sig, hh)
                    if key == 'TrainFSPC':
                        update_coordinates(accum[key], values, b, cc, groups, thresholds, True)
                    else:
                        grain_scores[key].update(values, b)
                if key in fallback:
                    keys = EmpiricalCalibrator.keys(model.cal, mu, sig, hh)
                    bins = len(model.cal.mean_edges)-1
                    ids = (keys[:, 0]*bins+keys[:, 1])*bins+keys[:, 2]
                    counts = np.asarray(bundle['resources'][key]['source_counts'])
                    fallback[key] += int(np.count_nonzero(counts[ids] < 128))
            seen += len(b)

    cache = aux/'diffusion/runs'/task['id']/'test_reservoir.npz' if task['backbone'] == 'DiffusionTS' else source/'runs'/task['id']/'test_reservoir.npz'
    with np.load(cache) as data:
        cached_full = len(data['targets']) == old['methods']['Native']['coordinate_count']
        if cached_full:
            consume(*(data[k] for k in ['rows', 'targets', 'horizon', 'channel']))
    if cached_full:
        steps, windows = old['test_batches'], old['test_windows']
        execution = 'all full-test coordinates replayed from complete Native cache'
    else:
        import torch
        from audit_diffusion_sampling import find_native_model, disable_clamp
        from reliability_metrics import assert_traversal
        from run_mean_improved_all import guard_checkpoint
        guard_checkpoint(task)
        sys.path.insert(0, str(source/'code'))
        import run_fsdc36 as original
        import run_task as legacy
        torch.set_num_threads(2)
        locks = root/'prepare_locks'
        locks.mkdir(exist_ok=True)
        with (locks/(task['dataset'].lower()+'.lock')).open('a') as data_lock:
            fcntl.flock(data_lock, fcntl.LOCK_EX)
            interface = original.prepare(task, out)
        if task['backbone'] == 'DiffusionTS':
            disable_clamp(find_native_model(interface))
        legacy.seed(task['seed']+30000)
        steps = windows = 0
        for step, batch in enumerate(interface['test'], 1):
            with torch.no_grad():
                x, y, _ = interface['predict'](batch, 'test', step)
            consume(*flatten_batch(x, y))
            steps, windows = step, windows+len(y)
            if step == 1 or step % 10 == 0:
                atomic(out/'progress.json', dict(stage='complete_test', batches=step,
                    expected_batches=old['test_batches'], coordinates=seen,
                    expected_coordinates=old['methods']['Native']['coordinate_count'], elapsed=time.monotonic()-start))
                print('FULL_TEST', task['id'], step, old['test_batches'], flush=True)
        assert_traversal(interface['test'], steps, windows)
        execution = 'complete frozen Native test iterator; shared draws for every setting'
    verify_population(seen, old['methods']['Native']['coordinate_count'])
    scores = {name: acc.result() for name, acc in accum.items()}
    check_parity(scores['Native'], old['methods']['Native'])
    check_parity(scores['FSPC'], old['methods']['FSPCMean'])
    for name in names:
        verify_population(scores[name]['coordinate_count'], seen)
        np.testing.assert_array_equal(scores[name]['state_coverage']['counts'], scores['Native']['state_coverage']['counts'])
    np.testing.assert_allclose(scores['ShapeOnly']['MSE'], scores['Native']['MSE'], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(scores['MeanOnly']['MSE'], scores['FSPC']['MSE'], rtol=1e-9, atol=1e-9)
    variants = {key: scores['FSPC'] if key == DEFAULT else grain_scores[key].result() for key in fallback}
    for metric in variants.values():
        verify_population(metric['coordinate_count'], seen)
    atomic(out/'results.json', dict(version=VERSION, task=task, methods={key:scores[key] for key in ['Native', 'MeanOnly', 'ShapeOnly', 'FSPC', 'TrainFSPC']},
        variants=variants, granularity_metrics=['CRPS'], fallback_percent={key:100*value/seen for key,value in fallback.items()},
        full_test_split=True, evaluation_execution=execution, test_batches=steps, test_windows=windows,
        coordinate_count=seen, shared_coordinates=True, native_parity=True, frozen_fspc_parity=True,
        original_archive_reference=True, author_score_overlays_applied=False, neural_training_performed=False,
        sources=bundle['sources'], elapsed_seconds=time.monotonic()-start))
    print('FULL_TEST_COMPLETE', task['id'], seen, round(time.monotonic()-start, 2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'source', 'means', 'aux', 'prior', 'task']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    run(load(a.task), a.root, a.source, a.means, a.aux, a.prior)
