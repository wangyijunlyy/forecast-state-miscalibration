"""Evaluate only Train FSPC on the original complete native test iterators."""
import argparse
import fcntl
import json
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from train_fspc import VERSION, METHODS, fit_training_variant, validate_record, verify_training_template
from mean_anchor import centered_forecasts, state_ids as bias_ids
from reliability_metrics import ReliabilityMetrics, state_ids, assert_traversal
from portable_dc import summaries, CompiledCalibrator


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def try_gpu_permit(root, gpu):
    directory = root/'gpu_permits'; directory.mkdir(exist_ok=True)
    for slot in range(2):
        handle = (directory/('gpu'+gpu+'_'+str(slot)+'.lock')).open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            handle.close()
    return None


def cohort_groups(rows, source, task):
    diagnostic = CompiledCalibrator.load(source/'compiled_deployment/runs'/task['id']/'compiled.npz')
    return state_ids(rows.T[None, :, None, :], diagnostic.mean_edges, diagnostic.scale_edges)


def evaluate_cohort(task, out, source, mean_root, cal, bias, strength, fit, thresholds):
    old_cohort = json.loads((source/'cohort/runs'/task['id']/'results.json').read_text())
    old_mean_cohort = json.loads((mean_root/'runs'/task['id']/'cohort.json').read_text())
    assert old_cohort['task'] == old_mean_cohort['task'] == task
    with np.load(source/'runs'/task['id']/'test_reservoir.npz') as data:
        rows, y, h, channels = [data[k] for k in ['rows', 'targets', 'horizon', 'channel']]
    groups = cohort_groups(rows, source, task)
    cohort = dict(Native=old_cohort['native'], DC36=old_cohort['methods']['b6_h1_n50000'],
        DC36ControlledMean=old_mean_cohort['methods']['DC36ControlledMean'])
    for name in ['TrainDC36', 'TrainFSPC']:
        forecast = (cal.transform_rows(rows, h) if name == 'TrainDC36'
                    else centered_forecasts(cal, rows, h, bias, strength))
        metric = ReliabilityMetrics(thresholds[channels], n_states=36)
        metric.update(forecast.T[None, :, None, :], y[None, None, :], groups, ordered=True)
        cohort[name] = metric.result(); del forecast
        np.testing.assert_array_equal(cohort[name]['state_coverage']['counts'],
                                      cohort['Native']['state_coverage']['counts'])
    atomic(out/'cohort.json', dict(version=VERSION, task=task, fit=fit,
        evaluation='shared retained test cohort', coordinates=len(y), methods=cohort,
        baseline_metrics_recomputed=False, newly_evaluated_methods=['TrainDC36', 'TrainFSPC'],
        diagnostic_partition='original validation-fitted 6x6 cohort boundaries'))
    del rows, y, h, channels, cohort
    print('TRAIN_FSPC_COHORT_COMPLETE', task['id'], 'strength', strength, flush=True)


def run(task, root, source, mean_root):
    out = root/'runs'/task['id']; out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        validate_record(json.loads((out/'results.json').read_text()), task); return
    start = time.monotonic()
    original = json.loads((source/'runs'/task['id']/'results.json').read_text())
    mean_original = json.loads((mean_root/'runs'/task['id']/'results.json').read_text())
    assert original['task'] == mean_original['task'] == task
    train_path = source/'train_calibration/runs'/task['id']/'calibration.npz'
    val_path = source/'runs'/task['id']/'calibration.npz'
    with np.load(train_path) as data:
        train, targets, horizon = [data[k] for k in ['rows', 'targets', 'horizon']]
    with np.load(val_path) as data:
        validation, val_y, val_h, thresholds = [data[k] for k in ['rows', 'targets', 'horizon', 'thresholds']]
    cal, bias, strength, fit = fit_training_variant(train, targets, horizon,
        validation, val_y, val_h, task['horizon'])
    del train, targets, horizon, validation, val_y, val_h
    # Import the same native interfaces and pickle classes as the frozen campaign.
    sys.path.insert(0, str(source/'code'))
    import torch
    import run_task as legacy
    import run_fsdc36 as native
    torch.set_num_threads(2)
    with (source/'train_calibration/runs'/task['id']/'calibrators.pkl').open('rb') as stream:
        frozen_train = pickle.load(stream)['DC36']
    verify_training_template(cal, frozen_train)
    del frozen_train
    fit.update(version=VERSION, task=task, training_source=str(train_path.resolve()),
        validation_source=str(val_path.resolve()), training_template_reference_parity=True,
        reference_train_calibrator=str(source/'train_calibration/runs'/task['id']/'calibrators.pkl'))
    atomic(out/'fit.json', fit)
    cal.save(out/'compiled.npz')
    np.savez_compressed(out/'mean_parameters.npz', bias=bias,
        counts=np.asarray(fit['training_state_counts']), strength=np.array(strength))
    # All parameters are frozen before the first held-out target is loaded.
    evaluate_cohort(task, out, source, mean_root, cal, bias, strength, fit, thresholds)
    locks = root/'data_prepare_locks'; locks.mkdir(exist_ok=True)
    with (locks/(task['dataset'].lower()+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        interface = native.prepare(task, out)
    accumulator = ReliabilityMetrics(thresholds, n_states=36)
    steps = windows = 0; generation = transform_seconds = metric_seconds = maximum_mean_error = 0.
    legacy.seed(task['seed']+30000)
    for step, batch in enumerate(interface['test'], 1):
        torch.cuda.synchronize(); begin = time.monotonic()
        with torch.no_grad():
            x, y, _ = interface['predict'](batch, 'test', step)
        torch.cuda.synchronize(); generation += time.monotonic()-begin
        b, s, h, c = x.shape
        rows = x.transpose(0, 2, 3, 1).reshape(-1, s)
        horizon = np.broadcast_to(np.arange(h)[None, :, None], (b, h, c)).reshape(-1)
        begin = time.monotonic()
        forecast = centered_forecasts(cal, rows, horizon, bias, strength)
        transform_seconds += time.monotonic()-begin
        mu, _, _ = summaries(rows)
        expected_mean = mu+strength*bias[bias_ids(cal, rows, horizon)]
        maximum_mean_error = max(maximum_mean_error, float(np.max(abs(forecast.mean(1)-expected_mean))))
        groups = state_ids(x, cal.mean_edges, cal.scale_edges)
        begin = time.monotonic()
        accumulator.update(forecast.reshape(b, h, c, s).transpose(0, 3, 1, 2), y, groups, ordered=True)
        metric_seconds += time.monotonic()-begin
        windows += len(y); steps = step
        del forecast, rows
        if step % 10 == 0:
            print('TRAIN_FSPC_TEST', task['id'], step, windows, round(time.monotonic()-start, 1), flush=True)
    assert_traversal(interface['test'], steps, windows)
    assert steps == original['test_batches'] and windows == original['test_windows']
    scores = {name:original['methods'][name] for name in ['Native', 'TrainDC36', 'DC36']}
    scores['DC36ControlledMean'] = mean_original['methods']['DC36ControlledMean']
    scores['TrainFSPC'] = accumulator.result()
    np.testing.assert_array_equal(scores['TrainFSPC']['state_coverage']['counts'],
                                  scores['Native']['state_coverage']['counts'])
    assert maximum_mean_error < 1e-8
    value = dict(version=VERSION, task=task, fit=fit, methods=scores, full_test_split=True,
        test_batches=steps, test_windows=windows, test_seed=task['seed']+30000,
        checkpoint=interface['checkpoint'], baseline_metrics_recomputed=False,
        newly_evaluated_methods=['TrainFSPC'], neural_training_performed=False,
        diagnostic_partition='original training-fitted 6x6 boundaries',
        native_generation_seconds=generation, transform_seconds=transform_seconds,
        metric_seconds=metric_seconds, maximum_mean_deviation=maximum_mean_error,
        total_seconds=time.monotonic()-start,
        source_primary=str(source/'runs'/task['id']/'results.json'),
        source_validation_fspc=str(mean_root/'runs'/task['id']/'results.json'))
    validate_record(value, task)
    atomic(out/'results.json', value)
    print('TRAIN_FSPC_FULL_COMPLETE', task['id'], round(value['total_seconds'], 2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['task', 'root', 'source', 'mean-root']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); task = json.loads(args.task.read_text())
    out = args.root/'runs'/task['id']; out.mkdir(parents=True, exist_ok=True)
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out/'results.json').exists():
            validate_record(json.loads((out/'results.json').read_text()), task)
        else:
            permit = None
            while permit is None:
                permit = try_gpu_permit(args.root, os.environ.get('CUDA_VISIBLE_DEVICES','0'))
                if permit is None: time.sleep(1)
            try:
                run(task, args.root, args.source, args.mean_root)
            finally:
                permit.close()
