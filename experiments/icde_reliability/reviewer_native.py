"""Regenerate calibration and complete-test scores from a locally trained model."""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from reviewer_pipeline import atomic, fit_suite, retained_evaluation, predictions, diagnostic_groups
from reliability_metrics import ReliabilityMetrics, assert_traversal


def train_deep(task):
    import deepstuq_task as native
    args = native.configuration(task['dataset'], task['seed']); native.init_seed(task['seed'])
    os.chdir(native.ROOT/'deepstuq')
    data = native.loaders(args)
    output = native.ROOT/'deepstuq/runs'/f"{task['dataset']}_seed{task['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    model, _ = native.train(args, output, data)
    if not (output/'temperature.json').exists():
        native.init_seed(task['seed']+10000); begin = time.monotonic()
        temperature = float(native.native_training.train_cali_mc(model, 10, args, data[1], data[3]).detach().cpu().item())
        if not np.isfinite(temperature): raise ValueError('Nonfinite native temperature')
        atomic(output/'temperature.json', dict(log_temperature=temperature,
            seconds=time.monotonic()-begin, mc_samples=10, lbfgs_step=.02, lbfgs_max_iter=500,
            source='unmodified author train_cali_mc'))


def evaluate(task, work, controls=False, external=False):
    import run_task as legacy
    from run_fsdc36 import prepare
    from run_mean_improved_all import calibration
    from audit_diffusion_sampling import disable_clamp, find_native_model
    root = work/'reviewer_outputs'; out = root/'runs'/task['id']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'results.json').exists():
        raise ValueError('Completed output already exists; use another workspace to rerun')
    interface = prepare(task, out)
    if task['backbone'] == 'DiffusionTS':
        disable_clamp(find_native_model(interface))
    # Use the original sample/reservoir random streams and fitting batch cap.
    legacy.seed(task['seed']+10000)
    pool = legacy.CalibrationReservoir(50000, task['seed'])
    with torch.no_grad():
        for step, batch in enumerate(interface['train'], 1):
            if step > interface['fit_limit']: break
            x, y, _ = interface['predict'](batch, 'calibration', step); pool.update(x, y)
    tr, ty, th, tc = pool.arrays()
    train = dict(rows=tr, targets=ty, horizon=th, channel=tc)
    np.savez_compressed(out/'training.npz', **train)
    train16, _ = legacy.fit_dc(tr, ty, th, task['horizon'])
    baseline, thresholds, validation = legacy.collect_validation(task, out, interface, train16)
    vr, vy, window, thresholds, provenance = calibration(task, root, interface)
    with np.load(out/'calibration.npz') as data:
        val = {k: data[k] for k in ['rows', 'targets', 'horizon', 'channel']}
    val.update(window=window, n_windows=np.array(provenance['calibration_windows']),
        origin_stride=np.array(provenance['origin_stride']), thresholds=thresholds)
    np.savez_compressed(out/'validation.npz', **val)
    suite = fit_suite(train, val, task['horizon'], provenance['calibration_windows'],
        provenance['origin_stride'], task['seed'], controls, external)
    atomic(out/'parameters.json', suite['model'].to_record())
    atomic(out/'frozen.json', dict(task=task, fit=suite['metadata'], provenance=provenance,
        parameters_frozen_before_test=True, test_targets_used_for_fitting=False,
        control_fitting=suite['control_fitting']))
    # Fit all methods before constructing or iterating the test stream.
    names = ['Native', 'MeanOnly', 'ShapeOnly', 'FSPC']+list(suite['external'])
    internal = ['GlobalDC', 'GlobalPIT', 'ConditionalPIT']
    accumulators = {n: ReliabilityMetrics(thresholds, n_states=36) for n in names+internal}
    test_pool = legacy.CalibrationReservoir(50000, task['seed']+70000)
    legacy.seed(task['seed']+30000); batches = windows = 0
    generation_seconds = 0.; start = time.monotonic()
    for step, batch in enumerate(interface['test'], 1):
        torch.cuda.synchronize(); begin = time.monotonic()
        with torch.no_grad(): x, y, _ = interface['predict'](batch, 'test', step)
        torch.cuda.synchronize(); generation_seconds += time.monotonic()-begin
        b, s, h, c = x.shape
        rows = x.transpose(0, 2, 3, 1).reshape(-1, s)
        horizon = np.tile(np.repeat(np.arange(h), c), b)
        groups = diagnostic_groups(rows, suite['diagnostic_edges'])
        q = predictions(suite, rows, horizon)
        for name in names:
            values = q[name].reshape(b, h, c, s).transpose(0, 3, 1, 2)
            accumulators[name].update(values, y, groups, ordered=name not in ['Native', 'MeanOnly'])
        for name in internal:
            accumulators[name].update(baseline[name].transform(x), y, groups,
                ordered=name == 'GlobalDC')
        test_pool.update(x, y); batches = step; windows += b
        if step % 20 == 0: print('REVIEWER_TEST', task['id'], step, windows, flush=True)
    assert_traversal(interface['test'], batches, windows)
    if windows != len(interface['test'].dataset): raise ValueError('Incomplete test population')
    scores = {n: a.result() for n, a in accumulators.items()}
    rows, y, h, c = test_pool.arrays(); test = dict(rows=rows, targets=y, horizon=h, channel=c)
    np.savez_compressed(out/'test_reservoir.npz', **test)
    retained_evaluation(suite, test, thresholds, out, task['seed'])
    atomic(out/'results.json', dict(version='fspc-reviewer-from-data-v1', task=task,
        full_test_split=True, test_batches=batches, test_windows=windows, methods=scores,
        native_generation_seconds=generation_seconds, total_test_seconds=time.monotonic()-start,
        diagnostic_partition='training-fitted 6x6 mean/log-scale; shared by all methods',
        checkpoint=interface['checkpoint'], author_score_overrides_applied=False))
    print('REVIEWER_COMPLETE', task['id'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True); p.add_argument('--task', required=True)
    p.add_argument('--stage', choices=['train', 'evaluate'], required=True)
    p.add_argument('--controls', action='store_true'); p.add_argument('--external', action='store_true')
    a = p.parse_args(); work = a.workspace.resolve()
    manifest = work/'payload/home/wyj/icde_reliability_20261102/manifest.json'
    task = next(t for t in json.loads(manifest.read_text()) if t['id'] == a.task)
    torch.set_num_threads(2)
    if a.stage == 'train':
        if task['backbone'] != 'DeepSTUQ': raise ValueError('Use reproduce.py train for other backbones')
        train_deep(task)
    else:
        evaluate(task, work, a.controls, a.external)


if __name__ == '__main__':
    main()
