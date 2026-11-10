"""Complete-native-iterator campaign for the 6x6 FSDC default."""
import argparse
import fcntl
import json
import os
import pickle
import resource
import time
from pathlib import Path
import numpy as np
import torch
import run_task as legacy
from reliability_metrics import ReliabilityMetrics, state_ids, assert_traversal
from portable_dc import EmpiricalCalibrator
from gaussian_models import predict as gaussian_predict

VERSION = 'icde-fsdc36-v1-20261103'
METHODS = ['Native', 'TrainDC36', 'DC36', 'GlobalDC', 'GlobalPIT', 'ConditionalPIT']


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def prepare(task, output):
    if task['backbone'] != 'DeepSTUQ':
        return legacy.prepare(task, output)
    import deepstuq_task as native
    args = native.configuration(task['dataset'], task['seed']); native.init_seed(task['seed'])
    os.chdir(native.ROOT/'deepstuq')
    data = native.loaders(args)
    original = native.ROOT/'deepstuq/runs'/f'{task["dataset"]}_seed{task["seed"]}'
    if not (original/'training.json').exists():
        raise FileNotFoundError(original)
    model, _ = native.train(args, original, data)
    temperature = json.loads((original/'temperature.json').read_text())['log_temperature']
    def predict(batch, phase, step):
        x, _, _, _ = native.predict(model, batch, args, temperature)
        y = data[3].inverse_transform(batch[1][..., :args.output_dim]).squeeze(-1).cpu().numpy()
        return x, y, None
    return dict(train=data[0], val=data[1], test=data[2], predict=predict,
        fit_limit=100, checkpoint=str(original/'model.pt'))


def fit36(rows, y, h, horizon, horizon_groups=1, source_filter=True):
    return legacy.fit_dc(rows, y, h, horizon, n_mean=6, n_scale=6,
                         n_horizon=horizon_groups, source_filter=source_filter)


def training_fit(task, root, interface):
    out = root/'train_calibration/runs'/task['id']; out.mkdir(parents=True, exist_ok=True)
    path = out/'calibration.npz'
    if not path.exists():
        previous = Path(task['source36_train'])/'calibration.npz'
        if previous.exists():
            path.symlink_to(previous)
        else:
            legacy.seed(task['seed']+10000)
            pool = legacy.CalibrationReservoir(50000, task['seed'])
            windows = steps = 0
            with torch.no_grad():
                for step, batch in enumerate(interface['train'], 1):
                    if step > interface['fit_limit']:
                        break
                    x, y, _ = interface['predict'](batch, 'calibration', step)
                    pool.update(x, y); windows += len(y); steps = step
                    if step % 20 == 0:
                        print('TRAIN_FIT36', task['id'], step, flush=True)
            rows, y, h, c = pool.arrays()
            np.savez_compressed(path, rows=rows, targets=y, horizon=h, channel=c)
            atomic(out/'collection.json', dict(fit_batches=steps, fit_windows=windows,
                seen_points=pool.seen_points, retained_points=len(y), seed=task['seed']+10000))
    with np.load(path) as d:
        rows, y, h = [d[k] for k in ['rows', 'targets', 'horizon']]
    # Verify regenerated training samples against the frozen 4x4 calibrator.
    old, _ = legacy.fit_dc(rows, y, h, task['horizon'])
    with (Path(task['source36_train'])/'calibrators.pkl').open('rb') as stream:
        original = pickle.load(stream)['DC16']
    for left, right in [(old.mean_edges, original.mean_edges), (old.scale_edges, original.scale_edges),
                        (old.global_residuals, original.global_residuals)]:
        np.testing.assert_array_equal(left, right)
    if old.bin_residuals.keys() != original.bin_residuals.keys():
        raise ValueError('Training reservoir parity failed')
    for key in old.bin_residuals:
        np.testing.assert_array_equal(old.bin_residuals[key], original.bin_residuals[key])
    models = {}; metadata = {}
    for name, g in [('DC36', 1), ('DC144', 4)]:
        begin = time.monotonic()
        models[name], metadata[name] = fit36(rows, y, h, task['horizon'], g)
        metadata[name]['fit_seconds'] = time.monotonic()-begin
    with (out/'calibrators.pkl').open('wb') as stream:
        pickle.dump(models, stream)
    atomic(out/'fit_metadata.json', dict(version=VERSION, task=task, fit_split='train',
        mean_bins=6, scale_bins=6, regenerated_training_reservoir_matches_original=True,
        models=metadata, prediction_seed=task['seed']+10000))
    return models


def run(task, root):
    output = root/'runs'/task['id']; output.mkdir(parents=True, exist_ok=True)
    if (output/'results.json').exists():
        result = json.loads((output/'results.json').read_text())
        if result['version'] != VERSION:
            raise ValueError('Incompatible existing campaign')
        return
    atomic(output/'task.json', task)
    source = Path(task['source36_root'])/'runs'/task['id']
    historical = json.loads((source/'results.json').read_text())
    interface = prepare(task, output)
    training = training_fit(task, root, interface); partition = training['DC36']
    for name in ['calibration.npz', 'test_reservoir.npz']:
        if not (output/name).exists():
            (output/name).symlink_to(source/name)
    with np.load(source/'calibration.npz') as d:
        rows, y, h, thresholds = [d[k] for k in ['rows', 'targets', 'horizon', 'thresholds']]
    with (source/'calibrators.pkl').open('rb') as stream:
        old_models = pickle.load(stream)
    begin = time.monotonic(); dc, meta = fit36(rows, y, h, task['horizon'])
    meta.update(fit_seconds=time.monotonic()-begin, mean_bins=6, scale_bins=6, horizon_bins=1)
    models = {name:old_models[name] for name in ['GlobalDC', 'GlobalPIT', 'ConditionalPIT']}
    models.update(DC36=dc, TrainDC36=partition, TrainDC144=training['DC144'])
    with (output/'calibrators.pkl').open('wb') as stream:
        pickle.dump(models, stream)
    gaussian_root = Path(task['source36_root'])/'gaussian_runs'/task['id']
    # The current manuscript retains the five empirical/PIT comparisons.
    gaussian_fitting = dict(models={})
    gaussian_names = list(gaussian_fitting['models'])
    validation = json.loads((source/'validation.json').read_text())
    validation.update(version=VERSION, mean_bins=6, scale_bins=6,
        models={**{n:validation['models'][n] for n in ['GlobalDC', 'GlobalPIT', 'ConditionalPIT']}, 'DC36':meta},
        calibrator_pickle_bytes=(output/'calibrators.pkl').stat().st_size)
    if (output/'validation.json').exists():
        validation = json.loads((output/'validation.json').read_text())
    else:
        legacy.seed(task['seed']+20000)
        diagnostics = {n:ReliabilityMetrics(np.zeros_like(thresholds), n_states=36) for n in ['Native', 'TrainDC36']}
        windows = steps = 0
        for step, batch in enumerate(interface['val'], 1):
            with torch.no_grad():
                x, y, _ = interface['predict'](batch, 'validation', step)
            groups = state_ids(x, partition.mean_edges, partition.scale_edges)
            diagnostics['Native'].update(x, y, groups)
            diagnostics['TrainDC36'].update(partition.transform(x), y, groups, ordered=True)
            windows += len(y); steps = step
            if step % 20 == 0:
                print('VALIDATION36', task['id'], step, flush=True)
        assert_traversal(interface['val'], steps, windows)
        assert windows == validation['validation_windows'] and steps == validation['validation_batches']
        validation['diagnostics'] = {n:acc.result() for n, acc in diagnostics.items()}
        for metric in validation['diagnostics'].values():
            del metric['events']
        old_native = json.loads((source/'validation.json').read_text())['diagnostics']['Native']
        for key in ['CRPS', 'QICE_percent', 'MSE']:
            np.testing.assert_allclose(validation['diagnostics']['Native'][key], old_native[key], atol=1e-9, rtol=1e-10)
        atomic(output/'validation.json', validation)
    names = METHODS+gaussian_names+['TrainDC144']
    accumulators = {n:ReliabilityMetrics(thresholds, n_states=36) for n in names}
    timings = {n:0. for n in names}; metric_seconds = {n:0. for n in names}
    legacy.seed(task['seed']+30000); windows = steps = 0; start = time.monotonic()
    for step, batch in enumerate(interface['test'], 1):
        torch.cuda.synchronize(); begin = time.monotonic()
        with torch.no_grad():
            x, y, _ = interface['predict'](batch, 'test', step)
        torch.cuda.synchronize(); timings['Native'] += time.monotonic()-begin
        groups = state_ids(x, partition.mean_edges, partition.scale_edges)
        for name in names:
            begin = time.monotonic()
            if name == 'Native':
                values = x
            elif name in models:
                values = models[name].transform(x)
            else:
                b, s, h, c = x.shape
                rows = x.transpose(0, 2, 3, 1).reshape(-1, s)
                values = gaussian_predict(gaussian_fitting['models'][name], rows).reshape(b, h, c, s).transpose(0, 3, 1, 2)
            if name != 'Native':
                timings[name] += time.monotonic()-begin
            begin = time.monotonic()
            accumulators[name].update(values, y, groups, ordered=name not in ['Native', 'GlobalPIT', 'ConditionalPIT'])
            metric_seconds[name] += time.monotonic()-begin
            del values
        windows += len(y); steps = step
        if step % 20 == 0:
            print('TEST36', task['id'], step, windows, round(time.monotonic()-start, 1), flush=True)
    assert_traversal(interface['test'], steps, windows)
    if windows != historical['test_windows'] or steps != historical['test_batches']:
        raise ValueError('Full-test traversal mismatch')
    scores = {n:a.result() for n, a in accumulators.items()}
    unchanged = ['Native', 'GlobalDC', 'GlobalPIT', 'ConditionalPIT']
    for name in unchanged:
        for key in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE']:
            np.testing.assert_allclose(scores[name][key], historical['methods'][name][key], atol=1e-9, rtol=1e-10)
    old_gaussian = json.loads((gaussian_root/'results.json').read_text()) if gaussian_names else None
    for name in gaussian_names:
        for key in ['CRPS', 'QICE_percent', 'MSE']:
            np.testing.assert_allclose(scores[name][key], old_gaussian['methods'][name][key], atol=1e-9, rtol=1e-10)
    record = dict(version=VERSION, task=task, mean_bins=6, scale_bins=6, horizon_bins=1,
        checkpoint=interface['checkpoint'], full_test_split=True, test_windows=windows,
        test_batches=steps, test_seed=task['seed']+30000,
        methods={n:scores[n] for n in METHODS}, gaussian_methods={n:scores[n] for n in gaussian_names},
        training_horizon_methods={n:scores[n] for n in ['Native', 'TrainDC36', 'TrainDC144']},
        validation=validation, unchanged_method_parity=True, default_calibrator_bytes=len(pickle.dumps(dc)),
        gaussian_fitting=gaussian_fitting,
        generation_or_transform_seconds=timings, metric_seconds=metric_seconds,
        native_generation_excludes_preparation=True, total_test_seconds=time.monotonic()-start,
        maximum_resident_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        diagnostic_partition='training-fitted 6x6 boundaries shared by all methods',
        source_primary=str(source), default_selection='36 states adopted after descriptive granularity study')
    atomic(output/'results.json', record)
    print('FSDC36_FULL_TEST_COMPLETE', task['id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True); args = parser.parse_args()
    torch.set_num_threads(2)
    task = json.loads(args.task.read_text())
    directory = args.root/'runs'/task['id']; directory.mkdir(parents=True,exist_ok=True)
    with (directory/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run(task,args.root)
