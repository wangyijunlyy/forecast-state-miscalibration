"""Run DeepSTUQ through every frozen-backbone calibration experiment."""
import argparse
import json
import os
import pickle
import resource
import time
from pathlib import Path
import numpy as np
import torch
import deepstuq_task as native
import run_task as primary
from gaussian_models import fit_gaussian, predict as gaussian_predict
from reliability_metrics import ReliabilityMetrics, state_ids, assert_traversal


def train_calibration(task, output, interface):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'fit_metadata.json').exists():
        with (output/'calibrators.pkl').open('rb') as stream:
            return pickle.load(stream)['DC16']
    primary.seed(task['seed']+10000)
    pool = primary.CalibrationReservoir(50000, task['seed'])
    windows = steps = 0
    for step, batch in enumerate(interface['train'], 1):
        if step > 100:
            break
        x, y, _ = interface['predict'](batch, 'calibration', step)
        pool.update(x, y); windows += len(y); steps = step
        if step % 20 == 0:
            print('TRAIN_FIT', task['id'], step, flush=True)
    rows, target, horizon, channel = pool.arrays()
    models = {}; metadata = {}
    for name, groups in [('DC16', 1), ('DC64', 4)]:
        models[name], metadata[name] = primary.fit_dc(rows, target, horizon, 12, n_horizon=groups)
    with (output/'calibrators.pkl').open('wb') as stream:
        pickle.dump(models, stream)
    np.savez_compressed(output/'calibration.npz', rows=rows, targets=target, horizon=horizon, channel=channel)
    primary.atomic(output/'fit_metadata.json', dict(fit_split='train', fit_batches=steps,
        fit_windows=windows, retained_points=len(target), seen_points=pool.seen_points,
        prediction_seed=task['seed']+10000, models=metadata, source_scale_threshold=1.01e-6,
        boundary_source='all retained fitting coordinates', source_filter_scope='residual sources only'))
    return models['DC16']


def gaussian_fitting(task, directory, source):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'fitting.json').exists():
        return json.loads((directory/'fitting.json').read_text())
    with np.load(source) as data:
        rows, target = data['rows'], data['targets']
    mean = rows.mean(1); scale = rows.std(1, ddof=1).clip(1e-6)
    models = {}; timings = {}
    for name, affine in [('GaussianVariance2P', False), ('GaussianAffine4P', True)]:
        start = time.monotonic(); models[name] = fit_gaussian(mean, scale, target, affine)
        timings[name] = time.monotonic()-start
        if not models[name]['optimizer_success']:
            raise ValueError('Unsuccessful native Gaussian control: '+name)
    record = dict(version='full-gaussian-controls-v1-20261102', task=task,
        fitting_split='validation', calibration_coordinates=len(target), coordinate_source=str(source),
        models=models, fit_seconds=timings)
    primary.atomic(directory/'fitting.json', record)
    return record


def run(dataset, seed, root, native_runs=None):
    task = dict(id=f'DeepSTUQ_{dataset}_h12_seed{seed}', backbone='DeepSTUQ', dataset=dataset, horizon=12, seed=seed)
    out = root/'runs'/task['id']; out.mkdir(parents=True, exist_ok=True)
    primary.atomic(out/'task.json', task)
    complete = (out/'results.json').exists() and (root/'gaussian_runs'/task['id']/'results.json').exists()
    if not complete:
        original_dir = (native_runs or native.ROOT/'deepstuq/runs')/f'{dataset}_seed{seed}'
        for name in ['training.json', 'model.pt', 'temperature.json', 'results.json']:
            if not (original_dir/name).is_file():
                raise ValueError('Missing full native training prerequisite: '+str(original_dir/name))
        original = json.loads((original_dir/'results.json').read_text())
        args = native.configuration(dataset, seed); native.init_seed(seed)
        os.chdir(native.ROOT/'deepstuq')
        data = native.loaders(args)
        model, training = native.train(args, original_dir, data)
        temperature = json.loads((original_dir/'temperature.json').read_text())['log_temperature']
        train_loader, val_loader, test_loader, scaler = data

        def predict(batch, split, step):
            x, raw, mean, variance = native.predict(model, batch, args, temperature)
            y = scaler.inverse_transform(batch[1][..., :args.output_dim]).squeeze(-1).cpu().numpy()
            return x, y, dict(raw=raw, mean=mean, variance=variance)

        interface = dict(train=train_loader, val=val_loader, test=test_loader, predict=predict,
            fit_limit=100, checkpoint=str(original_dir/'model.pt'))
        partition = train_calibration(task, root/'train_calibration/runs'/task['id'], interface)
        if (out/'validation.json').exists():
            validation = json.loads((out/'validation.json').read_text())
            with (out/'calibrators.pkl').open('rb') as stream:
                models = pickle.load(stream)
            with np.load(out/'calibration.npz') as fitting:
                thresholds = fitting['thresholds']
        else:
            models, thresholds, validation = primary.collect_validation(task, out, interface, partition)
        models['TrainDC16'] = partition
        with (root/'train_calibration/runs'/task['id']/'calibrators.pkl').open('rb') as stream:
            models['TrainDC64'] = pickle.load(stream)['DC64']
        gaussian_dir = root/'gaussian_runs'/task['id']
        fitting = gaussian_fitting(task, gaussian_dir, out/'calibration.npz')
        names = primary.METHODS+list(fitting['models'])+['TrainDC64']
        accumulators = {name:ReliabilityMetrics(thresholds) for name in names}
        components = ReliabilityMetrics(thresholds)
        timings = {name:0. for name in names}; metric_seconds = {name:0. for name in names}
        test_pool = primary.CalibrationReservoir(50000, seed+70000)
        primary.seed(seed+30000); windows = steps = 0; start = time.monotonic()
        for step, batch in enumerate(test_loader, 1):
            torch.cuda.synchronize(); begin = time.monotonic()
            x, y, additional = predict(batch, 'test', step)
            torch.cuda.synchronize(); timings['Native'] += time.monotonic()-begin
            groups = state_ids(x, partition.mean_edges, partition.scale_edges)
            test_pool.update(x, y)
            for name in names:
                begin = time.monotonic()
                if name == 'Native':
                    values = x
                elif name in models:
                    values = models[name].transform(x)
                else:
                    b, s, h, c = x.shape
                    rows = x.transpose(0, 2, 3, 1).reshape(-1, s)
                    values = gaussian_predict(fitting['models'][name], rows).reshape(b, h, c, s).transpose(0, 3, 1, 2)
                if name != 'Native':
                    timings[name] += time.monotonic()-begin
                begin = time.monotonic()
                accumulators[name].update(values, y, groups, ordered=True)
                metric_seconds[name] += time.monotonic()-begin
                del values
            components.update(additional['raw'], y, groups, ordered=True)
            windows += len(y); steps = step
            if step % 20 == 0:
                print('BACKBONE_TEST', task['id'], step, windows, round(time.monotonic()-start, 1), flush=True)
        assert_traversal(test_loader, steps, windows)
        if windows != original['test_windows'] or steps != original['test_batches']:
            raise ValueError('DeepSTUQ full-test population mismatch')
        scores = {name:acc.result() for name,acc in accumulators.items()}
        deltas = {key:scores['Native'][key]-original['methods']['DeepSTUQ'][key]
                  for key in ['CRPS', 'MSE', 'QICE_percent']}
        if max(abs(v) for v in deltas.values()) > 1e-9:
            primary.atomic(out/'native_parity_failure.json', deltas)
            raise ValueError('DeepSTUQ native replay differs from original full trained baseline')
        rows, target, h, c = test_pool.arrays()
        np.savez_compressed(out/'test_reservoir.npz', rows=rows, targets=target, horizon=h, channel=c)
        primary.controls(task, out, test_pool, partition, thresholds)
        record = dict(version=primary.VERSION, task=task, checkpoint=interface['checkpoint'], full_test_split=True,
            test_windows=windows, test_batches=steps, test_seed=seed+30000,
            methods={name:scores[name] for name in primary.METHODS}, validation=validation,
            native_score_deltas=deltas, generation_or_transform_seconds={name:timings[name] for name in primary.METHODS},
            metric_seconds=metric_seconds, native_generation_excludes_preparation=True,
            total_test_seconds=time.monotonic()-start,
            maximum_resident_memory_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            diagnostic_partition='training-fitted DC16 mean/scale boundaries shared by all methods',
            native_training=training, native_temperature=temperature, native_dropout_forward_passes=10,
            output_members=100, marginal_representation='100 midpoint quantiles of native DeepSTUQ predictive Gaussian')
        gaussian_record = dict(version='full-gaussian-controls-v1-20261102', task=task, full_test_split=True,
            checkpoint=interface['checkpoint'], test_windows=windows, test_batches=steps, test_seed=seed+30000,
            methods={name:scores[name] for name in ['Native', 'GaussianVariance2P', 'GaussianAffine4P']},
            fitting=fitting, native_score_deltas={key:0. for key in deltas},
            diagnostic_partition=record['diagnostic_partition'], same_primary_prediction_pass=True,
            generation_or_transform_seconds={name:timings[name] for name in fitting['models']})
        primary.atomic(gaussian_dir/'results.json', gaussian_record)
        primary.atomic(out/'training_partition_comparison.json', dict(task=task, full_test_split=True,
            test_windows=windows, test_batches=steps, test_seed=seed+30000,
            methods={name:scores[name] for name in ['Native', 'TrainDC16', 'TrainDC64']}))
        primary.atomic(out/'native_components.json', dict(method='NoTemperature', scores=components.result(),
            native_gaussian=original['native_gaussian'], original_reference=str(original_dir/'results.json')))
        primary.atomic(out/'results.json', record)
    if not (out/'training_partition_comparison.json').exists():
        # Earlier integration runs predate the training-DC64 comparison. Replay
        # the full native iterator with the same seed, without repeating fits.
        args = native.configuration(dataset, seed); native.init_seed(seed)
        os.chdir(native.ROOT/'deepstuq')
        data = native.loaders(args)
        original_dir = (native_runs or native.ROOT/'deepstuq/runs')/f'{dataset}_seed{seed}'
        model, _ = native.train(args, original_dir, data)
        temperature = json.loads((original_dir/'temperature.json').read_text())['log_temperature']
        with (root/'train_calibration/runs'/task['id']/'calibrators.pkl').open('rb') as stream:
            calibrators = pickle.load(stream)
        with np.load(out/'calibration.npz') as fitting:
            thresholds = fitting['thresholds']
        metrics = {name:ReliabilityMetrics(thresholds) for name in ['Native', 'TrainDC16', 'TrainDC64']}
        primary.seed(seed+30000); windows = steps = 0
        for step, batch in enumerate(data[2], 1):
            x, _, _, _ = native.predict(model, batch, args, temperature)
            y = data[3].inverse_transform(batch[1][..., :args.output_dim]).squeeze(-1).cpu().numpy()
            groups = state_ids(x, calibrators['DC16'].mean_edges, calibrators['DC16'].scale_edges)
            for name in metrics:
                values = x if name == 'Native' else calibrators[name.replace('Train', '')].transform(x)
                metrics[name].update(values, y, groups, ordered=True)
            windows += len(y); steps = step
        reference = json.loads((out/'results.json').read_text())
        scores = {name:acc.result() for name,acc in metrics.items()}
        for name in ['Native', 'TrainDC16']:
            for key in ['CRPS', 'MSE', 'QICE_percent']:
                np.testing.assert_allclose(scores[name][key], reference['methods'][name][key], atol=1e-10, rtol=1e-10)
        assert windows == reference['test_windows'] and steps == reference['test_batches']
        primary.atomic(out/'training_partition_comparison.json', dict(task=task, full_test_split=True,
            test_windows=windows, test_batches=steps, test_seed=seed+30000, methods=scores))
    import benchmark_compiled
    import run_state_controls
    benchmark_compiled.run(task, root)
    run_state_controls.ROOT = root
    run_state_controls.TRAIN = root/'train_calibration/runs'
    run_state_controls.run(task)
    print('DEEPSTUQ_BACKBONE_COMPLETE', task['id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'])
    parser.add_argument('--seed', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--native-runs', type=Path, help='Native full-training outputs, including locally retrained checkpoints')
    args = parser.parse_args(); torch.set_num_threads(2)
    run(args.dataset, args.seed, args.output_root or native.ROOT/'deepstuq_backbone', args.native_runs)
