"""Independent audit and seed-aware summaries of the fixed ICDE experiment.

Partial downloads are allowed, but only complete three-seed configurations enter
summaries. Backbone summaries average configurations within each training seed
before computing a sample standard deviation across those three seeds.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

VERSION = 'icde-reliability-v1-20261102'
METHODS = {'Native', 'TrainDC16', 'DC16', 'GlobalDC', 'GlobalPIT', 'ConditionalPIT'}
IDENTITY = ('backbone', 'dataset', 'horizon', 'seed', 'method')
METRICS = ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE', 'MedianMAE', 'state_error_pp',
           'pooled_error_pp', 'Brier80', 'Brier90', 'Brier95', 'BrierMean', 'transform_seconds']


def check_unique(rows):
    seen = set()
    for row in rows:
        identity = tuple(row[k] for k in IDENTITY)
        if identity in seen:
            raise ValueError(f'Duplicate task/method identity: {identity}')
        seen.add(identity)


def statistics(values, metrics):
    result = {}
    for key in metrics:
        v = [row.get(key) for row in values]
        valid = all(x is not None and math.isfinite(x) for x in v)
        result[key+'_mean'] = float(np.mean(v)) if valid else None
        result[key+'_std'] = float(np.std(v, ddof=1)) if valid else None
    return result


def complete_configurations(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['backbone'], row['dataset'], row['horizon'], row['method'])].append(row)
    return {key: values for key, values in groups.items()
            if {r['seed'] for r in values} == {1, 2, 3} and len(values) == 3}


def configuration_statistics(rows, metrics=METRICS):
    check_unique(rows)
    return [dict(zip(('backbone', 'dataset', 'horizon', 'method'), key),
                 n_training_seeds=3, **statistics(values, metrics))
            for key, values in sorted(complete_configurations(rows).items())]


def backbone_statistics(rows, metrics=METRICS):
    check_unique(rows)
    groups = defaultdict(list)
    for (backbone, dataset, horizon, method), values in complete_configurations(rows).items():
        groups[(backbone, method)].append(values)
    output = []
    for (backbone, method), configurations in sorted(groups.items()):
        summary = dict(backbone=backbone, method=method, n_configurations=len(configurations), n_training_seeds=3)
        for key in metrics:
            valid = [values for values in configurations
                     if all(r.get(key) is not None and math.isfinite(r[key]) for r in values)]
            summary[key+'_n_configurations'] = len(valid)
            seeds = [dict(**{key: float(np.mean([next(r[key] for r in values if r['seed'] == seed)
                                                  for values in valid]))}) for seed in (1, 2, 3)] if valid else []
            summary.update(statistics(seeds, [key]) if seeds else {key+'_mean':None, key+'_std':None})
        output.append(summary)
    return output


def close(actual, expected, message, atol=1e-10):
    if not np.allclose(actual, expected, atol=atol, rtol=1e-10):
        raise ValueError(message)


def audit_metric(metric):
    """Reconstruct coverage and event totals without importing experiment code."""
    n = metric['coordinate_count']
    if not isinstance(n, int) or n <= 0:
        raise ValueError('Invalid coordinate count')
    for key in ['CRPS', 'MSE', 'MAE', 'QICE_percent']:
        if not math.isfinite(metric[key]) or metric[key] < -1e-12:
            raise ValueError(f'Invalid {key}')
    close(metric['QICE_percent'], 100*metric['QICE'], 'QICE unit mismatch')
    close(metric['RMSE']**2, metric['MSE'], 'RMSE mismatch')
    state = metric['state_coverage']
    counts = np.asarray(state['counts']); hits = np.asarray(state['covered'])
    levels = np.asarray(state['levels']); eligible = counts >= 128
    if counts.shape != (16,) or hits.shape != (16, 3) or counts.sum() != n:
        raise ValueError('State population mismatch')
    if np.any(counts < 0) or np.any(hits < 0) or np.any(hits > counts[:, None]):
        raise ValueError('Invalid state coverage counts')
    if np.any(np.diff(hits, axis=1) < 0):
        raise ValueError('Quantile coverage is not monotone')
    close(state['coverage'], hits/np.maximum(counts[:, None], 1), 'State coverage mismatch')
    nn = int(counts[eligible].sum())
    if nn != state['eligible_coordinates'] or int(eligible.sum()) != state['eligible_states']:
        raise ValueError('Eligible population mismatch')
    if nn:
        pooled = hits[eligible].sum(0)/nn
        error = np.abs(hits[eligible]-counts[eligible, None]*levels).sum()/(3*nn)
        close(state['pooled_coverage'], pooled, 'Pooled coverage mismatch')
        close(state['pooled_error'], np.abs(pooled-levels).mean(), 'Pooled error mismatch')
        close(state['state_error'], error, 'State error mismatch')
    elif state['state_error'] is not None or state['pooled_error'] is not None:
        raise ValueError('Undefined coverage error should be null')
    if 'events' not in metric:
        return
    event = metric['events']
    bins = np.asarray(event['probability_bin_counts'])
    probabilities = np.asarray(event['probability_sums'])
    observed = np.asarray(event['observation_sums'])
    brier = np.asarray(event['brier'])
    if bins.shape != (3, 10) or np.any(bins.sum(1) != n):
        raise ValueError('Incomplete probability-bin populations')
    if np.any(observed < 0) or np.any(observed > bins) or np.any(brier < 0) or np.any(brier > 1):
        raise ValueError('Invalid event counts or scores')
    close(observed.sum(1)/n, event['observed_frequencies'], 'Event frequency mismatch')
    close(np.asarray(event['state_brier_sums']).sum(0)/n, brier, 'State/event Brier mismatch')
    low = np.arange(10)/10; high = (np.arange(10)+1)/10
    if np.any(probabilities < bins*low-1e-7) or np.any(probabilities > bins*high+1e-7):
        raise ValueError('Probability sums outside bin bounds')
    # Jensen lower bound independently links bin summaries with the Brier total.
    bound = ((probabilities-observed)**2/np.maximum(bins, 1)).sum(1)/n
    if np.any(brier < bound-1e-10):
        raise ValueError('Brier score violates its bin-summary lower bound')


def audit_run(result, control, validation):
    if any(d.get('version') != VERSION for d in (result, control, validation)):
        raise ValueError('Experiment version mismatch')
    if not result['full_test_split'] or not validation['validation_complete'] or not validation['full_validation_split']:
        raise ValueError('Incomplete iterator')
    if result['test_windows'] <= 0 or result['test_batches'] <= 0 or set(result['methods']) != METHODS:
        raise ValueError('Incomplete method/iterator identity')
    for metric in result['methods'].values():
        audit_metric(metric)
    native = result['methods']['Native']
    for metric in result['methods'].values():
        if metric['coordinate_count'] != native['coordinate_count']:
            raise ValueError('Methods scored different coordinate populations')
        if metric['state_coverage']['counts'] != native['state_coverage']['counts']:
            raise ValueError('Methods used different diagnostic partitions')
        close(metric['events']['observed_frequencies'], native['events']['observed_frequencies'], 'Event definitions differ')
    if control['full_test_coordinates_seen'] != native['coordinate_count']:
        raise ValueError('Test reservoir did not traverse the whole test population')
    if set(control['methods']) != {'Budget2000', 'Budget10000', 'Budget50000', 'DC64', 'UnfilteredDC16'}:
        raise ValueError('Incomplete calibration-budget/mechanism control set')
    for metric in control['methods'].values():
        audit_metric(metric)
        if metric['coordinate_count'] != control['retained_test_coordinates']:
            raise ValueError('Control evaluation population mismatch')
    for metric in validation['diagnostics'].values():
        audit_metric(metric)
    if max(abs(x) for x in result['native_score_deltas'].values()) > 1e-9:
        raise ValueError('Native score differs from original paired test run')


def flatten(result):
    rows = []
    task = result['task']
    for method, metric in result['methods'].items():
        state = metric['state_coverage']; events = metric['events']
        row = {k: task[k] for k in ['id', 'backbone', 'dataset', 'horizon', 'seed']}
        row.update(method=method, coordinate_count=metric['coordinate_count'],
                   eligible_coordinates=state['eligible_coordinates'],
                   state_error_pp=None if state['state_error'] is None else 100*state['state_error'],
                   pooled_error_pp=None if state['pooled_error'] is None else 100*state['pooled_error'],
                   transform_seconds=result['generation_or_transform_seconds'][method])
        row.update({k: metric[k] for k in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE', 'MedianMAE']})
        row.update(zip(['Brier80', 'Brier90', 'Brier95'], events['brier']))
        row['BrierMean'] = float(np.mean(events['brier']))
        rows.append(row)
    return rows


def control_rows(result, control):
    """Controls are a separately labelled shared reservoir, never full-test rows."""
    task = result['task']; rows = []
    for method, metric in control['methods'].items():
        row = {k:task[k] for k in ['id', 'backbone', 'dataset', 'horizon', 'seed']}
        row.update(method=method, evaluation=control['evaluation'], coordinate_count=metric['coordinate_count'])
        row.update({k:metric[k] for k in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE', 'MedianMAE']})
        row.update(control['resources'][method]); rows.append(row)
    return rows


def resource_rows(result, control):
    task = result['task']; n = result['methods']['Native']['coordinate_count']
    generation = result['generation_or_transform_seconds']['Native']
    row = {k:task[k] for k in ['id', 'backbone', 'dataset', 'horizon', 'seed']}
    fit = result['validation']['models']['DC16']
    row.update(method='DC16', coordinate_count=n,
        native_generation_seconds=generation, native_generation_us_per_coordinate=1e6*generation/n,
        dc16_transform_us_per_coordinate=1e6*result['generation_or_transform_seconds']['DC16']/n,
        dc16_transform_to_generation_ratio=result['generation_or_transform_seconds']['DC16']/generation,
        dc16_fit_seconds=result['validation']['models']['DC16']['fit_seconds'],
        dc16_calibrator_bytes=control['resources']['Budget50000']['calibrator_bytes'],
        calibration_coordinates=fit['calibration_coordinates'], retained_sources=fit['retained_sources'],
        excluded_sources=fit['excluded_sources'], source_excluded_percent=100*fit['excluded_sources']/fit['calibration_coordinates'],
        local_residual_pools=fit['local_pools'],
        maximum_resident_memory_kib=result['maximum_resident_memory_kib'],
        maximum_cuda_allocated_bytes=result['maximum_cuda_allocated_bytes'])
    return row


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = {t['id']: t for t in json.loads(args.manifest.read_text())}
    rows = []; controls = []; resources = []; completed = []; errors = []
    for path in sorted(args.runs.glob('*/results.json')):
        try:
            result = json.loads(path.read_text()); task = result['task']; identity = task['id']
            if identity not in expected or identity != path.parent.name:
                raise ValueError('Unexpected task identity')
            for key in ['backbone', 'dataset', 'horizon', 'seed']:
                if task[key] != expected[identity][key]:
                    raise ValueError('Manifest identity mismatch')
            control = json.loads((path.parent/'controls.json').read_text())
            audit_run(result, control, json.loads((path.parent/'validation.json').read_text()))
            rows.extend(flatten(result)); controls.extend(control_rows(result, control))
            resources.append(resource_rows(result, control)); completed.append(identity)
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
    check_unique(rows)
    configurations = configuration_statistics(rows)
    backbones = backbone_statistics(rows)
    missing = sorted(set(expected)-set(completed))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'raw_task_metrics.csv', rows)
    write_csv(args.output/'complete_configuration_statistics.csv', configurations)
    write_csv(args.output/'partial_backbone_statistics.csv', backbones)
    write_csv(args.output/'raw_reservoir_controls.csv', controls)
    write_csv(args.output/'partial_reservoir_control_statistics.csv', backbone_statistics(controls, ['CRPS', 'QICE_percent']))
    write_csv(args.output/'raw_resource_metrics.csv', resources)
    resource_keys = ['native_generation_us_per_coordinate', 'dc16_transform_us_per_coordinate',
                     'dc16_transform_to_generation_ratio', 'dc16_fit_seconds', 'dc16_calibrator_bytes',
                     'source_excluded_percent', 'local_residual_pools']
    write_csv(args.output/'partial_resource_statistics.csv', backbone_statistics(resources, resource_keys))
    report = dict(expected_tasks=len(expected), audited_tasks=len(completed),
                  all_tasks_complete=not missing and not errors, errors=errors,
                  missing_tasks=missing, complete_configuration_method_rows=len(configurations),
                  manuscript_standard_deviations='Measured sample SD rounded to three decimals; raw CSV retains full precision',
                  aggregation='equal configurations within each seed, then mean/sample SD over seeds 1,2,3',
                  partial_summary_warning='Backbone rows contain only complete three-seed configurations; use final tables only after all manifest tasks pass.')
    (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['missing_tasks', 'errors']}, indent=2))
    if errors:
        raise SystemExit(json.dumps(errors, indent=2))


if __name__ == '__main__':
    main()
