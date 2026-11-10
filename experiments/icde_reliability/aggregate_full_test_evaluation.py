"""Audit and aggregate complete-test factorial, granularity and temporal arms."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np
from aggregate_fsdc36 import metric_row, verify_pair
from aggregation import backbone_statistics, configuration_statistics, write_csv
from full_test_evaluation import VERSION, DEFAULT, load, atomic
from fspc_granularity import VARIANTS
from granularity import method_name

METRICS = ['MSE', 'MAE', 'RMSE', 'MedianMAE', 'CRPS', 'QICE_percent',
    'state_error_pp', 'pooled_error_pp', 'Brier80', 'Brier90', 'Brier95', 'BrierMean']
FIVE = ['MSE', 'CRPS', 'QICE_percent', 'state_error_pp', 'BrierMean']
ARMS = ['Native', 'MeanOnly', 'ShapeOnly', 'FSPC']


def paired_ratios(rows, methods, metrics):
    lookup = {(r['id'], r['method']): r for r in rows}
    configurations = defaultdict(list)
    for r in rows:
        if r['method'] == 'Native':
            configurations[r['backbone'], r['dataset'], r['horizon']].append(r)
    result = {m: dict(method=m) for m in methods}
    for metric in metrics:
        run_values = {m: defaultdict(list) for m in methods}
        count = 0
        for backbone in sorted({key[0] for key in configurations}):
            eligible = [group for key, group in configurations.items() if key[0] == backbone
                and {r['seed'] for r in group} == {1, 2, 3}
                and all(lookup[r['id'], m][metric] is not None for r in group for m in ['Native']+methods)]
            if not eligible:
                raise ValueError('Missing complete eligible triples: '+backbone+'/'+metric)
            count += len(eligible)
            for seed in [1, 2, 3]:
                ids = [r['id'] for group in eligible for r in group if r['seed'] == seed]
                denominator = np.mean([lookup[i, 'Native'][metric] for i in ids])
                for m in methods:
                    run_values[m][seed].append(np.mean([lookup[i, m][metric] for i in ids])/denominator)
        for m in methods:
            values = [np.mean(run_values[m][seed]) for seed in [1, 2, 3]]
            result[m].update({metric+'_mean': float(np.mean(values)),
                metric+'_std': float(np.std(values, ddof=1)), metric+'_n_configurations': count})
    return list(result.values())


def export_group(folder, rows, methods, metrics):
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder/'raw_task_metrics.csv', rows)
    write_csv(folder/'backbone_statistics.csv', backbone_statistics(rows, metrics))
    write_csv(folder/'configuration_statistics.csv', configuration_statistics(rows, metrics))
    write_csv(folder/'native_ratios.csv', paired_ratios(rows, methods, metrics))


def main(root, main_reference=None):
    tasks = load(root/'manifest.json')
    variants = [method_name(*v) for v in VARIANTS]
    full, grain, completed, missing, errors, mismatches = [], [], [], [], [], []
    reference = {}
    if main_reference:
        reference = {(r['id'], r['method']): r for r in csv.DictReader(main_reference.open())}
    for task in tasks:
        path = root/'runs'/task['id']/'results.json'
        if not path.exists():
            missing.append(task['id'])
            continue
        try:
            r = load(path)
            assert r['version'] == VERSION and r['task'] == task and r['full_test_split']
            assert r['shared_coordinates'] and r['native_parity'] and r['frozen_fspc_parity']
            assert not r['author_score_overlays_applied'] and not r['neural_training_performed']
            assert set(r['methods']) == set(ARMS+['TrainFSPC']) and set(r['variants']) == set(variants)
            pair = {}
            for method, metric in r['methods'].items():
                verify_pair(metric, r['methods']['Native'])
                pair[method] = metric_row(task, method, metric)
                assert pair[method]['coordinate_count'] == r['coordinate_count']
                if (task['id'], method) in reference:
                    for key in FIVE:
                        a, b = pair[method][key], reference[task['id'], method][key]
                        if a is None and b in ('', 'None'):
                            continue
                        if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9):
                            mismatches.append(dict(task=task['id'], method=method, metric=key,
                                full_test_replay=a, table_iii_reference=float(b)))
            full.extend(pair.values())
            for key in variants:
                row = {field: task[field] for field in ['id', 'backbone', 'dataset', 'horizon', 'seed']}
                row.update(method=key, CRPS=r['variants'][key]['CRPS'],
                    coordinate_count=r['variants'][key]['coordinate_count'])
                row['fallback_percent'] = r['fallback_percent'][key]
                grain.append(row)
                if key == DEFAULT:
                    assert row['CRPS'] == pair['FSPC']['CRPS']
            row = {field: pair['Native'][field] for field in ['id', 'backbone', 'dataset', 'horizon', 'seed', 'method', 'CRPS', 'coordinate_count']}
            row['fallback_percent'] = None
            grain.append(row)
            completed.append(task['id'])
        except Exception as error:
            errors.append(dict(task=task['id'], error=repr(error)))
    complete = len(completed) == 249 and not missing and not errors
    audit = dict(version=VERSION, expected_tasks=249, completed_tasks=len(completed),
        all_passed=complete and (not reference or not mismatches),
        full_test_measurements_valid=complete,
        table_iii_reference_checked=bool(reference),
        table_iii_reference_consistent=(not mismatches) if reference else None,
        full_test_split=True, methods=ARMS+['TrainFSPC'], granularity_settings=13,
        missing_tasks=missing, errors=errors, table_iii_reference_mismatches=mismatches,
        all_default_rows_share_predictions=True, author_score_overlays_applied=False)
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    atomic(out/'audit.json', audit)
    if complete:
        factorial = [r for r in full if r['method'] in ARMS]
        export_group(out/'factorial', factorial, ARMS, METRICS)
        temporal = [r for r in full if r['method'] in ['Native', 'FSPC', 'TrainFSPC']]
        export_group(out/'temporal', temporal, ['FSPC', 'TrainFSPC'], METRICS)
        export_group(out/'granularity', grain, variants, ['CRPS'])
        write_csv(out/'granularity/fallback_backbone_statistics.csv',
            backbone_statistics(grain, ['fallback_percent']))
        print('Complete paired full-test aggregation:', len(completed), 'tasks')
    print(json.dumps({k: v for k, v in audit.items() if k not in ['missing_tasks', 'table_iii_reference_mismatches']}, indent=2))
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--main-reference', type=Path)
    args = p.parse_args()
    main(args.root, args.main_reference)
