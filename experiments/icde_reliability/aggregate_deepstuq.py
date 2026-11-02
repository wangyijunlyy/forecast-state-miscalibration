"""Audit and summarize the full, separately matched ICDE traffic baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric, configuration_statistics, backbone_statistics, write_csv, close, METRICS

DATASETS = {'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'}
METHODS = {'DeepSTUQ', 'DeepSTUQ+DC16', 'DeepSTUQ+GlobalDC', 'DeepSTUQ+PIT', 'NoTemperature'}
BASELINE_METRICS = [key for key in METRICS if key != 'transform_seconds']


def audit(path):
    result = json.loads(path.read_text()); training = json.loads((path.parent/'training.json').read_text())
    history = json.loads((path.parent/'training_history.json').read_text())
    temperature = json.loads((path.parent/'temperature.json').read_text())
    dataset, seed = result['dataset'], result['seed']
    if dataset not in DATASETS or seed not in (1, 2, 3) or path.parent.name != f'{dataset}_seed{seed}':
        raise ValueError('Unexpected traffic task identity')
    if result['version'] != 'full-deepstuq-icde2023-v1' or set(result['methods']) != METHODS:
        raise ValueError('Baseline version or method identity mismatch')
    if not training['training_complete'] or training['awa_epochs'] != 20 or training['awa_snapshots'] != 10:
        raise ValueError('Incomplete full baseline training stages')
    if training['args']['seed'] != seed or training['args']['dataset'] != dataset:
        raise ValueError('Training seed/dataset mismatch')
    if training['author_revision'] != '4f2281531d07b48ba81aa1c519a90e65a71993e2':
        raise ValueError('Unexpected author source revision')
    if not (1 <= len(history) <= 100) or len(history) != training['pretraining_epochs']:
        raise ValueError('Missing full pretraining history')
    values = np.array([[h['train_loss'], h['validation_loss']] for h in history])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite training history')
    best = history[int(np.argmin(values[:, 1]))]['epoch']
    if best != training['best_pretraining_epoch']:
        raise ValueError('Validation checkpoint-selection mismatch')
    if temperature['mc_samples'] != 10 or temperature['lbfgs_max_iter'] != 500:
        raise ValueError('Incomplete native temperature calibration')
    close(temperature['log_temperature'], result['native_temperature'], 'Temperature mismatch')
    if not result['full_test_split'] or result['test_windows'] <= 0 or result['test_batches'] <= 0:
        raise ValueError('Incomplete native traffic test iterator')
    if result['output_members'] != 100 or result['native_dropout_forward_passes'] != 10:
        raise ValueError('Native uncertainty/output representation mismatch')
    native = result['methods']['DeepSTUQ']; counts = native['state_coverage']['counts']
    if native['coordinate_count'] != result['test_windows']*12*training['args']['num_nodes']:
        raise ValueError('Native traffic test population mismatch')
    for metric in result['methods'].values():
        audit_metric(metric)
        if metric['state_coverage']['counts'] != counts:
            raise ValueError('Methods used different diagnostic populations')
        close(metric['events']['observed_frequencies'], native['events']['observed_frequencies'], 'Traffic events differ')
    gaussian = result['native_gaussian']
    if gaussian['NLL_convention'] != 'author GaussianNLLLoss(full=False, eps=1e-6)':
        raise ValueError('Native NLL convention mismatch')
    if not all(np.isfinite(gaussian[k]) for k in ['NLL', 'PICP95', 'MPIW95']):
        raise ValueError('Invalid native Gaussian metrics')
    if not 0 <= gaussian['PICP95'] <= 1 or gaussian['MPIW95'] < 0:
        raise ValueError('Invalid native interval scores')
    rows = []
    for method, metric in result['methods'].items():
        state = metric['state_coverage']
        row = dict(backbone='DeepSTUQ', dataset=dataset, horizon=12, seed=seed, method=method,
                   coordinate_count=metric['coordinate_count'], eligible_coordinates=state['eligible_coordinates'])
        row.update({k:metric[k] for k in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE', 'MedianMAE']})
        row.update(state_error_pp=None if state['state_error'] is None else 100*state['state_error'],
                   pooled_error_pp=None if state['pooled_error'] is None else 100*state['pooled_error'])
        row.update(zip(['Brier80', 'Brier90', 'Brier95'], metric['events']['brier']))
        row['BrierMean'] = float(np.mean(metric['events']['brier']))
        rows.append(row)
    return (dataset, seed), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = {(d, s) for d in DATASETS for s in (1, 2, 3)}
    completed = set(); rows = []; errors = []
    for path in sorted(args.runs.glob('*/results.json')):
        try:
            identity, values = audit(path)
            if identity in completed:
                raise ValueError('Duplicate baseline identity')
            completed.add(identity); rows.extend(values)
        except Exception as error:
            errors.append(dict(path=str(path), error=str(error)))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'raw_task_metrics.csv', rows)
    write_csv(args.output/'complete_dataset_statistics.csv', configuration_statistics(rows, BASELINE_METRICS))
    write_csv(args.output/'partial_suite_statistics.csv', backbone_statistics(rows, BASELINE_METRICS))
    report = dict(expected_tasks=12, audited_tasks=len(completed), all_tasks_complete=completed == expected and not errors,
                  missing_tasks=sorted(f'{d}_seed{s}' for d,s in expected-completed), errors=errors,
                  suite='separate original PeMS traffic tasks, not absolute cross-suite rankings',
                  manuscript_standard_deviations='Measured sample SD rounded to three decimals; raw statistics retain full precision')
    (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
