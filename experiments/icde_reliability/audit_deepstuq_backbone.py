"""Check seventh-backbone coverage against native records and common protocols."""
import argparse
import json
from pathlib import Path
import numpy as np
from aggregation import audit_run, close
from aggregate_gaussian import audit as audit_gaussian


def audit(root, native_root, require_all=True):
    tasks = json.loads((root/'manifest.json').read_text())
    expected = {t['id']: t for t in tasks}
    assert {(t['dataset'], t['seed']) for t in tasks} == {
        (d, s) for d in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'] for s in [1, 2, 3]}
    complete = []; errors = []; missing = []
    for task in tasks:
        identity = task['id']; directory = root/'runs'/identity
        if not (directory/'results.json').exists():
            missing.append(identity); continue
        try:
            result = json.loads((directory/'results.json').read_text())
            validation = json.loads((directory/'validation.json').read_text())
            control = json.loads((directory/'controls.json').read_text())
            audit_run(result, control, validation)
            comparison = json.loads((directory/'training_partition_comparison.json').read_text())
            assert comparison['full_test_split'] and comparison['test_windows'] == result['test_windows']
            assert set(comparison['methods']) == {'Native', 'TrainDC16', 'TrainDC64'}
            from aggregation import audit_metric
            for name, metric in comparison['methods'].items():
                audit_metric(metric)
                assert metric['coordinate_count'] == result['methods']['Native']['coordinate_count']
                if name != 'TrainDC64':
                    for key in ['CRPS', 'QICE_percent', 'MSE']:
                        close(metric[key], result['methods'][name][key], 'Training-partition native parity failed')
            assert result['task'] == task
            original_dir = native_root/f"{task['dataset']}_seed{task['seed']}"
            original = json.loads((original_dir/'results.json').read_text())
            training = json.loads((original_dir/'training.json').read_text())
            assert training['training_complete'] and training['awa_epochs'] == 20 and training['awa_snapshots'] == 10
            assert result['test_windows'] == original['test_windows']
            assert result['test_batches'] == original['test_batches']
            for key in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE', 'MedianMAE']:
                close(result['methods']['Native'][key], original['methods']['DeepSTUQ'][key], 'Original native marginal parity failed')
            train_fit = json.loads((root/'train_calibration/runs'/identity/'fit_metadata.json').read_text())
            assert train_fit['fit_split'] == 'train' and train_fit['prediction_seed'] == task['seed']+10000
            assert 0 < train_fit['fit_batches'] <= 100
            assert train_fit['retained_points'] == 50000
            assert validation['calibration_fit_limit'] == 100
            assert validation['full_validation_split'] and validation['checkpoint_selection_uses_validation']
            assert result['native_dropout_forward_passes'] == 10 and result['output_members'] == 100
            assert result['diagnostic_partition'].startswith('training-fitted')
            gaussian_id, _ = audit_gaussian(root/'gaussian_runs'/identity/'results.json', root/'runs', expected)
            assert gaussian_id == identity
            native_count = result['methods']['Native']['coordinate_count']
            assert native_count == result['test_windows']*12*training['args']['num_nodes']
            assert control['retained_test_coordinates'] == min(50000, native_count)
            with np.load(directory/'calibration.npz') as fitting:
                assert fitting['rows'].shape == (50000, 100)
                assert fitting['thresholds'].shape == (training['args']['num_nodes'], 3)
            complete.append(identity)
        except Exception as error:
            errors.append(dict(task=identity, error=repr(error)))
    report = dict(expected_tasks=12, audited_tasks=len(complete), missing_tasks=missing, errors=errors,
        all_tasks_complete=len(complete) == 12 and not errors,
        original_training_preserved=True, dataset_scope='All four datasets from DeepSTUQ Section V-A',
        methods='The same six primary methods and two Gaussian controls as the other six backbones',
        neural_weights_distributed=False)
    output = root/'backbone_audit.json'; output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if errors or (require_all and missing):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--native-runs', type=Path, required=True)
    parser.add_argument('--allow-pending', action='store_true')
    args = parser.parse_args(); audit(args.root, args.native_runs, not args.allow_pending)
