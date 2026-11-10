"""Fit/evaluate final FSPC using newly supplied ensemble arrays; CPU supported."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/icde_reliability'))
from reviewer_pipeline import fit_suite, retained_evaluation, atomic


def coordinates(path, validation=False):
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    required = {'rows', 'targets', 'horizon', 'channel'}
    if validation: required |= {'window', 'n_windows', 'origin_stride', 'thresholds'}
    if not required.issubset(result): raise ValueError(f'{path}: missing {required-set(result)}')
    rows = result['rows']; n = len(rows)
    if rows.ndim != 2 or rows.shape[1] != 100 or not np.isfinite(rows).all():
        raise ValueError('Expected finite [N,100] ensemble rows')
    for key in ['targets', 'horizon', 'channel']:
        if result[key].shape != (n,): raise ValueError('One '+key+' value required per row')
    if not np.isfinite(result['targets']).all(): raise ValueError('Nonfinite targets')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['train', 'validation', 'test', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--horizon', type=int, required=True); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--controls', action='store_true'); p.add_argument('--external', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'retained_results.json').exists(): raise ValueError('Choose a fresh output directory')
    train = coordinates(a.train); val = coordinates(a.validation, validation=True)
    suite = fit_suite(train, val, a.horizon, int(val['n_windows']), int(val['origin_stride']),
        a.seed, a.controls, a.external)
    atomic(a.output/'parameters.json', suite['model'].to_record())
    atomic(a.output/'frozen.json', dict(fit=suite['metadata'], parameters_frozen_before_test=True,
        test_targets_used_for_fitting=False, control_fitting=suite['control_fitting']))
    # Test file is opened only after every requested model is fitted/frozen.
    test = coordinates(a.test)
    result = retained_evaluation(suite, test, val['thresholds'], a.output, a.seed)
    print('FSPC CRPS:', result['methods']['FSPC']['CRPS'])
    print('Exact compilation checks:', result['parity'])


if __name__ == '__main__':
    main()
