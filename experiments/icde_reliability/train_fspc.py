"""Training-template FSPC with unchanged validation-selected mean control."""
import numpy as np
from portable_dc import EmpiricalCalibrator
from mean_anchor import fit_bias, select_strength

VERSION = 'train-fspc36-v1-20261103'
METHODS = ['Native', 'TrainDC36', 'DC36', 'DC36ControlledMean', 'TrainFSPC']


def fit_training_variant(train, targets, horizon, validation, val_targets, val_horizon,
                         horizon_length):
    empirical = EmpiricalCalibrator.fit_rows(train, targets, horizon, horizon_length,
        n_horizon_bins=1, source_filter=True, summary_bins=6)
    cal = empirical.compile()
    bias, counts = fit_bias(cal, train, targets, horizon)
    strength, selection = select_strength(cal, validation, val_targets, val_horizon, bias)
    meta = dict(summary_bins=6, horizon_groups=1, output_members=100,
        template_fit_split='train', template_centered=True, pseudo_count=128, min_bias_count=128,
        state_bias=bias.tolist(), training_state_counts=counts.tolist(), selection=selection,
        training_coordinates=len(targets), retained_sources=empirical.retained_sources,
        excluded_sources=empirical.excluded_sources, local_pools=len(empirical.local_pools),
        alpha_frozen_before_test=True, neural_training_performed=False)
    return cal, bias, strength, meta


def validate_record(value, task):
    fit = value['fit']
    checks = [value['task'] == task, fit['task'] == task,
        value['version'] == fit['version'] == VERSION,
        set(value['methods']) == set(METHODS), value['full_test_split'],
        value['test_seed'] == task['seed'] + 30000,
        not value['baseline_metrics_recomputed'], value['newly_evaluated_methods'] == ['TrainFSPC'],
        not value['neural_training_performed'], fit['template_fit_split'] == 'train',
        fit['summary_bins'] == 6, fit['horizon_groups'] == 1, fit['output_members'] == 100,
        fit['pseudo_count'] == fit['min_bias_count'] == 128, fit['template_centered'],
        fit['alpha_frozen_before_test'], fit['training_template_reference_parity'],
        fit['selection']['bias_fit_split'] == 'train',
        fit['selection']['selection_split'] == 'validation',
        not fit['selection']['test_targets_used'], 0 <= fit['selection']['strength'] <= 1]
    if not all(checks):
        raise ValueError('Training-fit FSPC record mismatch: ' + task['id'])


def verify_training_template(cal, reference):
    """Compare compiled output with the frozen historical TrainDC36 pools."""
    np.testing.assert_array_equal(cal.mean_edges, reference.mean_edges)
    np.testing.assert_array_equal(cal.scale_edges, reference.scale_edges)
    grid = (np.arange(100) + .5) / 100
    for m in range(6):
        for s in range(6):
            pool = reference.bin_residuals.get((0, m, s), reference.global_residuals)
            np.testing.assert_array_equal(cal.quantiles[cal.state_to_template[6*m+s]],
                                          np.quantile(pool, grid))
