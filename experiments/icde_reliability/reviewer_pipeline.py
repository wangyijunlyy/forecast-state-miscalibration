"""Fresh-data experiment adapter around the paper's unchanged FSPC functions.

This adapter does not read historical result files or fitted parameters.
"""
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
from fspc_unified import fit_unified
from final_fspc import CompiledFSPC, factorial_forecasts, save_parity_probe
from granularity import fit_partition, method_name
from portable_dc import summaries, edges, bounded, EmpiricalCalibrator
from reliability_metrics import ReliabilityMetrics


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def score(rows, targets, channel, groups, thresholds, ordered=True):
    acc = ReliabilityMetrics(thresholds[channel], n_states=36)
    acc.update(rows.T[None, :, None, :], targets[None, None, :], groups, ordered=ordered)
    return acc.result()


def diagnostic_edges(train):
    m, _, s = summaries(train)
    return edges(m, 6), edges(np.log(s), 6)


def diagnostic_groups(rows, bounds):
    m, _, s = summaries(rows); me, se = bounds
    return 6*np.searchsorted(me[1:-1], m, side='right')+np.searchsorted(se[1:-1], np.log(s), side='right')


def fit_suite(train, val, horizon, n_windows, stride, seed, controls=False, external=False):
    args = (train['rows'], train['targets'], train['horizon'], val['rows'], val['targets'],
        val['horizon'], val['window'], n_windows, stride, horizon)
    durations = []; compilation = []
    # Full fitting includes the mean models, purged selection, quantile grid and refit.
    for _ in range(3):
        begin = time.perf_counter(); model, meta = fit_unified(*args)
        durations.append(time.perf_counter()-begin)
        begin = time.perf_counter(); compiled = CompiledFSPC.from_model(model)
        compilation.append(time.perf_counter()-begin)
    models = {}; fitting = {}; assignment = {}
    if controls:
        from fspc_granularity import VARIANTS
        for b, g, budget in VARIANTS:
            name = method_name(b, g, budget)
            models[name], fitting[name] = fit_unified(*args, bins=b, horizon_groups=g,
                budget=budget, ordering_seed=seed+60000)
        models['TrainFSPC'], fitting['TrainFSPC'] = fit_unified(*args, template_split='train')
        for name, source_filter in [('DC36', True), ('UnfilteredDC36', False)]:
            cal, fitting[name] = fit_partition(val['rows'], val['targets'], val['horizon'],
                horizon, 6, 1, source_filter=source_filter)
            models[name] = cal.compile()
            if name == 'DC36': genuine = cal
        # Same residual mask, occupancy and fallback/global pool; change only association.
        mean, raw, scale = summaries(val['rows']); valid = raw > 1.01e-6
        residual = (val['targets']-mean)/scale
        keys, inverse = np.unique(genuine.keys(mean, scale, val['horizon']), axis=0, return_inverse=True)
        global_cal, _ = fit_partition(val['rows'], val['targets'], val['horizon'], horizon, 1, 1)
        assignment['GlobalTemplate'] = global_cal.compile()
        assignment['Genuine'] = genuine.compile()
        for repeat in range(5):
            z = residual.copy()
            z[valid] = residual[valid][np.random.default_rng(seed+80000+repeat).permutation(valid.sum())]
            pools = {tuple(k): bounded(z[(inverse == i)&valid]) for i, k in enumerate(keys)
                if ((inverse == i)&valid).sum() >= 128}
            if pools.keys() != genuine.local_pools.keys():
                raise ValueError('Shuffling changed local eligibility')
            changed = EmpiricalCalibrator(genuine.mean_edges, genuine.scale_edges,
                genuine.global_pool, pools, horizon, 1, int(valid.sum()), int((~valid).sum()))
            assignment['Shuffled'+str(repeat)] = changed.compile()
    external_models = {}
    if external:
        from external_baselines import EMOS, SoftCRPSPIT, QRF
        external_models['EMOS'] = EMOS.fit(val['rows'], val['targets'])
        external_models['Qian2025SoftCRPS'] = SoftCRPSPIT.fit(val['rows'], val['targets'], seed+71000)
        external_models['QRF2016'] = QRF.fit(val['rows'], val['targets'], seed+73000)
    return dict(model=model, compiled=compiled, metadata=meta, models=models,
        assignment=assignment, external=external_models, control_fitting=fitting,
        fit_seconds=durations, compilation_seconds=compilation,
        diagnostic_edges=diagnostic_edges(train['rows']))


def predictions(suite, rows, horizon):
    result = factorial_forecasts(suite['model'], rows, horizon)
    for name, model in suite['external'].items(): result[name] = model.predict(rows)
    return result


def retained_evaluation(suite, test, thresholds, output, seed):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows, y, h, c = [test[k] for k in ['rows', 'targets', 'horizon', 'channel']]
    groups = diagnostic_groups(rows, suite['diagnostic_edges'])
    arms = predictions(suite, rows, h)
    metric = {name: score(q, y, c, groups, thresholds, name not in ['Native', 'MeanOnly'])
        for name, q in arms.items()}
    # All control models were fitted before this function accessed test targets.
    controls = {}
    for name, model in suite['models'].items():
        q = model.predict(rows, h) if hasattr(model, 'predict') else model.transform_rows(rows, h)
        controls[name] = score(q, y, c, groups, thresholds)
    assignment = {name: score(model.transform_rows(rows, h), y, c, groups, thresholds)
        for name, model in suite['assignment'].items()}
    reference = arms['FSPC']; compiled = suite['compiled']
    actual = compiled.transform_rows(rows, h)
    np.testing.assert_array_equal(actual, reference)
    coordinate_thresholds = thresholds[c]
    def queries(q, at=coordinate_thresholds):
        return np.stack([(q > at[:, j, None]).mean(1) for j in range(at.shape[1])], axis=1)
    expected = queries(reference)
    np.testing.assert_array_equal(compiled.probabilities_rows(rows, h, coordinate_thresholds), expected)
    ties = reference[:, [0, 49, 99]]
    np.testing.assert_array_equal(compiled.probabilities_rows(rows, h, ties), queries(reference, ties))
    compiled.save(output/'compiled.npz'); loaded = CompiledFSPC.load(output/'compiled.npz')
    np.testing.assert_array_equal(loaded.transform_rows(rows, h), reference)
    np.testing.assert_array_equal(loaded.probabilities_rows(rows, h, coordinate_thresholds), expected)
    save_parity_probe(output/'parity_probe.npz', rows, h, coordinate_thresholds, reference)
    with np.load(output/'parity_probe.npz') as probe:
        np.testing.assert_array_equal(loaded.transform_rows(probe['rows'], probe['horizon']), probe['reference'])
    mean, _, scale = summaries(rows)
    np.testing.assert_array_equal(compiled.probabilities_summaries(mean, scale, h, coordinate_thresholds), expected)
    np.testing.assert_allclose(((expected-(y[:, None] > coordinate_thresholds))**2).mean(0),
        metric['FSPC']['events']['brier'], rtol=1e-12, atol=1e-12)
    operations = dict(reference_decode=lambda: suite['model'].predict(rows, h),
        compiled_decode=lambda: compiled.transform_rows(rows, h),
        reference_decode_and_queries=lambda: queries(suite['model'].predict(rows, h)),
        compiled_decode_and_queries=lambda: queries(compiled.transform_rows(rows, h)),
        direct_queries_from_rows=lambda: compiled.probabilities_rows(rows, h, coordinate_thresholds),
        direct_queries_from_summaries=lambda: compiled.probabilities_summaries(mean, scale, h, coordinate_thresholds))
    for operation in operations.values(): operation()
    times = {name: [] for name in operations}; rng = np.random.default_rng(20261110+seed)
    for _ in range(3):
        for name in rng.permutation(list(operations)):
            begin = time.perf_counter(); value = operations[name]()
            times[name].append(time.perf_counter()-begin); del value
    contrasts = {}
    for key in ['MSE', 'CRPS', 'QICE_percent']:
        contrasts[key] = metric['FSPC'][key]-metric['MeanOnly'][key]-metric['ShapeOnly'][key]+metric['Native'][key]
    result = dict(coordinates=len(y), full_test_split=False, methods=metric, controls=controls,
        state_assignment=assignment, interaction=contrasts,
        parity=dict(exact_reconstructed_values=True, maximum_value_difference=0.,
            exact_event_probabilities=True, exact_boundary_ties=True, saved_state_reload=True),
        cost=dict(full_fit_seconds=suite['fit_seconds'], compilation_seconds=suite['compilation_seconds'],
            times=times, median_seconds={k: float(np.median(v)) for k, v in times.items()},
            reference_pickle_bytes=len(pickle.dumps(suite['model'], protocol=4)),
            compiled_npz_bytes=(output/'compiled.npz').stat().st_size,
            excludes_native_generation_file_io_and_scoring=True),
        environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()))
    atomic(output/'retained_results.json', result)
    return result
