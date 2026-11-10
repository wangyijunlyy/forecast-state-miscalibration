"""Full-test paired audits and Native-normalized, seed-aware all-backbone tables."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric, backbone_statistics, configuration_statistics, write_csv
from aggregate_fsdc36 import metric_row
from aggregate_mean_anchor import check_selection, same, METRICS, LABELS
from mean_anchor_campaign import METHODS, validate_tasks, validate_record, version_for, source_proof_matches

KEYS = ['MSE', 'CRPS', 'QICE_percent', 'BrierMean']


def normalized_ratios(rows, metrics=METRICS, reference='Native'):
    """Configuration mean within seed -> paired Native ratio -> equal backbones."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['backbone'], row['method'], row['seed']].append(row)
    methods = sorted({r['method'] for r in rows})
    backbones = sorted({r['backbone'] for r in rows})
    paired = []
    for backbone in backbones:
        for method in methods:
            for seed in [1, 2, 3]:
                values = groups[backbone, method, seed]
                native = groups[backbone, reference, seed]
                assert values and len(values) == len(native)
                record = dict(backbone=backbone, method=method, seed=seed)
                for key in metrics:
                    candidate = [r[key] for r in values]
                    reference_values = [r[key] for r in native]
                    if any(v is None for v in candidate+reference_values) or np.mean(reference_values) <= 0:
                        record[key] = None
                    else:
                        record[key] = float(np.mean(candidate)/np.mean(reference_values))
                paired.append(record)
    summary = []
    for method in methods:
        record = dict(method=method, n_backbones=len(backbones), n_seeds=3)
        for key in metrics:
            values = [[r[key] for r in paired if r['method'] == method and r['seed'] == seed]
                      for seed in [1, 2, 3]]
            if any(v is None for group in values for v in group):
                record[key+'_mean'] = record[key+'_std'] = None
            else:
                runs = np.mean(values, axis=1)
                record[key+'_mean'] = float(np.mean(runs))
                record[key+'_std'] = float(np.std(runs, ddof=1))
        summary.append(record)
    return summary, paired


def paired_changes(rows):
    lookup = {(r['id'], r['method']): r for r in rows}
    output = []
    for row in rows:
        if row['method'] == 'Native':
            continue
        for reference in ['Native', 'DC36']:
            if reference == row['method']:
                continue
            baseline = lookup[row['id'], reference]
            record = {k: row[k] for k in ['id', 'backbone', 'dataset', 'horizon', 'seed', 'method']}
            record['reference'] = reference
            for key in METRICS:
                record[key+'_change_percent'] = (100*(row[key]/baseline[key]-1)
                    if row[key] is not None and baseline[key] is not None and baseline[key] > 0 else None)
            output.append(record)
    return output


def aggregate(root, reference, selection_source=None):
    tasks = json.loads((root/'manifest.json').read_text())
    scope = validate_tasks(tasks, reference=json.loads((reference/'manifest.json').read_text()))
    assert scope['tasks'] == 249 and scope['configurations'] == 83
    out = root/'analysis'; out.mkdir(exist_ok=True)
    rows = []; cohorts = []; selection = []; complete = []; errors = []; fits = {}; reused_baselines = 0
    reused = json.loads((root/'reused_records.json').read_text())
    assert {r['id'] for r in reused} == {t['id'] for t in tasks if t['backbone']=='DiffusionTS'}
    assert len(reused)==27 and all(r['identity_and_protocol_validated'] for r in reused)
    for task in tasks:
        identity = task['id']; path = root/'runs'/identity/'results.json'
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text()); validate_record(value, task)
            if not value.get('baseline_metrics_recomputed',True):
                assert value['newly_evaluated_methods'] == METHODS[2:]
            cohort = json.loads((path.parent/'cohort.json').read_text())
            assert cohort.get('baseline_metrics_recomputed',True) == value.get('baseline_metrics_recomputed',True)
            assert cohort.get('newly_evaluated_methods',METHODS) == value.get('newly_evaluated_methods',METHODS)
            old = json.loads((reference/'runs'/identity/'results.json').read_text())
            old_cohort = json.loads((reference/'cohort/runs'/identity/'results.json').read_text())
            assert old['task'] == old_cohort['task'] == task
            assert old['version'] == 'icde-fsdc36-v1-20261103'
            assert old_cohort['version'] == 'fsdc36-cohort-v1-20261103'
            fit = value['fit']; alpha = fit['selection']['strength']
            assert cohort['task'] == task and cohort['version'] == version_for(task)
            assert cohort['fit'] == fit and set(cohort['methods']) == set(METHODS)
            assert json.loads((path.parent/'fit.json').read_text()) == fit
            assert value['test_batches'] == old['test_batches'] and value['test_windows'] == old['test_windows']
            assert fit['selection']['selected_mse'] <= fit['selection']['native_mse']+1e-12
            with np.load(path.parent/'mean_parameters.npz') as data:
                same(data['bias'], fit['state_bias']); same(data['counts'], fit['training_state_counts'])
                same(data['strength'], alpha)
            if selection_source is not None:
                check_selection(task, fit, selection_source)
            task_rows = []; cohort_rows = []
            for container, target in [(value, task_rows), (cohort, cohort_rows)]:
                native = container['methods']['Native']
                for name, metric in container['methods'].items():
                    audit_metric(metric)
                    assert metric['coordinate_count'] == native['coordinate_count']
                    np.testing.assert_array_equal(metric['state_coverage']['counts'], native['state_coverage']['counts'])
                    same(metric['events']['observed_frequencies'], native['events']['observed_frequencies'])
                    target.append(metric_row(task, name, metric))
                same(container['methods']['DC36MeanPreserving']['MSE'], native['MSE'])
            for name in ['Native', 'DC36']:
                for key in ['CRPS', 'QICE_percent', 'MSE', 'MAE', 'RMSE']:
                    same(value['methods'][name][key], old['methods'][name][key])
                    cohort_reference = (old_cohort['native'] if name == 'Native' else
                                        old_cohort['methods']['b6_h1_n50000'])
                    same(cohort['methods'][name][key], cohort_reference[key])
            np.testing.assert_array_equal(value['methods']['Native']['state_coverage']['counts'],
                                          old['methods']['Native']['state_coverage']['counts'])
            np.testing.assert_array_equal(cohort['methods']['Native']['state_coverage']['counts'],
                                          old_cohort['native']['state_coverage']['counts'])
            assert max(value['maximum_mean_deviation'].values()) < 1e-8
            selection.append(dict(id=identity, **{k:task[k] for k in ['backbone','dataset','horizon','seed']},
                strength=alpha, native_validation_mse=fit['selection']['native_mse'],
                selected_validation_mse=fit['selection']['selected_mse'],
                active_bias_states=int((np.array(fit['training_state_counts'])>=128).sum()),
                maximum_mean_deviation=max(value['maximum_mean_deviation'].values())))
            rows.extend(task_rows); cohorts.extend(cohort_rows); complete.append(identity); fits[identity]=fit
            reused_baselines += int(not value.get('baseline_metrics_recomputed',True))
        except Exception as error:
            errors.append(dict(id=identity, error=repr(error)))
    missing = sorted({t['id'] for t in tasks}-set(complete))
    audit = dict(all_passed=len(complete)==249 and not errors, expected_tasks=249,
        audited_tasks=len(complete), missing_tasks=missing, errors=errors, **scope,
        methods=METHODS, default_states=36, full_test_and_retained_cohort_reported_separately=True,
        native_and_original_fsdc_records_match_reference=True,
        baseline_metrics_recomputed_tasks=len(complete)-reused_baselines,
        baseline_metrics_reused_tasks=reused_baselines, mean_preserving_mse_matches_native=True,
        parameters_selected_without_test_targets=True, neural_training_performed=False)
    proof = out/'source_selection_audit.json'
    if selection_source is not None and audit['all_passed']:
        proof.write_text(json.dumps(dict(all_passed=True, tasks=249,
            independent_train_bias_and_validation_strength_reconstruction=True,
            test_targets_used=False, manifest=tasks, fits=fits), indent=2)+'\n')
    audit['raw_train_validation_sources_independently_audited'] = (proof.exists() and
        source_proof_matches(json.loads(proof.read_text()),tasks,fits))
    audit['metric_and_record_audit_passed'] = audit['all_passed']
    audit['all_passed'] = audit['all_passed'] and audit['raw_train_validation_sources_independently_audited']
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    write_csv(out/'mean_selection.csv', selection)
    if not audit['all_passed']:
        print(json.dumps(audit)); return audit
    for folder, data in [('full_test', rows), ('retained_cohort', cohorts)]:
        directory = out/folder; directory.mkdir(exist_ok=True)
        write_csv(directory/'raw_task_metrics.csv', data)
        write_csv(directory/'configuration_statistics.csv', configuration_statistics(data, METRICS))
        write_csv(directory/'backbone_statistics.csv', backbone_statistics(data, METRICS))
        ratios, paired = normalized_ratios(data)
        write_csv(directory/'native_normalized_ratios.csv', ratios)
        write_csv(directory/'backbone_seed_ratios.csv', paired)
        fsdc_ratios, fsdc_paired = normalized_ratios(data, reference='DC36')
        write_csv(directory/'fsdc_normalized_ratios.csv', fsdc_ratios)
        write_csv(directory/'backbone_seed_ratios_vs_fsdc.csv', fsdc_paired)
        write_csv(directory/'paired_changes.csv', paired_changes(data))
    full = backbone_statistics(rows, METRICS); configs = configuration_statistics(rows, METRICS)
    ratio, paired = normalized_ratios(rows); by_ratio = {r['method']:r for r in ratio}
    fsdc_ratio, _ = normalized_ratios(rows, reference='DC36')
    fsdc_controlled = next(r for r in fsdc_ratio if r['method']=='DC36ControlledMean')
    by_config = {(r['backbone'],r['dataset'],r['horizon'],r['method']):r for r in configs}
    identities = sorted({(t['backbone'],t['dataset'],t['horizon']) for t in tasks})
    wins = {}; adverse = []
    for method in METHODS[1:]:
        wins[method] = {}
        for ref in ['Native','DC36']:
            if method == ref: continue
            wins[method][ref] = {}
            for metric in KEYS:
                delta = np.array([by_config[identity+(method,)][metric+'_mean']-
                                  by_config[identity+(ref,)][metric+'_mean'] for identity in identities])
                wins[method][ref][metric] = dict(improved=int((delta < -1e-10).sum()),
                    worsened=int((delta > 1e-10).sum()), tied=int((abs(delta)<=1e-10).sum()))
    for identity in identities:
        candidate = by_config[identity+('DC36ControlledMean',)]
        native = by_config[identity+('Native',)]
        record = dict(zip(['backbone','dataset','horizon'], identity))
        record.update({key+'_change_percent':100*(candidate[key+'_mean']/native[key+'_mean']-1) for key in KEYS})
        if any(record[key+'_change_percent'] > 1e-7 for key in ['MSE','CRPS']): adverse.append(record)
    write_csv(out/'adverse_configurations_vs_native.csv', adverse)
    summary = dict(audit=audit, full_test=full, native_normalized_ratios=ratio,
        fsdc_normalized_ratios=fsdc_ratio,
        configuration_wins=wins, adverse_configurations_vs_native=adverse,
        alpha_min=min(r['strength'] for r in selection), alpha_max=max(r['strength'] for r in selection),
        alpha_zero=sum(r['strength']==0 for r in selection), alpha_one=sum(r['strength']==1 for r in selection),
        source_protocol='MEAN_ANCHOR_ALL_PROTOCOL.md', manuscript_default_changed=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report = ['# Controlled-mean FSDC: all seven backbones','',
        '249 paired runs: 83 configurations × three original seeds. Each run consumes the full native test iterator. '
        'The 27 Diffusion-TS runs reuse the fixed pilot; the other 222 runs use the identical frozen algorithm. '
        'Native and original 36-state FSDC records match the reference full-test scores. No neural training is repeated.',
        f"Baseline metrics were already evaluated in {audit['baseline_metrics_recomputed_tasks']} runs; "
        f"{audit['baseline_metrics_reused_tasks']} subsequent runs reuse the frozen baseline records and evaluate only "
        'the mean-preserving and controlled-mean versions, following the user instruction. Native sample '
        'generation remains necessary to evaluate both new versions.','',
        'Mean bias uses training coordinates, with pseudo-count 128 and a minimum of 128 state coordinates. '
        'One strength in [0,1] is selected from the existing validation coordinates before test evaluation. '
        'Validation also fits the original distribution templates and served original checkpoint selection. '
        'No test targets enter fitting or selection.','',
        '## Native-normalized aggregate','',
        'Native = 1; lower is better. Within each seed, average configurations within each backbone, divide '
        'by its paired Native score, then equally average all seven backbones. Report mean ± sample SD over '
        'three seeds. Absolute traffic errors are not pooled with normalized time-series errors.','',
        '| Method | MSE ratio | CRPS ratio | QICE ratio | Brier ratio |',
        '|---|---:|---:|---:|---:|']
    for name in METHODS:
        report.append('| '+LABELS[name]+' | '+' | '.join(
            f"{by_ratio[name][k+'_mean']:.6f} ± {by_ratio[name][k+'_std']:.6f}" for k in KEYS)+' |')
    report += ['', 'Controlled mean versus original FSDC, using the same paired backbone/run '
        'normalization with original FSDC = 1: '+', '.join(
            f"{key} {fsdc_controlled[key+'_mean']:.6f} ± {fsdc_controlled[key+'_std']:.6f} "
            f"({100*(fsdc_controlled[key+'_mean']-1):+.3f}%)" for key in KEYS)+'.']
    report += ['', '## Controlled mean: configuration-level comparison','',
        'Compare three-seed configuration means; numerical ties use an absolute tolerance of 1e-10.','',
        '| Reference | Metric | Improved | Worsened | Tied |','|---|---|---:|---:|---:|']
    for ref in ['Native','DC36']:
        for key in KEYS:
            w = wins['DC36ControlledMean'][ref][key]
            report.append(f"| {LABELS[ref]} | {key} | {w['improved']} | {w['worsened']} | {w['tied']} |")
    report += ['', '## Absolute backbone scores','',
        'Equally average configurations within each seed, then mean ± sample SD across three seeds.','',
        '| Backbone | Method | MSE | CRPS | QICE (pp) | Brier |','|---|---|---:|---:|---:|---:|']
    for backbone in sorted(scope['backbones']):
        for method in METHODS:
            value = next(r for r in full if r['backbone']==backbone and r['method']==method)
            report.append('| '+backbone+' | '+LABELS[method]+' | '+' | '.join(
                f"{value[k+'_mean']:.6f} ± {value[k+'_std']:.6f}" for k in KEYS)+' |')
    report += ['', '## All 83 configurations','',
        'Scores are three-seed means. Each metric cell is Native / original FSDC / controlled-mean FSDC. '
        'Full means and SDs for all four methods and all twelve metrics are in the configuration CSV.','',
        '| Backbone | Dataset | Horizon | MSE | CRPS | QICE (pp) | Brier |',
        '|---|---|---:|---:|---:|---:|---:|']
    for identity in identities:
        cells = [' / '.join(f"{by_config[identity+(name,)][k+'_mean']:.6f}" for name in
                 ['Native','DC36','DC36ControlledMean']) for k in KEYS]
        report.append('| '+' | '.join(map(str,identity))+' | '+' | '.join(cells)+' |')
    report += ['', '## Mean strength and records','',
        f"Selected strength: {summary['alpha_zero']}/249 zero, {summary['alpha_one']}/249 one, "
        f"{249-summary['alpha_zero']-summary['alpha_one']}/249 interior. Per-task values are in analysis/mean_selection.csv.",
        'The mean-preserving control matches Native MSE in all 249 tasks. Controlled-mean test MSE can '
        'increase despite its validation constraint; every adverse configuration is retained in '
        'analysis/adverse_configurations_vs_native.csv. Existing manuscript defaults are unchanged.','',
        'Full-test and retained-cohort outputs occupy separate folders. Each provides 996 task/method rows, '
        '332 configuration/method rows, 28 backbone/method rows, Native-normalized ratios and paired changes. '
        'analysis/audit.json records scalar-score parity, identical paired diagnostic populations, '
        'coverage/event arithmetic checks, mean identities, and independent train/validation reconstruction.']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(all_passed=audit['all_passed'], audited=len(complete), ratios=ratio,
        controlled_wins=wins['DC36ControlledMean'], alpha_zero=summary['alpha_zero']),indent=2))
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--selection-source',type=Path)
    args=parser.parse_args()
    if not aggregate(args.root,args.reference,args.selection_source)['all_passed']:raise SystemExit(1)
