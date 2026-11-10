"""Independent source audits and complete Train FSPC comparisons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric, configuration_statistics, backbone_statistics, write_csv
from aggregate_fsdc36 import metric_row
from aggregate_mean_anchor_all import normalized_ratios
from mean_anchor_campaign import validate_tasks
from portable_dc import CompiledCalibrator
from train_fspc import VERSION, METHODS, validate_record

METRICS = ['MSE','MAE','RMSE','MedianMAE','CRPS','QICE_percent','state_error_pp',
           'pooled_error_pp','Brier80','Brier90','Brier95','BrierMean']
KEYS = ['MSE','CRPS','QICE_percent','BrierMean']
LABELS = dict(Native='Native', TrainDC36='Train FSDC', DC36='FSDC',
              DC36ControlledMean='FSPC', TrainFSPC='Train FSPC')


def same(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)


def template_record(cal):
    return dict({key:getattr(cal,key).tolist() for key in
            ['mean_edges','scale_edges','quantiles','state_to_template']},
            horizon_length=int(cal.horizon_length),n_horizon_bins=int(cal.n_horizon_bins))


def audit_training_sources(task, fit, cal, source):
    """Reconstruct pools, raw biases and validation optimum without the fitter."""
    with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as data:
        rows, targets = data['rows'].astype(float), data['targets'].astype(float)
    mu = rows.mean(1); raw = rows.std(1, ddof=1); sigma = raw.clip(1e-6)
    expected_edges = []
    for values in [mu, np.log(sigma)]:
        boundary = np.quantile(values, np.linspace(0,1,7))
        boundary[0] = -np.inf; boundary[-1] = np.inf; expected_edges.append(boundary)
    np.testing.assert_array_equal(cal.mean_edges, expected_edges[0])
    np.testing.assert_array_equal(cal.scale_edges, expected_edges[1])
    assert cal.horizon_length == task['horizon'] and cal.n_horizon_bins == 1
    def groups(means, scales):
        return 6*np.searchsorted(expected_edges[0][1:-1],means,side='right') + \
            np.searchsorted(expected_edges[1][1:-1],np.log(scales),side='right')
    ids = groups(mu,sigma); valid = raw > 1.01e-6; residual = (targets-mu)/sigma
    def bounded(values):
        values = np.sort(values)
        return values if len(values)<=20000 else values[np.linspace(0,len(values)-1,20000).astype(int)]
    pooled = bounded(residual[valid]); counts = np.array([(ids==g).sum() for g in range(36)])
    bias = np.array([(targets[ids==g]-mu[ids==g]).sum()/(counts[g]+128)
                     if counts[g]>=128 else 0. for g in range(36)])
    np.testing.assert_array_equal(counts,fit['training_state_counts']); same(bias,fit['state_bias'])
    assert fit['training_coordinates']==len(targets)
    assert fit['retained_sources']==int(valid.sum()) and fit['excluded_sources']==int((~valid).sum())
    local = 0
    for g in range(36):
        values = residual[(ids==g)&valid]
        pool = bounded(values) if len(values)>=128 else pooled
        local += int(len(values)>=128)
        same(cal.quantiles[cal.state_to_template[g]],np.quantile(pool,(np.arange(100)+.5)/100))
    assert fit['local_pools']==local
    del rows, targets, residual
    with np.load(source/'runs'/task['id']/'calibration.npz') as data:
        rows, targets = data['rows'].astype(float), data['targets'].astype(float)
    mu = rows.mean(1); sigma = rows.std(1,ddof=1).clip(1e-6)
    shift = bias[groups(mu,sigma)]; error = targets-mu
    numerator = float(np.sum(error*shift)); denominator = float(np.sum(shift**2))
    alpha = float(np.clip(numerator/denominator,0,1)) if denominator>0 else 0.
    selection = fit['selection']
    same(alpha,selection['strength']); same(numerator,selection['numerator'])
    same(denominator,selection['denominator']); same(np.mean(error**2),selection['native_mse'])
    same(np.mean((error-alpha*shift)**2),selection['selected_mse'])
    assert selection['coordinates']==len(targets)


def aggregate(root, source, mean_root, selection_source=None):
    tasks = json.loads((root/'manifest.json').read_text())
    scope = validate_tasks(tasks,json.loads((source/'manifest.json').read_text()))
    validate_tasks(tasks,json.loads((mean_root/'manifest.json').read_text()))
    out = root/'analysis'; out.mkdir(exist_ok=True)
    rows = []; cohort_rows = []; errors = []; completed = []; fits = {}; templates = {}; selections = []
    for task in tasks:
        identity = task['id']; folder = root/'runs'/identity
        if not (folder/'results.json').exists(): continue
        try:
            value = json.loads((folder/'results.json').read_text()); validate_record(value,task)
            cohort = json.loads((folder/'cohort.json').read_text())
            fit = json.loads((folder/'fit.json').read_text())
            assert fit==value['fit']==cohort['fit'] and cohort['task']==task and cohort['version']==VERSION
            assert set(cohort['methods'])==set(METHODS)
            assert cohort['diagnostic_partition']=='original validation-fitted 6x6 cohort boundaries'
            assert cohort['newly_evaluated_methods']==['TrainDC36','TrainFSPC']
            original = json.loads((source/'runs'/identity/'results.json').read_text())
            mean = json.loads((mean_root/'runs'/identity/'results.json').read_text())
            original_cohort = json.loads((source/'cohort/runs'/identity/'results.json').read_text())
            mean_cohort = json.loads((mean_root/'runs'/identity/'cohort.json').read_text())
            assert original['task']==mean['task']==original_cohort['task']==mean_cohort['task']==task
            for name in ['Native','TrainDC36','DC36']:
                assert value['methods'][name]==original['methods'][name]
            assert value['methods']['DC36ControlledMean']==mean['methods']['DC36ControlledMean']
            assert cohort['methods']['Native']==original_cohort['native']
            assert cohort['methods']['DC36']==original_cohort['methods']['b6_h1_n50000']
            assert cohort['methods']['DC36ControlledMean']==mean_cohort['methods']['DC36ControlledMean']
            assert value['test_batches']==original['test_batches'] and value['test_windows']==original['test_windows']
            assert cohort['coordinates']==original_cohort['coordinates']
            assert value['maximum_mean_deviation']<1e-8
            cal = CompiledCalibrator.load(folder/'compiled.npz')
            with np.load(folder/'mean_parameters.npz') as data:
                same(data['bias'],fit['state_bias']); same(data['counts'],fit['training_state_counts'])
                same(data['strength'],fit['selection']['strength'])
            if selection_source is not None:
                audit_training_sources(task,fit,cal,selection_source)
            task_rows = []; task_cohorts = []
            for record,target in [(value,task_rows),(cohort,task_cohorts)]:
                native = record['methods']['Native']
                for name,metric in record['methods'].items():
                    audit_metric(metric)
                    assert metric['coordinate_count']==native['coordinate_count']
                    np.testing.assert_array_equal(metric['state_coverage']['counts'],native['state_coverage']['counts'])
                    same(metric['events']['observed_frequencies'],native['events']['observed_frequencies'])
                    target.append(metric_row(task,name,metric))
                if fit['selection']['strength']==0:
                    for key in ['MSE','MAE','RMSE']:
                        same(record['methods']['TrainFSPC'][key],native[key])
            assert fit['selection']['selected_mse']<=fit['selection']['native_mse']+1e-10
            selections.append(dict(id=identity,**{k:task[k] for k in ['backbone','dataset','horizon','seed']},
                strength=fit['selection']['strength'],native_validation_mse=fit['selection']['native_mse'],
                selected_validation_mse=fit['selection']['selected_mse'],
                maximum_mean_deviation=value['maximum_mean_deviation']))
            rows.extend(task_rows); cohort_rows.extend(task_cohorts)
            completed.append(identity); fits[identity]=fit; templates[identity]=template_record(cal)
        except Exception as error:
            errors.append(dict(id=identity,error=repr(error)))
    missing = sorted({t['id'] for t in tasks}-set(completed))
    complete = len(completed)==249 and not errors
    proof_path = out/'source_selection_audit.json'
    if complete and selection_source is not None:
        proof_path.write_text(json.dumps(dict(all_passed=True,tasks=249,manifest=tasks,fits=fits,
            templates=templates,raw_training_templates_bias_and_validation_strength_reconstructed=True),indent=2)+'\n')
    proof = json.loads(proof_path.read_text()) if proof_path.exists() else {}
    raw_sources_audited = (proof.get('all_passed') is True and proof.get('tasks')==249
        and proof.get('manifest')==tasks and proof.get('fits')==fits and proof.get('templates')==templates)
    audit = dict(all_passed=bool(complete and raw_sources_audited),**scope,audited_tasks=len(completed),
        missing_tasks=missing,errors=errors,methods=METHODS,
        reused_full_test_baseline_tasks=len(completed),baseline_metrics_recomputed=False,
        zero_strength_native_point_score_parity=True,
        full_test_and_cohort_populations_audited_separately=True,
        independent_raw_training_validation_audit=raw_sources_audited,neural_training_performed=False)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    write_csv(out/'mean_selection.csv',selections)
    if not audit['all_passed']: return audit
    all_summaries = {}
    for name,data in [('full_test',rows),('retained_cohort',cohort_rows)]:
        folder = out/name; folder.mkdir(exist_ok=True)
        configurations = configuration_statistics(data,METRICS)
        backbones = backbone_statistics(data,METRICS)
        ratios, paired = normalized_ratios(data,METRICS)
        for filename,records in [('raw_task_metrics',data),('configuration_statistics',configurations),
            ('backbone_statistics',backbones),('native_ratios',ratios),('backbone_seed_native_ratios',paired)]:
            write_csv(folder/(filename+'.csv'),records)
        by = {(r['backbone'],r['dataset'],r['horizon'],r['method']):r for r in configurations}
        identities = sorted({key[:3] for key in by})
        wins = {}; changes = []
        for reference in METHODS[:-1]:
            wins[reference] = {}
            for metric in METRICS:
                differences = [by[i+('TrainFSPC',)][metric+'_mean']-by[i+(reference,)][metric+'_mean']
                    for i in identities if by[i+('TrainFSPC',)][metric+'_mean'] is not None
                    and by[i+(reference,)][metric+'_mean'] is not None]
                delta = np.array(differences)
                wins[reference][metric] = dict(improved=int((delta < -1e-10).sum()),
                    worsened=int((delta > 1e-10).sum()),tied=int((abs(delta)<=1e-10).sum()),eligible=len(delta))
            for identity in identities:
                candidate = by[identity+('TrainFSPC',)]; baseline = by[identity+(reference,)]
                row = dict(backbone=identity[0],dataset=identity[1],horizon=identity[2],reference=reference)
                for metric in METRICS:
                    a,b = candidate[metric+'_mean'],baseline[metric+'_mean']
                    row[metric+'_change_percent'] = 100*(a/b-1) if a is not None and b is not None and b>0 else None
                changes.append(row)
        write_csv(folder/'configuration_changes.csv',changes)
        (folder/'wins.json').write_text(json.dumps(wins,indent=2)+'\n')
        all_summaries[name] = dict(ratios=ratios,wins=wins)
    (out/'summary.json').write_text(json.dumps(all_summaries,indent=2)+'\n')
    report = ['# Train FSPC complete results','',
        'Seven backbones, 83 configurations, three seeds, 249 complete native test evaluations. '
        'Train FSPC uses training-fitted templates and raw mean bias; validation selects the unchanged strength. '
        'All existing full-test comparison scores are reused. No neural network was retrained.','',
        'Ratios divide each backbone/seed configuration mean by paired Native, equally average seven backbones, '
        'then report mean ± sample SD across three seeds. All metrics are minimized.','']
    for name,summary in all_summaries.items():
        report += ['## '+name.replace('_',' '),'',
            '| Method | MSE / Native | CRPS / Native | QICE / Native | Brier / Native |',
            '|---|---:|---:|---:|---:|']
        by_method = {r['method']:r for r in summary['ratios']}
        for method in METHODS:
            row = by_method[method]
            report.append('| '+LABELS[method]+' | '+' | '.join(
                f"{row[k+'_mean']:.6f} ± {row[k+'_std']:.6f}" for k in KEYS)+' |')
        report += ['', '| Reference | Metric | Improved | Worsened | Tied |','|---|---|---:|---:|---:|']
        for reference in METHODS[:-1]:
            for metric in KEYS:
                w = summary['wins'][reference][metric]
                report.append(f"| {LABELS[reference]} | {metric} | {w['improved']} | {w['worsened']} | {w['tied']} |")
        report += ['']
    report += ['All twelve metrics, per-configuration means/sample SDs, backbone summaries, paired ratios and changes '
        'are available in the separate full_test and retained_cohort analysis folders. '
        'Undefined coverage metrics remain null. The source audit binds the exact manifest, fitted parameters '
        'and compiled templates for record-only local reconstruction.','']
    (root/'REPORT.md').write_text('\n'.join(report))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root','source','mean-root','selection-source']:
        parser.add_argument('--'+name,type=Path,required=name!='selection-source')
    args = parser.parse_args()
    audit = aggregate(args.root,args.source,args.mean_root,args.selection_source)
    print(json.dumps(audit,indent=2))
    if not audit['all_passed']: raise SystemExit(1)
