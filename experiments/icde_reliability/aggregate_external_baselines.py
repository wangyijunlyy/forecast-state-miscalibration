"""Audit complete-test external comparisons; aggregate only complete run triples."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric,backbone_statistics,configuration_statistics,write_csv
from aggregate_fsdc36 import metric_row,verify_pair
from external_baselines import VERSION,METHODS

METRICS=['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE','state_error_pp',
    'pooled_error_pp','Brier80','Brier90','Brier95','BrierMean']
ALL=['Native','FSPC']+METHODS+['GlobalDC','GlobalPIT','ConditionalPIT']

def aggregate(root):
    tasks=json.loads((root/'manifest.json').read_text());rows=[];done=[];errors=[];resources=[]
    assert len(tasks)==249 and len({t['id'] for t in tasks})==249
    numerical_path=root/'numerical_stability_audit.json'
    numerical=json.loads(numerical_path.read_text()) if numerical_path.exists() else None
    if numerical:assert numerical['all_passed'] and numerical['screened_tasks']==249
    for task in tasks:
        path=root/'runs'/task['id']/'results.json'
        if not path.exists():continue
        try:
            r=json.loads(path.read_text());assert r['version']==VERSION and r['task']==task and r['full_test_split']
            assert set(r['methods'])==set(ALL) and r['fit']['all_parameters_frozen_before_test_access']
            assert not r['backbone_retrained'] and not r['fit']['backbone_retrained']
            assert all(abs(v)<1e-8 for v in r['native_parity'].values())
            if numerical and task['id'] in numerical['affected_tasks']:
                assert r['fit']['methods']['Qian2025SoftCRPS']['numerical_revision']==numerical['revision']
            native=r['methods']['Native'];n=r['fit']['calibration_coordinates']
            assert 0<n<=50000
            assert r['fit']['methods']['QRF2016']['trees']==1000
            assert r['fit']['methods']['QRF2016']['max_samples_leaf'] is None
            assert r['fit']['methods']['QRF2016']['weighted_leaves']
            assert len(r['fit']['methods']['Qian2025SoftCRPS']['candidates'])==20
            for name,score in r['methods'].items():
                verify_pair(score,native);rows.append(metric_row(task,name,score))
            for name in METHODS:
                assert r['fit']['methods'][name]['calibration_coordinates']==n
                resources.append(dict(id=task['id'],backbone=task['backbone'],method=name,
                    fit_seconds=r['fit']['fit_seconds'][name],transform_seconds=r['transform_seconds'][name],
                    model_bytes=r['serialized_model_bytes'][name],peak_rss_kib=r['peak_rss_kib'],
                    calibration_coordinates=n,test_coordinates=native['coordinate_count']))
            done.append(task['id'])
        except Exception as e:errors.append(dict(task=task['id'],error=repr(e)))
    # Never include invalid partially-appended task rows in any summary.
    rows=[r for r in rows if r['id'] in set(done)];resources=[r for r in resources if r['id'] in set(done)]
    out=root/'analysis';out.mkdir(exist_ok=True)
    write_csv(out/'raw_task_metrics.csv',rows);write_csv(out/'resources.csv',resources)
    configs=configuration_statistics(rows,METRICS);stats=backbone_statistics(rows,METRICS)
    write_csv(out/'configuration_statistics.csv',configs);write_csv(out/'backbone_statistics.csv',stats)
    lookup=defaultdict(dict)
    for r in configs:lookup[r['backbone'],r['dataset'],r['horizon']][r['method']]=r
    wins=[]
    for method in ALL:
        if method=='FSPC':continue
        for key,d in lookup.items():
            if method not in d or 'FSPC' not in d:continue
            for metric in ['CRPS','QICE_percent','MSE']:
                a=d['FSPC'][metric+'_mean'];b=d[method][metric+'_mean']
                wins.append(dict(backbone=key[0],dataset=key[1],horizon=key[2],reference=method,metric=metric,
                    FSPC=a,baseline=b,FSPC_wins=a<b,FSPC_change_percent=100*(a/b-1) if b>0 else None))
    write_csv(out/'paired_configuration_comparisons.csv',wins)
    complete=len(done)==len(tasks) and not errors;ratio=[]
    if complete:
        grouped=defaultdict(list)
        for r in rows:grouped[r['backbone'],r['method'],r['seed']].append(r)
        for method in ALL:
            record=dict(method=method,n_backbones=7,n_training_runs=3)
            for metric in METRICS:
                vals=[]
                for seed in [1,2,3]:
                    ratios=[]
                    for backbone in sorted({t['backbone'] for t in tasks}):
                        eligible={key for key,values in lookup.items() if key[0]==backbone and
                            all(values[n][metric+'_mean'] is not None for n in ALL)}
                        a=[r[metric] for r in grouped[backbone,method,seed]
                           if (r['backbone'],r['dataset'],r['horizon']) in eligible]
                        b=[r[metric] for r in grouped[backbone,'Native',seed]
                           if (r['backbone'],r['dataset'],r['horizon']) in eligible]
                        ratios.append(np.mean(a)/np.mean(b) if a and b and np.mean(b)>0 else np.nan)
                    vals.append(np.mean(ratios))
                record[metric+'_mean']=float(np.mean(vals)) if np.isfinite(vals).all() else None
                record[metric+'_std']=float(np.std(vals,ddof=1)) if np.isfinite(vals).all() else None
            ratio.append(record)
        write_csv(out/'native_normalized_ratios.csv',ratio)
    audit=dict(version=VERSION,expected_tasks=len(tasks),audited_tasks=len(done),all_tasks_complete=complete,
        complete_three_run_configurations=len(lookup),missing_tasks=[t['id'] for t in tasks if t['id'] not in done],
        errors=errors,summary_scope=('all 83 configurations and 249 complete native test runs' if complete else 'complete three-run configurations only; partial until all 249 tasks pass'),
        metrics=METRICS,methods=ALL,existing_paper_tables_modified=False)
    if numerical:audit['numerical_stability_audit']='numerical_stability_audit.json';audit['stable_kernel_affected_full_test_evaluations']=numerical['affected_full_test_evaluations']
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# External calibration comparison', '',
        'Complete test tasks: {}/249. Complete three-run configurations: {}/83.'.format(len(done),len(lookup)),
        'Status: '+('complete and audited' if complete else 'in progress; partial results'),'',
        'All four methods use paired native forecasts and the complete original test iterator.',
        'FSPC retains its previously specified training-derived mean bias; external methods fit the shared validation reservoir.',
        'Paper comparisons use the common complete three-run configurations; numeric audit and scope accompany the exported tables.','',
        'Methods: Gaussian EMOS; Qian2025 soft-weighted CRPS PIT; ICML2022 distribution recalibration; QRF2016.',
        'Raw task scores and configuration/backbone statistics are in analysis/*.csv.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['missing_tasks','methods','metrics']},indent=2))
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if aggregate(a.root)['errors']:raise SystemExit(1)
