"""Audit complete-test Gaussian controls and their parity with primary forecasts."""
import argparse
import json
from pathlib import Path
import numpy as np
from aggregation import audit_metric, close, flatten, backbone_statistics, configuration_statistics, write_csv, METRICS

VERSION = 'full-gaussian-controls-v1-20261102'
METHODS = {'Native','GaussianVariance2P','GaussianAffine4P'}


def audit(path, primary, expected):
    value = json.loads(path.read_text()); task = value['task']; identity = task['id']
    if identity not in expected or path.parent.name != identity or value['version'] != VERSION:
        raise ValueError('Gaussian control identity/version mismatch')
    for key in ['backbone','dataset','horizon','seed']:
        if task[key] != expected[identity][key]:
            raise ValueError('Manifest mismatch')
    original = json.loads((primary/identity/'results.json').read_text())
    if not value['full_test_split'] or set(value['methods']) != METHODS:
        raise ValueError('Incomplete Gaussian methods or evaluation')
    for key in ['test_windows','test_batches','test_seed']:
        if value[key] != original[key]:
            raise ValueError('Gaussian test population/seed differs')
    fit = value['fitting']
    if fit['fitting_split'] != 'validation' or fit['calibration_coordinates'] != original['validation']['retained_calibration_coordinates']:
        raise ValueError('Fitting population differs')
    for name in METHODS-{'Native'}:
        model = fit['models'][name]
        if not model['optimizer_success'] or not np.isfinite(model['fit_analytic_gaussian_crps']):
            raise ValueError('Unsuccessful Gaussian fit')
    native = original['methods']['Native']; rows = []
    for method,metric in value['methods'].items():
        audit_metric(metric)
        if metric['coordinate_count'] != native['coordinate_count'] or metric['state_coverage']['counts'] != native['state_coverage']['counts']:
            raise ValueError('Different Gaussian comparison population')
        close(metric['events']['observed_frequencies'],native['events']['observed_frequencies'],'Different events')
        if method == 'Native':
            for key in ['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE']:
                close(metric[key],native[key],'Native forecast parity failed')
            continue
        row={k:task[k] for k in ['id','backbone','dataset','horizon','seed']}; state=metric['state_coverage']
        row.update(method=method,coordinate_count=metric['coordinate_count'],eligible_coordinates=state['eligible_coordinates'],
            state_error_pp=None if state['state_error'] is None else 100*state['state_error'],
            pooled_error_pp=None if state['pooled_error'] is None else 100*state['pooled_error'])
        row.update({k:metric[k] for k in ['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE']})
        row.update(zip(['Brier80','Brier90','Brier95'],metric['events']['brier']))
        row['BrierMean']=float(np.mean(metric['events']['brier']));rows.append(row)
    return identity, rows


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ['runs','primary-runs','manifest','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();expected={t['id']:t for t in json.loads(args.manifest.read_text())}
    rows=[];completed=[];errors=[]
    for path in sorted(args.runs.glob('*/results.json')):
        try:
            identity,values=audit(path,args.primary_runs,expected);completed.append(identity);rows.extend(values)
        except Exception as error:
            errors.append(dict(path=str(path),error=str(error)))
    args.output.mkdir(parents=True,exist_ok=True);keys=[k for k in METRICS if k != 'transform_seconds']
    write_csv(args.output/'raw_task_metrics.csv',rows)
    write_csv(args.output/'configuration_statistics.csv',configuration_statistics(rows,keys))
    write_csv(args.output/'backbone_statistics.csv',backbone_statistics(rows,keys))
    missing=sorted(set(expected)-set(completed));report=dict(expected_tasks=len(expected),audited_tasks=len(completed),
        all_tasks_complete=not missing and not errors,missing_tasks=missing,errors=errors,
        evaluation='complete native test iterators; Native scores independently checked against primary suite')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='missing_tasks'},indent=2))
    if errors:
        raise SystemExit(1)
