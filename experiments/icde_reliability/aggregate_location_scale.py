"""Aggregate complete, paired full-test experiments; no result selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from location_scale import VERSION,complete_result


def aggregate(root):
    root=Path(root);records=[];groups={};coverage=[]
    jobs=json.loads((root/'manifest.json').read_text())
    for job in jobs:
        p=root/job['mode']/'runs'/job['task']['id']/'results.json'
        r=json.loads(p.read_text()) if p.exists() else None
        complete=r is not None and complete_result(r,job['task'],job['mode'],job['external'])
        coverage.append(dict(mode=job['mode'],task=job['task']['id'],dataset=job['task']['dataset'],seed=job['task']['seed'],
            complete=complete,reason='complete' if complete else ('missing' if r is None else 'protocol incomplete')))
        if not complete:continue
        if r['mode']=='original':assert r['native_parity']=={'Native':True,'FSPCRefit':True}
        task=r['task']
        for method,v in r['methods'].items():
            metrics={'MSE':v['MSE'],'CRPS':v['CRPS'],'QICE_percent':v['QICE_percent'],
                'StateError_percent':None if v['state_coverage']['state_error'] is None else 100*v['state_coverage']['state_error'],
                'BrierMean':float(np.mean(v['events']['brier']))}
            for metric,value in metrics.items():
                if value is None:continue
                records.append(dict(mode=r['mode'],task=task['id'],dataset=task['dataset'],seed=task['seed'],method=method,metric=metric,value=value))
                groups.setdefault((r['mode'],task['dataset'],method,metric),[]).append(value)
    out=root/'analysis';out.mkdir(exist_ok=True)
    if records:
        with (out/'raw_metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        with (out/'dataset_statistics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['mode','dataset','method','metric','runs','mean','std']);writer.writeheader()
            for key,values in sorted(groups.items()):
                writer.writerow(dict(zip(['mode','dataset','method','metric'],key),runs=len(values),mean=np.mean(values),std=np.std(values,ddof=1) if len(values)>1 else 0.))
    with (out/'coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['mode','task','dataset','seed','complete','reason']);writer.writeheader();writer.writerows(coverage)
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    complete_count=sum(c['complete'] for c in coverage)
    by_task={}
    for c in coverage:by_task.setdefault(c['task'],[]).append(c)
    summary=dict(full_test_records=complete_count,metric_records=len(records),expected_jobs=len(jobs),
        missing_jobs=len(jobs)-complete_count,comparison_status='complete' if complete_count==len(jobs) else 'provisional',
        missing=[c for c in coverage if not c['complete']],controller_failure_history=status.get('failures',[]),
        paired_sampling_tasks=[k for k,v in by_task.items() if len(v)>1 and all(c['complete'] for c in v)])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);aggregate(p.parse_args().root)
