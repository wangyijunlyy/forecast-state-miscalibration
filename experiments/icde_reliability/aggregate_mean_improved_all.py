"""Paired seed-aware summaries with explicit incomplete coverage."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from mean_improved_all import METHODS,valid_result,coverage

METRICS=['MSE','CRPS','QICE_percent','StateError_percent','BrierMean']


def metrics(v):
    return dict(MSE=v['MSE'],CRPS=v['CRPS'],QICE_percent=v['QICE_percent'],
        StateError_percent=None if v['state_coverage']['state_error'] is None else 100*v['state_coverage']['state_error'],
        BrierMean=float(np.mean(v['events']['brier'])))


def summarize_records(tasks,records):
    states={t['id']:valid_result(records[t['id']],t) for t in tasks if t['id'] in records}
    raw=[];groups=defaultdict(list);expected=defaultdict(set)
    for t in tasks:
        identity=(t['backbone'],t['dataset'],t['horizon']);expected[identity].add(t['seed'])
        if states.get(t['id']) is not True:continue
        r=records[t['id']]
        for m in METHODS:
            row={k:t[k] for k in ['id','backbone','dataset','horizon','seed']}
            row.update(method=m,**metrics(r['methods'][m]));raw.append(row)
            groups[identity+(m,)].append(row)
    complete_ids=[key for key,seeds in expected.items() if all({r['seed'] for r in groups[key+(m,)]}==seeds for m in METHODS)]
    configurations=[];backbones=[];wins={k:dict(improved=0,worsened=0,tied=0,unavailable=0) for k in METRICS}
    for identity in sorted(complete_ids):
        records_by_method={}
        for m in METHODS:
            rows=groups[identity+(m,)];record=dict(zip(['backbone','dataset','horizon'],identity),method=m,runs=len(rows))
            for metric in METRICS:
                values=[r[metric] for r in rows]
                record[metric+'_mean']=None if any(v is None for v in values) else float(np.mean(values))
                record[metric+'_std']=None if any(v is None for v in values) else float(np.std(values,ddof=1))
            configurations.append(record);records_by_method[m]=record
        for metric in METRICS:
            a=records_by_method['FSPCRefit'][metric+'_mean'];b=records_by_method['FSPCMean'][metric+'_mean']
            if a is None or b is None:wins[metric]['unavailable']+=1;continue
            delta=b-a;wins[metric]['improved' if delta < -1e-10 else 'worsened' if delta>1e-10 else 'tied']+=1
    for name in sorted({t['backbone'] for t in tasks}):
        identities=[i for i in complete_ids if i[0]==name]
        if not identities:continue
        n_expected=sum(i[0]==name for i in expected)
        seeds=sorted({t['seed'] for t in tasks if t['backbone']==name})
        for m in METHODS:
            record=dict(backbone=name,method=m,completed_configurations=len(identities),expected_configurations=n_expected,
                status='complete' if len(identities)==n_expected else 'provisional',runs=len(seeds))
            for metric in METRICS:
                values=[]
                for seed in seeds:
                    v=[next(r for r in groups[i+(m,)] if r['seed']==seed)[metric] for i in identities]
                    values.append(None if any(x is None for x in v) else float(np.mean(v)))
                record[metric+'_mean']=None if any(v is None for v in values) else float(np.mean(values))
                record[metric+'_std']=None if any(v is None for v in values) else float(np.std(values,ddof=1))
            backbones.append(record)
    return dict(coverage=coverage(tasks,states),raw=raw,configurations=configurations,backbones=backbones,
        completed_configurations=len(complete_ids),expected_configurations=len(expected),configuration_wins=wins)


def write_csv(path,rows):
    if not rows:
        path.write_text('');return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate(root):
    root=Path(root);tasks=json.loads((root/'manifest.json').read_text());records={};parse_errors={}
    for t in tasks:
        p=root/'runs'/t['id']/'results.json'
        if p.exists():
            try:records[t['id']]=json.loads(p.read_text())
            except (ValueError,OSError) as error:parse_errors[t['id']]=str(error)
    result=summarize_records(tasks,records);out=root/'analysis';out.mkdir(exist_ok=True)
    for key,file in [('raw','raw_metrics.csv'),('configurations','configuration_statistics.csv'),('backbones','backbone_statistics.csv')]:write_csv(out/file,result[key])
    cov=result['coverage'];missing=set(cov['missing']);invalid=set(cov['invalid'])
    write_csv(out/'coverage.csv',[dict(task=t['id'],backbone=t['backbone'],dataset=t['dataset'],horizon=t['horizon'],seed=t['seed'],
        status='invalid' if t['id'] in invalid else 'missing' if t['id'] in missing else 'complete') for t in tasks])
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    summary={k:v for k,v in result.items() if k not in ['raw','configurations','backbones']}
    summary.update(status='complete' if cov['complete']==len(tasks) else 'provisional',
        parse_errors=parse_errors,controller_failures=status.get('failures',[]),
        comparison='improved mean versus original FSPC; DiffusionTS both use corrected unclipped sampling',
        backbone_retrained=False,summary_unit='complete configurations averaged within each seed; sample SD across seeds')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# Mean-only FSPC across frozen backbones','',
        f"Full-test paired results: {cov['complete']}/{len(tasks)} runs; {result['completed_configurations']}/{result['expected_configurations']} configurations have all three runs.",
        'No backbone is retrained. Completed DiffusionTS unclipped results are reused. All other tasks match the original Native and FSPC full-test scores before reporting the new mean.',
        'The mean is a global affine fit plus state residual sum/(count+128). Blend strength is selected by validation CRPS subject to nonincreasing validation MSE; it is frozen before refitting on all calibration coordinates. This is a validation constraint, not a test guarantee.',
        'Chronological fitting/selection purges overlapping target windows. Where the existing validation population cannot support independent blocks, historical training forecasts fit and validation selects; each fit.json labels that fallback.',
        'Backbone summaries below include only configurations with all three paired runs. Provisional backbone coverage is shown explicitly. Values average configurations within each run, then show mean and sample SD across the three runs.','']
    labels=['MSE ↓','CRPS ↓','QICE (pp) ↓','State error (pp) ↓','Brier ↓']
    for name in sorted({r['backbone'] for r in result['backbones']}):
        rows=[r for r in result['backbones'] if r['backbone']==name];first=rows[0]
        report.extend([f"## {name} ({first['completed_configurations']}/{first['expected_configurations']} configurations; {first['status']})",'',
            '| Method | '+' | '.join(labels)+' |','|---|'+ '|'.join(['---:']*5)+'|'])
        for row in rows:
            cells=[]
            for metric in METRICS:
                value=row[metric+'_mean'];std=row[metric+'_std']
                if value is None:cells.append('N/A');continue
                best=min(r[metric+'_mean'] for r in rows if r[metric+'_mean'] is not None)
                mean=f'{value:.6f}';mean='**'+mean+'**' if value==best else mean
                cells.append(mean+f'<sub>±{std:.6f}</sub>')
            report.append('| '+row['method']+' | '+' | '.join(cells)+' |')
        report.append('')
    report.extend(['## Paired configuration changes','',
        'All available complete configurations; lower is better. These counts remain provisional until the entire manifest completes.','',
        '| Metric | Improved | Worsened | Tied | Unavailable |','|---|---:|---:|---:|---:|'])
    for metric,value in result['configuration_wins'].items():report.append('| '+metric+' | '+' | '.join(str(value[k]) for k in ['improved','worsened','tied','unavailable'])+' |')
    report.extend(['','Per-configuration means and SDs: analysis/configuration_statistics.csv. All paired task scores: analysis/raw_metrics.csv. Missing and failed tasks: analysis/coverage.csv and status.json. Manuscript remains unchanged.'])
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(complete=cov['complete'],expected=len(tasks),complete_configurations=result['completed_configurations'])))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);aggregate(p.parse_args().root)
