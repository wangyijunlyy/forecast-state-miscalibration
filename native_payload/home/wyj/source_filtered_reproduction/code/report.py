"""Export and independently recompute real three-seed source-filtered scores."""
import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev
ROOT=Path('/home/wyj/source_filtered_reproduction')
METRICS=['CRPS','QICE','QICE_percent','MSE','MAE','RMSE','MedianMAE']

def save(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize():
    tasks=json.loads((ROOT/'manifest.json').read_text());dest=ROOT/'results';dest.mkdir(exist_ok=True)
    raw=[];native=[];complete=[];invalid=[]
    for t in tasks:
        folder=ROOT/'runs'/t['id'];p=folder/'results.json'
        if not p.exists():continue
        r=json.loads(p.read_text());tr=json.loads((Path(t['run_root'])/'training.json').read_text())
        try:
            assert r['full_test_split'] and tr['training_complete'] and Path(tr['checkpoint']).is_file()
            assert r['test_seed']==t['seed']+30000
            sf=r['source_filter'];assert sf['retained_sources']+sf['excluded_sources']==sf['retained_points']
            assert sf['retained_sources']>0 and sf['source_scale_threshold']==1.01e-6
            assert set(r['common_normalized'])=={'Native','DC16','DC64'}
            if r['evaluation_provenance'].startswith('exact unchanged'):
                assert sf['excluded_sources']==0 and all(sf['original_raw_pool_exact_parity'].values())
            assert len({v['coordinate_count'] for v in r['common_normalized'].values()})==1
            for m,s in r['common_normalized'].items():
                assert all(math.isfinite(s[k]) for k in METRICS)
                raw.append(dict(task_id=t['id'],backbone=t['backbone'],dataset=t['dataset'],horizon=t['horizon'],seed=t['seed'],method=m,**{k:s[k] for k in METRICS},test_windows=r['test_windows'],source_filter=(m!='Native'),excluded_fitting_sources=sf['excluded_sources']))
                ns=r['native_metrics'][m];q=ns.get('QICE_percent',ns.get('QICE',0)*(1 if t['backbone']=='PPM' else 100))
                native.append(dict(task_id=t['id'],backbone=t['backbone'],dataset=t['dataset'],horizon=t['horizon'],seed=t['seed'],method=m,CRPS=ns['CRPS'],QICE_percent=q,MSE=ns.get('MSE'),MAE=ns.get('MAE'),RMSE=ns.get('RMSE',math.sqrt(ns['MSE']) if 'MSE' in ns else None)))
            complete.append(t['id'])
        except Exception as e:invalid.append(dict(task=t['id'],error=repr(e)))
    cells=0
    for prefix,rows in [('',raw),('native_',native)]:
        save(dest/(prefix+'raw_scores.csv'),rows)
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in ['backbone','dataset','horizon','method'])].append(r)
        metrics=METRICS if not prefix else ['CRPS','QICE_percent','MSE','MAE','RMSE']
        stats=[]
        for key,sub in groups.items():
            if {r['seed'] for r in sub}!={1,2,3}:continue
            out=dict(zip(['backbone','dataset','horizon','method'],key));out['n_seeds']=3
            for m in metrics:
                if any(r[m] is None for r in sub):out[m+'_mean']=None;out[m+'_std']=None;continue
                a=[r[m] for r in sub];out[m+'_mean']=mean(a);out[m+'_std']=stdev(a)
                assert math.isclose(stdev(a),math.sqrt(sum((v-mean(a))**2 for v in a)/2),rel_tol=1e-12,abs_tol=1e-14);cells+=2
            stats.append(out)
        save(dest/(prefix+'task_mean_std.csv'),stats)
        agg=[]
        for b in sorted({t['backbone'] for t in tasks}):
            expected=sum(t['backbone']==b and t['seed']==1 for t in tasks)
            for method in ['Native','DC16','DC64']:
                sub=[r for r in rows if r['backbone']==b and r['method']==method]
                if any(sum(r['seed']==s for r in sub)!=expected for s in (1,2,3)):continue
                out=dict(backbone=b,method=method,n_tasks_per_seed=expected,n_seeds=3)
                for m in metrics:
                    if any(r[m] is None for r in sub):out[m+'_mean']=None;out[m+'_std']=None;continue
                    a=[mean(r[m] for r in sub if r['seed']==s) for s in (1,2,3)]
                    out[m+'_mean']=mean(a);out[m+'_std']=stdev(a);cells+=2
                agg.append(out)
        save(dest/(prefix+'backbone_mean_std.csv'),agg)
    paired=[]
    groups=defaultdict(dict)
    for r in raw:groups[r['backbone'],r['dataset'],r['horizon'],r['seed']][r['method']]=r
    contrasts=defaultdict(list)
    for (b,d,h,s),ms in groups.items():
        for m in ['DC16','DC64']:
            values={k:ms[m][k]-ms['Native'][k] for k in METRICS}
            values['CRPS_improvement_percent']=100*(1-ms[m]['CRPS']/ms['Native']['CRPS'])
            contrasts[b,d,h,m].append(values)
    for key,sub in contrasts.items():
        if len(sub)!=3:continue
        row=dict(zip(['backbone','dataset','horizon','method'],key));row['n_seeds']=3
        for metric in METRICS:
            a=[r[metric] for r in sub];row[metric+'_delta_mean']=mean(a);row[metric+'_delta_std']=stdev(a)
        a=[r['CRPS_improvement_percent'] for r in sub];row['CRPS_improvement_percent_mean']=mean(a);row['CRPS_improvement_percent_std']=stdev(a)
        paired.append(row)
    save(dest/'paired_changes_mean_std.csv',paired)
    audit=dict(complete=len(complete)==237 and not invalid,expected_tasks=237,complete_test_tasks=len(complete),method_scores=len(raw),invalid=invalid,missing=[t['id'] for t in tasks if t['id'] not in complete],source_filter_threshold=1.01e-6)
    (dest/'completion_audit.json').write_text(json.dumps(audit,indent=2))
    (dest/'final_independent_audit.json').write_text(json.dumps(dict(passed=audit['complete'],checked_statistic_cells=cells,real_training_seeds=[1,2,3],sample_std_ddof=1),indent=2))
    (dest/'manifest.json').write_text(json.dumps(tasks,indent=2))
    print('REPORT',len(complete),'/237',len(invalid),'invalid',flush=True)
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        report=summarize()
        if report['complete'] or not a.watch:break
        status=json.loads((ROOT/'status.json').read_text())
        if not status['pending'] and not status['active']:break
        time.sleep(30)
