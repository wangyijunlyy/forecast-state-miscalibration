"""Task/seed audit and real three-training-seed mean/sample-std tables."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
from core import seed_statistics,atomic
from deliverables import evidence_audit,render_report
METRICS=['CRPS','QICE','QICE_percent','MSE','MAE','RMSE','MedianMAE']


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def freeze(root,tasks):
    groups={};sources=[]
    for t in tasks:
        p=Path(t['run_root'])/'validation.json'
        if not p.exists():raise ValueError('Validation missing: '+t['id'])
        r=json.loads(p.read_text());assert r['validation_complete'] and r['full_validation_split']
        assert r['task_id']==t['id'] and r['seed']==t['seed']
        key=t['backbone']+'/'+t['dataset'];group=groups.setdefault(key,{'16':False,'64':False})
        for bins in ('16','64'):group[bins]=group[bins] or r['diagnostics'][bins]['triggered']
        sources.append(dict(task_id=t['id'],diagnostics=r['diagnostics']))
    obj=dict(validation_complete_tasks=len(tasks),no_test_metrics_used=True,groups=groups,source_diagnostics=sources,
             decision_unit='backbone,dataset,bin_setting',across_seeds_horizons='any severe constituent',thresholds=dict(R_val=5,A_val=100))
    atomic(root/'frozen_gates.json',obj);return obj


def summarize(root,tasks):
    rows=[];native_rows=[];missing=[];fail=[]
    for t in tasks:
        p=Path(t['run_root'])/'results.json'
        if not p.exists():missing.append(t['id']);continue
        r=json.loads(p.read_text());tr=json.loads((p.parent/'training.json').read_text())
        if not r['full_test_split'] or not tr['training_complete'] or not Path(tr['checkpoint']).is_file():fail.append(t['id']);continue
        for name,score in r['common_normalized'].items():
            row=dict(task_id=t['id'],backbone=t['backbone'],dataset=t['dataset'],horizon=t['horizon'],seed=t['seed'],method=name,
                     **{m:score[m] for m in METRICS},test_windows=r['test_windows'],MAD=False if name=='Native' else r['mad_trigger'][name[2:]])
            rows.append(row)
            ns=r['native_metrics'][name]
            qpercent=ns.get('QICE_percent',ns.get('QICE',0)*(1 if t['backbone']=='PPM' else 100))
            native_rows.append(dict(task_id=t['id'],backbone=t['backbone'],dataset=t['dataset'],horizon=t['horizon'],seed=t['seed'],method=name,
                CRPS=ns['CRPS'],QICE_percent=qpercent,MSE=ns.get('MSE'),MAE=ns.get('MAE'),RMSE=ns.get('RMSE',math.sqrt(ns['MSE']) if 'MSE' in ns else None)))
    write_csv(root/'raw_scores.csv',rows)
    write_csv(root/'native_raw_scores.csv',native_rows)
    grouped=defaultdict(dict)
    for r in rows:grouped[(r['backbone'],r['dataset'],r['horizon'],r['method'])][r['seed']]=r
    stats=[]
    for (b,d,h,m),seeds in grouped.items():
        if set(seeds)!={1,2,3}:continue
        row=dict(backbone=b,dataset=d,horizon=h,method=m,n_seeds=3)
        for metric in METRICS:
            s=seed_statistics({k:v[metric] for k,v in seeds.items()});row[metric+'_mean']=s['mean'];row[metric+'_std']=s['std']
        stats.append(row)
    write_csv(root/'task_mean_std.csv',stats)
    native_grouped=defaultdict(dict)
    for r in native_rows:native_grouped[r['backbone'],r['dataset'],r['horizon'],r['method']][r['seed']]=r
    native_stats=[]
    for (b,d,h,m),seeds in native_grouped.items():
        if set(seeds)!={1,2,3}:continue
        row=dict(backbone=b,dataset=d,horizon=h,method=m,n_seeds=3)
        for metric in ('CRPS','QICE_percent','MSE','MAE','RMSE'):
            if any(seeds[k][metric] is None for k in seeds):
                row[metric+'_mean']=None;row[metric+'_std']=None
            else:
                v=seed_statistics({k:seeds[k][metric] for k in seeds});row[metric+'_mean']=v['mean'];row[metric+'_std']=v['std']
        native_stats.append(row)
    write_csv(root/'native_task_mean_std.csv',native_stats)
    contrasts=[]
    for (b,d,h,method),seeds in grouped.items():
        if method=='Native' or set(seeds)!={1,2,3}:continue
        baseline=grouped[b,d,h,'Native']
        if set(baseline)!={1,2,3}:continue
        row=dict(backbone=b,dataset=d,horizon=h,method=method,n_seeds=3)
        for metric in METRICS:
            changes={k:seeds[k][metric]-baseline[k][metric] for k in seeds}
            v=seed_statistics(changes);row[metric+'_delta_mean']=v['mean'];row[metric+'_delta_std']=v['std']
        gains={k:100*(baseline[k]['CRPS']-seeds[k]['CRPS'])/baseline[k]['CRPS'] for k in seeds}
        v=seed_statistics(gains);row['CRPS_improvement_percent_mean']=v['mean'];row['CRPS_improvement_percent_std']=v['std'];contrasts.append(row)
    write_csv(root/'paired_changes_mean_std.csv',contrasts)
    seed_aggs=defaultdict(list)
    for r in rows:seed_aggs[r['backbone'],r['method'],r['seed']].append(r)
    expected_counts=defaultdict(int)
    for t in tasks:
        if t['seed']==1:expected_counts[t['backbone']]+=1
    agg=[]
    for b,n in expected_counts.items():
        for method in ('Native','DC16','DC64'):
            if any(len(seed_aggs[b,method,s])!=n for s in (1,2,3)):continue
            row=dict(backbone=b,method=method,n_tasks_per_seed=n,n_seeds=3)
            for metric in METRICS:
                means={s:sum(x[metric] for x in seed_aggs[b,method,s])/n for s in (1,2,3)}
                v=seed_statistics(means);row[metric+'_mean']=v['mean'];row[metric+'_std']=v['std']
            agg.append(row)
    write_csv(root/'backbone_mean_std.csv',agg)
    native_seed_aggs=defaultdict(list)
    for r in native_rows:native_seed_aggs[r['backbone'],r['method'],r['seed']].append(r)
    native_agg=[]
    for b,n in expected_counts.items():
        for method in ('Native','DC16','DC64'):
            if any(len(native_seed_aggs[b,method,s])!=n for s in (1,2,3)):continue
            row=dict(backbone=b,method=method,n_tasks_per_seed=n,n_seeds=3)
            for metric in ('CRPS','QICE_percent','MSE','MAE','RMSE'):
                if any(x[metric] is None for s in (1,2,3) for x in native_seed_aggs[b,method,s]):
                    row[metric+'_mean']=None;row[metric+'_std']=None
                else:
                    means={s:sum(x[metric] for x in native_seed_aggs[b,method,s])/n for s in (1,2,3)}
                    v=seed_statistics(means);row[metric+'_mean']=v['mean'];row[metric+'_std']=v['std']
            native_agg.append(row)
    write_csv(root/'native_backbone_mean_std.csv',native_agg)
    audit=dict(expected_tasks=len(tasks),complete_test_tasks=len(rows)//3,missing=missing,invalid=fail,
               expected_method_scores=len(tasks)*3,method_scores=len(rows),task_mean_std_rows=len(stats),backbone_mean_std_rows=len(agg),native_backbone_mean_std_rows=len(native_agg),
               complete=not missing and not fail and len(rows)==711 and len(stats)==237 and len(agg)==18 and len(native_agg)==18)
    verified=evidence_audit(root,tasks)
    audit['requirement_evidence']=verified
    audit['complete']=audit['complete'] and verified['complete']
    atomic(root/'completion_audit.json',audit)
    gate_path=root/'frozen_gates.json'
    gates=json.loads(gate_path.read_text()) if gate_path.exists() else None
    (root/'REPORT.md').write_text(render_report(agg,audit,gates,native_agg))
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    tasks=json.loads((a.root/'manifest.json').read_text());print(freeze(a.root,tasks) if a.freeze else summarize(a.root,tasks))
if __name__=='__main__':main()
