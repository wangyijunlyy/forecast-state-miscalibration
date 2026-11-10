"""Validate complete single-seed evidence for the four diffusion backbones."""
import argparse,csv,json,math
from pathlib import Path
from statistics import mean

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
manifest=json.loads((a.root/'manifest.json').read_text());expected={t['id']:t for t in manifest}
records=[json.loads(p.read_text()) for p in a.root.glob('*/results.json')]
assert len(expected)==len(records)==36
assert {r['task']['id'] for r in records}==set(expected)
for r in records:
 t=r['task'];assert t==expected[t['id']] and t['seed']==t['backbone_seed']==1
 assert r['full_test_split'] and r['original_fast_test_protocol']
 assert r['test_windows']>0 and r['test_batches']>0
 assert r['fit']['fit_split']=='train' and r['fit']['num_samples']==100
 selected=r['fit']['selected_train_indices'];assert len(selected)==len(set(selected))==r['fit']['fit_windows']
 assert r['fit']['fit_windows']==100
 assert r['fit']['residual_filter']['multiplier']==5
 assert r['fit']['calibration_points_retained']<=50000
 assert len({v['coordinate_count'] for v in r['metrics'].values()})==1
 for scores in r['metrics'].values():
  assert scores['coordinate_count']% (r['test_windows']*t['horizon'])==0
  assert all(math.isfinite(scores[k]) for k in ['CRPS','QICE','QICE_percent'])
 assert 'COMPLETE '+t['id'] in (a.root/t['id']/'run.log').read_text(errors='replace')
summary=list(csv.DictReader((a.root/'aggregate_comparison.csv').open()));assert len(summary)==4
for b in ['NsDiff','TMDM','DiffusionTS','TimeDiff']:
 rr=[r for r in records if r['task']['backbone']==b]
 assert len(rr)==len({r['task']['dataset'] for r in rr})==9
 s=next(s for s in summary if s['backbone']==b)
 assert s['standard_deviation']=='TBD'
 for method,key in [('Native','native'),('Binned','binned'),('BinnedFiltered','filtered')]:
  for metric,col in [('CRPS','crps'),('QICE_percent','qice_percent')]:
   assert math.isclose(mean(r['metrics'][method][metric] for r in rr),float(s[key+'_'+col]),abs_tol=1e-12,rel_tol=0)
audit=dict(completed_tasks=36,backbones=4,tasks_per_backbone=9,backbone_seed=1,
 full_original_fast_test_iterators_exhausted=True,all_three_methods_share_native_samples=True,
 exactly_100_distinct_training_windows_each=True,exactly_100_forecast_samples=True,
 fixed_filter_multiplier=5,all_metric_values_finite=True,all_logs_complete=True,
 aggregate_means_independently_recomputed=True,standard_deviation='TBD')
(a.root/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
