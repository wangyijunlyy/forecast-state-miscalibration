"""Recompute every FSPC component study on a shared retained test cohort."""
import argparse,fcntl,gzip,pickle,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from run_mean_improved_all import atomic,load
from fspc_unified import UnifiedMean,fit_unified,VERSION
from fspc_granularity import VARIANTS
from granularity import fit_partition,method_name
from location_scale import replay_windows
from portable_dc import summaries,edges
from mean_anchor import state_ids as calibration_states
from reliability_metrics import ReliabilityMetrics


def score(q,y,c,g,thresholds,ordered=True):
 acc=ReliabilityMetrics(thresholds[c],n_states=36)
 acc.update(q.T[None,:,None,:],y[None,None,:],g,ordered=ordered)
 return acc.result()


def run(task,root,source,mean_root,prior):
 out=root/'components/runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
 lock=(out/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 if (out/'results.json').exists():
  assert load(out/'results.json')['version']==VERSION;return task['id']
 start=time.monotonic();is_diff=task['backbone']=='DiffusionTS'
 new=load(mean_root/'runs'/task['id']/'results.json');provenance=new['provenance']
 val_path=(prior/'unclipped/runs'/task['id']/'val_calibration.npz') if is_diff else source/'runs'/task['id']/'calibration.npz'
 train_path=(prior/'unclipped/runs'/task['id']/'train_calibration.npz') if is_diff else source/'train_calibration/runs'/task['id']/'calibration.npz'
 with np.load(val_path) as d:
  val,vy,vh,vc=[d[k] for k in ['rows','targets','horizon','channel']]
  if 'window' in d:window=d['window']
  else:window,hh,cc=replay_windows(provenance['calibration_windows'],task['horizon'],int(vc.max())+1,50000,task['seed']+40000,provenance['batch_sizes']);np.testing.assert_array_equal(hh,vh);np.testing.assert_array_equal(cc,vc)
 with np.load(train_path) as d:train,ty,th=[d[k] for k in ['rows','targets','horizon']]
 with np.load(source/'runs'/task['id']/'calibration.npz') as d:thresholds=d['thresholds']
 model_path=(prior/'unclipped/runs'/task['id']/'models.pkl.gz') if is_diff else mean_root/'runs'/task['id']/'models.pkl.gz'
 with gzip.open(model_path,'rb') as f:model=UnifiedMean.from_existing(pickle.load(f)['FSPCMean'])
 atomic(out/'parameters.json',model.to_record())
 fitted={};resources={}
 for b,g,budget in VARIANTS:
  key=method_name(b,g,budget)
  fitted[key],resources[key]=fit_unified(train,ty,th,val,vy,vh,window,provenance['calibration_windows'],provenance['origin_stride'],task['horizon'],bins=b,horizon_groups=g,budget=budget,ordering_seed=task['seed']+60000)
  atomic(out/'progress.json',dict(stage='fit',variants=len(fitted),expected=len(VARIANTS),elapsed=time.monotonic()-start))
 train_model,train_fit=fit_unified(train,ty,th,val,vy,vh,window,provenance['calibration_windows'],provenance['origin_stride'],task['horizon'],template_split='train')
 unfiltered,unfiltered_fit=fit_partition(val,vy,vh,task['horizon'],6,1,source_filter=False);unfiltered=unfiltered.compile()
 atomic(out/'frozen.json',dict(version=VERSION,task=task,variants=resources,training_template=train_fit,parameters_frozen_before_test=True))
 # No test targets are accessed before every fitted component is frozen.
 test_path=(root/'diffusion/runs'/task['id']/'test_reservoir.npz') if is_diff else source/'runs'/task['id']/'test_reservoir.npz'
 with np.load(test_path) as d:rows,y,h,c=[d[k] for k in ['rows','targets','horizon','channel']]
 with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:original_train=d['rows']
 mu,_,sig=summaries(original_train);me=edges(mu,6);se=edges(np.log(sig),6)
 tm,_,ts=summaries(rows);groups=6*np.searchsorted(me[1:-1],tm,side='right')+np.searchsorted(se[1:-1],np.log(ts),side='right')
 native=score(rows,y,c,groups,thresholds,False);default_q=model.predict(rows,h)
 control_scores={'FSPCMean':score(default_q,y,c,groups,thresholds),'DC36':score(model.uncentered(rows,h),y,c,groups,thresholds),
   'UnfilteredDC36':score(UnifiedMean(unfiltered,np.zeros(36),0.,{},0.).uncentered(rows,h),y,c,groups,thresholds),
   'TrainFSPC':score(train_model.predict(rows,h),y,c,groups,thresholds)}
 variants={}
 for key,fit in fitted.items():
  q=fit.predict(rows,h)
  if key=='b6_h1_n50000':np.testing.assert_allclose(q,default_q,atol=1e-9,rtol=1e-10)
  metric=score(q,y,c,groups,thresholds);variants[key]={k:metric[k] for k in ['CRPS','QICE_percent','MSE','MAE','RMSE','MedianMAE','coordinate_count']}
  ids=calibration_states(fit.cal,rows,h);counts=np.array(resources[key]['source_counts'])
  resources[key]['fallback_percent']=100*float(np.mean(counts[ids]<128));resources[key]['strength']=resources[key]['weight']
  atomic(out/'progress.json',dict(stage='score',variants=len(variants),expected=len(VARIANTS),elapsed=time.monotonic()-start))
 cases=load(root/'source_cases.json')
 for case in cases:
  if case['id']!=task['id']:continue
  meta=load(root/'case_metadata'/(task['id']+'.json'));idx=meta['coordinate_index'];assert h[idx]==meta['horizon_id'] and c[idx]==meta['channel_id']
  vr,vraw,vsig=summaries(val);res=(vy-vr)/vsig
  meta.update(source_version=VERSION,fitting_coordinates=len(vy),near_floor_sources=int((vraw<=1.01e-6).sum()),
    extreme_sources=int((abs(res)>100).sum()),near_floor_extreme_sources=int(((vraw<=1.01e-6)&(abs(res)>100)).sum()),
    retained_test_coordinates=len(y),native_mean=float(tm[idx]),native_scale=float(ts[idx]),target=float(y[idx]),
    filtered_equals_unfiltered=bool((vraw>1.01e-6).all()),paired_metric_absolute_differences={'default_generalized_mean':float(np.max(abs(fitted['b6_h1_n50000'].predict(rows[idx:idx+1],h[idx:idx+1])-default_q[idx:idx+1])))})
  nrow=rows[idx:idx+1];hh=h[idx:idx+1];filtered=model.uncentered(nrow,hh)[0];unf=UnifiedMean(unfiltered,np.zeros(36),0.,{},0.).uncentered(nrow,hh)[0]
  caseout=root/'source_case_results';caseout.mkdir(exist_ok=True)
  np.savez_compressed(caseout/(task['id']+'.npz'),raw_scale=vraw,scale=vsig,residual=res,probabilities=(np.arange(100)+.5)/100,
   native=np.sort(rows[idx]),filtered=filtered,unfiltered=unf,fspc=default_q[idx])
  meta['fspc']=dict(method_record_key='FSPCMean',output_mean=float(default_q[idx].mean()),max_absolute=float(abs(default_q[idx]).max()),
    source='uniform affine-plus-state mean and centered source-filtered shape; original coordinate identity',test_target_used_for_curve_or_selection=False)
  atomic(caseout/(task['id']+'.json'),meta)
 result=dict(version=VERSION,task=task,coordinates=len(y),native=native,methods=control_scores,variants=variants,resources=resources,
   template_fit=resources['b6_h1_n50000'],training_template_fit=train_fit,source_excluded_fraction=float(np.mean(summaries(val)[1]<=1.01e-6)),
   neural_training_performed=False,full_test_split=False,evaluation='identical retained test cohort; at most 50000 coordinates',elapsed_seconds=time.monotonic()-start)
 atomic(out/'results.json',result);print('UNIFIED_COMPONENT_COMPLETE',task['id'],round(time.monotonic()-start,1),flush=True)
 return task['id']


def main():
 p=argparse.ArgumentParser()
 for n in ['root','source','mean-root','prior']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--workers',type=int,default=12);a=p.parse_args();tasks=load(a.source/'manifest.json');atomic(a.root/'manifest.json',tasks)
 controller=(a.root/'components_controller.lock').open('a');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pending=[t for t in tasks if not (a.root/'components/runs'/t['id']/'results.json').exists()];failures=[];done=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  active={}
  while pending or active:
   for f in list(active):
    if f.done():
     t=active.pop(f)
     try:done.append(f.result())
     except Exception as error:failures.append(dict(task=t['id'],error=repr(error)));print('COMPONENT_FAILED',t['id'],repr(error),flush=True)
   ready=[t for t in pending if t['backbone']!='DiffusionTS' or (a.root/'diffusion/runs'/t['id']/'results.json').exists()]
   while ready and len(active)<a.workers:
    t=ready.pop(0);pending.remove(t);f=pool.submit(run,t,a.root,a.source,a.mean_root,a.prior);active[f]=t
   atomic(a.root/'components_status.json',dict(expected=249,complete=sum((a.root/'components/runs'/t['id']/'results.json').exists() for t in tasks),pending=len(pending),active=[t['id'] for t in active.values()],failures=failures))
   if pending and not active and all(t['backbone']=='DiffusionTS' for t in pending):
    ds=load(a.root/'diffusion_status.json')
    if ds['failures'] and not ds['active']:raise RuntimeError('Corrected DiffusionTS prerequisites failed')
   time.sleep(3)
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
