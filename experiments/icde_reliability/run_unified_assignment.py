"""Matched state-association controls for corrected DiffusionTS sampling."""
import argparse,time,fcntl
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from run_mean_improved_all import load,atomic
from run_unified_components import score
from fspc_unified import VERSION
from granularity import fit_partition
from portable_dc import summaries,edges,bounded,EmpiricalCalibrator


def run(t,root,source,prior):
 out=root/'assignment/runs'/t['id'];out.mkdir(parents=True,exist_ok=True)
 if (out/'results.json').exists():return
 with np.load(prior/'unclipped/runs'/t['id']/'val_calibration.npz') as d:x,y,h=[d[k] for k in ['rows','targets','horizon']]
 with (root/'diffusion/runs'/t['id']/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_SH)
  with np.load(root/'diffusion/runs'/t['id']/'test_reservoir.npz') as d:tx,ty,th,c=[d[k] for k in ['rows','targets','horizon','channel']]
 with np.load(source/'runs'/t['id']/'calibration.npz') as d:thresholds=d['thresholds']
 with np.load(source/'train_calibration/runs'/t['id']/'calibration.npz') as d:train=d['rows']
 m,raw,s=summaries(x);valid=raw>1.01e-6;z=(y-m)/s;cal,meta=fit_partition(x,y,h,t['horizon'],6,1)
 tm,_,ts=summaries(train);me=edges(tm,6);se=edges(np.log(ts),6);mu,_,sigma=summaries(tx)
 groups=6*np.searchsorted(me[1:-1],mu,side='right')+np.searchsorted(se[1:-1],np.log(sigma),side='right')
 methods=dict(Native=score(tx,ty,c,groups,thresholds,False),DC36=score(cal.compile().transform_rows(tx,th),ty,c,groups,thresholds))
 globalq=np.quantile(cal.global_pool,(np.arange(100)+.5)/100);methods['GlobalDC']=score(mu[:,None]+sigma[:,None]*globalq,ty,c,groups,thresholds)
 keys,inv=np.unique(cal.keys(m,s,h),axis=0,return_inverse=True)
 for repeat in range(5):
  zz=z.copy();zz[valid]=z[valid][np.random.default_rng(t['seed']+80000+repeat).permutation(valid.sum())]
  pools={tuple(k):bounded(zz[(inv==i)&valid]) for i,k in enumerate(keys) if ((inv==i)&valid).sum()>=128}
  changed=EmpiricalCalibrator(cal.mean_edges,cal.scale_edges,cal.global_pool,pools,t['horizon'],1,int(valid.sum()),int((~valid).sum()))
  assert changed.local_pools.keys()==cal.local_pools.keys()
  methods['Shuffled'+str(repeat)]=score(changed.compile().transform_rows(tx,th),ty,c,groups,thresholds)
 atomic(out/'results.json',dict(version=VERSION,task=t,methods=methods,coordinates=len(ty),fixed_source_mask=True,identical_global_pool=True,identical_state_counts=True,backbone_retrained=False,sampling='unclipped'))
 print('UNIFIED_ASSIGNMENT_COMPLETE',t['id'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['root','source','prior']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();tasks=[t for t in load(a.source/'manifest.json') if t['backbone']=='DiffusionTS'];done=set()
 while len(done)<27:
  ready=[t for t in tasks if t['id'] not in done and (a.root/'diffusion/runs'/t['id']/'results.json').exists()]
  with ProcessPoolExecutor(max_workers=6) as pool:
   futures={pool.submit(run,t,a.root,a.source,a.prior):t['id'] for t in ready}
   for f in as_completed(futures):f.result();done.add(futures[f])
  atomic(a.root/'assignment_status.json',dict(expected=27,complete=len(done)))
  if len(done)<27:time.sleep(5)
