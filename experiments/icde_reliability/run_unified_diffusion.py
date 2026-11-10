"""Fill matched unclipped DiffusionTS internal controls; never train backbone."""
import argparse,fcntl,gzip,pickle,sys,time
from pathlib import Path
import numpy as np
import torch
from run_mean_improved_all import atomic,load,guard_checkpoint,check_parity
from audit_diffusion_sampling import find_native_model,disable_clamp
from fspc_unified import UnifiedMean,VERSION,complete_diffusion,atomic_npz,missing_internal_controls
from portable_dc import summaries,edges
from reliability_metrics import ReliabilityMetrics,state_ids,assert_traversal
from granularity import fit_partition


def run(task,root,source,prior):
 out=root/'diffusion/runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
 lock=(out/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 previous=load(out/'results.json') if (out/'results.json').exists() else None
 if previous and complete_diffusion(previous):return
 assert task['backbone']=='DiffusionTS';guard_checkpoint(task);start=time.monotonic();torch.set_num_threads(2)
 sys.path.insert(0,str(source/'code'));import run_fsdc36 as original;import run_task as legacy
 interface=original.prepare(task,out);disable_clamp(find_native_model(interface))
 corrected=prior/'unclipped/runs'/task['id']
 with np.load(corrected/'val_calibration.npz') as d:val,y,h,c=[d[k] for k in ['rows','targets','horizon','channel']]
 with np.load(source/'runs'/task['id']/'calibration.npz') as d:thresholds=d['thresholds']
 with gzip.open(corrected/'models.pkl.gz','rb') as f:models=pickle.load(f)
 mean=UnifiedMean.from_existing(models['FSPCMean']);atomic(out/'parameters.json',mean.to_record())
 missing=missing_internal_controls(previous);controls={}
 if missing:
  fitted,_,_=legacy.fit_models(task,val,y,h,c);controls={k:fitted[k] for k in missing}
 with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:tr=d['rows']
 mu,_,sigma=summaries(tr);me=edges(mu,6);se=edges(np.log(sigma),6)
 # Native validation diagnostics on the corrected sampler, using fixed historical states.
 if not (out/'validation.json').exists():
  legacy.seed(task['seed']+20000);va=ReliabilityMetrics(thresholds,n_states=36);steps=windows=0
  for step,batch in enumerate(interface['val'],1):
   with torch.no_grad():x,yy,_=interface['predict'](batch,'validation',step)
   va.update(x,yy,state_ids(x,me,se));steps=step;windows+=len(yy)
  assert_traversal(interface['val'],steps,windows);atomic(out/'validation.json',dict(task=task,diagnostics={'Native':va.result()},validation_windows=windows,validation_batches=steps))
 atomic(out/'frozen.json',dict(task=task,version=VERSION,parameters=mean.to_record(),backbone_retrained=False))
 accum={n:ReliabilityMetrics(thresholds,n_states=36) for n in ['Native','FSPCMean','DC36']+list(controls)}
 from reservoir import CalibrationReservoir
 pool=CalibrationReservoir(50000,task['seed']+70000);legacy.seed(task['seed']+30000);steps=windows=0
 for step,batch in enumerate(interface['test'],1):
  with torch.no_grad():x,yy,_=interface['predict'](batch,'test',step)
  b,s,H,C=x.shape;flat=x.transpose(0,2,3,1).reshape(-1,s);hh=np.tile(np.repeat(np.arange(H),C),b)
  groups=state_ids(x,me,se);pool.update(x,yy);accum['Native'].update(x,yy,groups)
  accum['FSPCMean'].update(mean.predict(flat,hh).reshape(b,H,C,s).transpose(0,3,1,2),yy,groups,ordered=True)
  accum['DC36'].update(mean.uncentered(flat,hh).reshape(b,H,C,s).transpose(0,3,1,2),yy,groups,ordered=True)
  for name,cal in controls.items():accum[name].update(cal.transform(x),yy,groups,ordered=name=='GlobalDC')
  steps=step;windows+=len(yy)
  if step==1 or step%10==0:
   print('UNIFIED_DIFF_TEST',task['id'],step,flush=True);atomic(out/'progress.json',dict(batches=step,windows=windows))
 assert_traversal(interface['test'],steps,windows);scores={n:a.result() for n,a in accum.items()}
 old=load(corrected/'results.json');check_parity(scores['Native'],old['methods']['Native']);check_parity(scores['FSPCMean'],old['methods']['FSPCMean'])
 rows,targets,hh,cc=pool.arrays();atomic_npz(out/'test_reservoir.npz',rows=rows,targets=targets,horizon=hh,channel=cc)
 if previous:scores={**previous['methods'],**scores}
 record=dict(version=VERSION,task=task,methods=scores,full_test_split=True,test_batches=steps,test_windows=windows,
   original_corrected_scores_reproduced=True,backbone_retrained=False,sampling='unclipped',total_seconds=time.monotonic()-start)
 assert complete_diffusion(record),'Incomplete matched internal controls'
 atomic(out/'results.json',record)
 print('UNIFIED_DIFF_COMPLETE',task['id'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['root','source','prior','task']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();run(load(a.task),a.root,a.source,a.prior)
