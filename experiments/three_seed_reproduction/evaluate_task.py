"""Fit, complete-validation diagnosis, then full-test paired evaluation."""
import argparse,json,pickle,random,sys,time
from pathlib import Path
import numpy as np
import torch
from core import CommonMetrics,severe_trigger,atomic
from interfaces import prepare
sys.path.insert(0,'/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf')
sys.path.insert(0,'/home/wyj/distribution_binned_diffusion/code')
from reservoir import CalibrationReservoir
from robust_residuals import filter_residuals
from sparc_adapters.posthoc_baselines import DistributionCalibrationCalibrator


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def fit(t,out,interface):
    set_seed(t['seed']+10000);pool=CalibrationReservoir(50000,t['seed'])
    count=0
    with torch.no_grad():
        for step,batch in enumerate(interface['train'],1):
            if step>interface['fit_limit']:break
            x,y,_=interface['predict'](batch,'calibration',step);pool.update(x,y);count+=len(y)
            if step%10==0:print('FIT',t['id'],step,flush=True)
    rows,y,h,_=pool.arrays();mu=rows.mean(1);sigma=rows.std(1,ddof=1).clip(1e-6);z=(y-mu)/sigma
    filtered,meta=filter_residuals(z);models={};configurations={}
    for bins,nh in [(16,1),(64,4)]:
        options=dict(horizon_length=t['horizon'],n_mean_bins=4,n_scale_bins=4,n_horizon_bins=nh,min_bin_size=128,max_pool_size=20000)
        configurations[str(bins)]=options
        for tag,values in [('raw',z),('mad',filtered)]:
            models[f'{bins}_{tag}']=DistributionCalibrationCalibrator.fit_summaries(values,mu,sigma,h,**options)
    with (out/'calibrators.pkl').open('wb') as f:pickle.dump(models,f)
    atomic(out/'fit_metadata.json',dict(fit_split='train',fit_windows=count,retained_points=len(z),seen_points=pool.seen_points,residual_filter=meta,seed=t['seed']+10000,output_samples=100,configurations=configurations))
    return models


def validation(t,out):
    if (out/'validation.json').exists():return
    interface=prepare(t,out);models=fit(t,out,interface);set_seed(t['seed']+20000)
    acc={'Native':CommonMetrics(),'16':CommonMetrics(),'64':CommonMetrics()};amplitude={'16':0.,'64':0.};nonfinite={'16':False,'64':False};windows=0;steps=0
    with torch.no_grad():
        for step,batch in enumerate(interface['val'],1):
            x,y,_=interface['predict'](batch,'validation',step);acc['Native'].update(x,y)
            mean=np.asarray(x,dtype=float).mean(1);denom=np.maximum(np.max(np.abs(x-mean[:,None]),axis=1),1e-6)
            for bins in (16,64):
                key=str(bins);values=models[f'{bins}_raw'].transform(x)
                if not np.isfinite(values).all():nonfinite[key]=True;continue
                acc[key].update(values,y);amp=float(np.max(np.max(np.abs(values-mean[:,None]),axis=1)/denom));amplitude[key]=max(amplitude[key],amp)
            windows+=len(y);steps=step
            if step%20==0:print('VALIDATION',t['id'],step,'windows',windows,flush=True)
    assert windows>0
    try:assert steps==len(interface['val'])
    except TypeError:pass
    base=acc['Native'].result();diag={}
    for bins in ('16','64'):
        metrics=acc[bins].result() if not nonfinite[bins] else None
        ratio=metrics['CRPS']/max(base['CRPS'],1e-15) if metrics else None
        diag[bins]=dict(metrics=metrics,R_val=ratio,A_val=amplitude[bins],binned_nonfinite=nonfinite[bins],
                        triggered=severe_trigger(ratio or 0,amplitude[bins],nonfinite[bins]))
    atomic(out/'validation.json',dict(task_id=t['id'],backbone=t['backbone'],dataset=t['dataset'],seed=t['seed'],horizon=t['horizon'],
        validation_complete=True,full_validation_split=True,validation_windows=windows,validation_batches=steps,native=base,diagnostics=diag))
    print('VALIDATION_COMPLETE',t['id'],diag,flush=True)


def test(t,out):
    if (out/'results.json').exists():return
    root=Path(t['run_root']).parents[1];gates=json.loads((root/'frozen_gates.json').read_text())
    assert gates['validation_complete_tasks']==237 and gates['no_test_metrics_used']
    chosen=gates['groups'][t['backbone']+'/'+t['dataset']]
    interface=prepare(t,out)
    with (out/'calibrators.pkl').open('rb') as f:models=pickle.load(f)
    cal={f'DC{bins}':models[f'{bins}_'+('mad' if chosen[str(bins)] else 'raw')] for bins in (16,64)}
    names=['Native','DC16','DC64'];common={n:CommonMetrics() for n in names};native={n:interface['native_factory']() for n in names}
    original={n:CommonMetrics() for n in names} if t['backbone']=='K2VAE' else None
    set_seed(t['seed']+30000);windows=0;steps=0;begin=time.monotonic()
    with torch.no_grad():
        for step,batch in enumerate(interface['test'],1):
            x,y,origin=interface['predict'](batch,'test',step);outputs={'Native':x,**{n:c.transform(x) for n,c in cal.items()}}
            for name,values in outputs.items():
                common[name].update(values,y);interface['native_update'](native[name],values,y,origin)
                if original:original[name].update(interface['denormalize'](values),origin)
            windows+=len(y);steps=step
            if step%20==0:print('TEST',t['id'],step,'windows',windows,round(time.monotonic()-begin,1),flush=True)
    assert windows>0
    try:assert steps==len(interface['test'])
    except TypeError:pass
    atomic(out/'results.json',dict(task=t,checkpoint=interface['checkpoint'],full_test_split=True,test_windows=windows,test_batches=steps,
        common_normalized={n:a.result() for n,a in common.items()},native_metrics={n:a.result() for n,a in native.items()},
        common_original_units={n:a.result() for n,a in original.items()} if original else None,
        mad_trigger=chosen,test_seed=t['seed']+30000,total_seconds=time.monotonic()-begin))
    print('TEST_COMPLETE',t['id'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--stage',choices=['validation','test'],required=True);a=p.parse_args()
    t=json.loads(a.task.read_text());out=Path(t['run_root']);torch.set_num_threads(2)
    (validation if a.stage=='validation' else test)(t,out)
if __name__=='__main__':main()
