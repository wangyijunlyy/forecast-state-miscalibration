"""Full-test native / binned / MAD-filtered binned evaluation for four backbones."""
import argparse
from dataclasses import fields
import importlib
import json
import os
from pathlib import Path
import time
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from metrics import ScoreAccumulator
from robust_residuals import filter_residuals
from timediff_batching import sample_timediff_batched

K2ROOT='/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf'
sys.path.insert(0,K2ROOT)
from sparc_adapters.posthoc_baselines import DistributionCalibrationCalibrator
from reservoir import CalibrationReservoir


CLASSES={'NsDiff':('NsDiff_SPARC_PIT','NsDiffSPARCPITForecast'),
'TMDM':('TMDM_SPARC_PIT','TMDMSPARCPITForecast'),
'DiffusionTS':('DiffusionTS_SPARC_PIT','DiffusionTSSPARCPITForecast'),
'TimeDiff':('TimeDiff_SPARC_PIT','TimeDiffSPARCPITForecast')}


def prepare(task,out):
    os.chdir(task['repo']);sys.path.insert(0,task['repo'])
    module,name=CLASSES[task['backbone']]
    cls=getattr(importlib.import_module('src.experiments.'+module),name)
    names={f.name for f in fields(cls)}
    args={k:v for k,v in task['model_args'].items() if k in names}
    args.update(device='cuda:0',num_worker=0,checkpoint_dir=task['checkpoint'],
        data_path=task.get('resolved_data_root',str(Path(task['repo'])/'data')),save_dir=str(out/'setup'),output_dir=str(out/'unused_sparc'))
    if task['backbone']=='NsDiff':args['load_pretrain']=False # all three frozen components are loaded below
    if task['backbone']=='DiffusionTS':
        args.update(cache_dir=str(out/'temporary_native_cache'),cache_dtype='float32',sample_chunk_size=task['sampling_chunk_size'])
    if task['backbone']=='TimeDiff':
        args.update(persistent_workers=False,pin_memory=False,prefetch_factor=2)
    model=cls(**args)
    # Metrics are accumulated synchronously below with the original score
    # semantics; do not create unused original multiprocessing worker pools.
    model._init_metrics=lambda:None
    if hasattr(model,'_normalize_runtime_flags'):model._normalize_runtime_flags()
    model._setup_run(task['seed'])
    if hasattr(model,'_apply_loader_runtime_options'):model._apply_loader_runtime_options()
    model._load_frozen_checkpoint()
    if task.get('native_sampling_execution')=='batched_original_forward_tf32_disabled':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    if task['backbone']=='NsDiff':model.diffusion_config.testing.minisample=task['sampling_chunk_size']
    if task['backbone']=='TMDM':model.tmdm_minisample=task['sampling_chunk_size']
    if hasattr(model,'_set_backbone_eval'):model._set_backbone_eval()
    else:model.model.eval()
    count=min(task['fit_windows'],len(model.train_loader.dataset))
    selected=np.random.default_rng(task['seed']).choice(len(model.train_loader.dataset),count,replace=False)
    fit=DataLoader(Subset(model.train_loader.dataset,selected.tolist()),batch_size=1,shuffle=False,num_workers=0)
    def predict(batch,phase,step):
        x,y,_,origin_y,xm,ym=batch
        x=x.to('cuda:0').float();y=y.to('cuda:0').float()
        xm=xm.to('cuda:0').float();ym=ym.to('cuda:0').float()
        target=model._target(y).detach().cpu().numpy()
        cache_hit=False
        with torch.no_grad():
            if task['backbone']=='NsDiff':samples=model._sample_nsdiff(x,xm,ym,100)
            elif task['backbone']=='TMDM':samples=model._sample_tmdm(x,xm,ym,100)
            elif task['backbone']=='DiffusionTS':
                samples=model._sample_diffusionts(x,y,100,split='binned_'+phase,batch_index=step)
                # These files belong only to this new run. No historical cache
                # is changed; avoid retaining multi-GB calibration ensembles.
                cache=model._indexed_cache_path('binned_'+phase,100,task['sampling_chunk_size'],step)
                cache.unlink(missing_ok=True)
            else:
                p=Path(task['test_cache_root'])/( __import__('hashlib').sha1(f'test_step{step}_output'.encode()).hexdigest()+'_n100.pt')
                if phase=='test' and p.exists():
                    cached=torch.load(p,map_location='cpu',weights_only=True);md=cached['metadata']
                    assert md['dataset']==model._dataset_label() and md['data_split']=='test'
                    assert md['num_samples']==100 and md['prediction_length']==task['horizon']
                    assert md['timediff_checkpoint_hash']==model._timediff_checkpoint_hash()
                    samples=cached['samples'].float();cache_hit=True
                    assert samples.shape==(len(x),100,task['horizon'],model.dataset.num_features)
                else:
                    if task.get('native_sampling_execution')=='batched_original_forward_tf32_disabled':
                        samples=sample_timediff_batched(model.model,x,xm,y,ym,100,task['sampling_chunk_size'])
                    else:
                        samples=model._sample_timediff_uncached(x,y,xm,ym,100,task['sampling_chunk_size'])
        native=samples.detach().cpu().numpy()
        assert native.shape[1]==100 and native.shape[2:]==target.shape[1:]
        assert np.isfinite(native).all() and np.isfinite(target).all()
        return native,target,origin_y,cache_hit
    return model,fit,model.test_loader,predict,selected.tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();task=json.loads(a.task.read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():return
    torch.set_num_threads(2);started=time.monotonic()
    model,fit,test,predict,selected=prepare(task,out)
    reservoir=CalibrationReservoir(50000,task['seed'])
    for step,batch in enumerate(fit,1):
        native,target,_,_=predict(batch,'calibration',step);reservoir.update(native,target)
        if step%10==0:print('FIT_FORECAST',task['id'],step,round(time.monotonic()-started,1),flush=True)
    rows,target,horizon,_=reservoir.arrays();mu=rows.mean(1);sigma=rows.std(1,ddof=1).clip(1e-6)
    residual=(target-mu)/sigma;filtered,filter_metadata=filter_residuals(residual)
    begin=time.monotonic()
    kwargs=dict(horizon_length=reservoir.horizon_length,n_mean_bins=4,n_scale_bins=4,n_horizon_bins=4,min_bin_size=128,max_pool_size=20000)
    original=DistributionCalibrationCalibrator.fit_summaries(residual,mu,sigma,horizon,**kwargs)
    robust=DistributionCalibrationCalibrator.fit_summaries(filtered,mu,sigma,horizon,**kwargs)
    fit_metadata=dict(fit_windows=len(selected),fit_batches=step,fit_split='train',selected_train_indices=selected,
        calibration_points_seen=reservoir.seen_points,calibration_points_retained=len(rows),num_samples=100,
        bins_per_axis=4,min_bin_size=128,max_pool_size=20000,residual_filter=filter_metadata,
        fit_seconds=time.monotonic()-begin)
    (out/'fit_metadata.json').write_text(json.dumps(fit_metadata,indent=2))
    print('FIT_COMPLETE',task['id'],len(selected),flush=True)
    del rows,target,mu,sigma,residual,filtered,reservoir
    acc={name:ScoreAccumulator() for name in ['Native','Binned','BinnedFiltered']}
    transform_seconds={'Binned':0.,'BinnedFiltered':0.};windows=0;batches=0;cache_hits=0
    for step,batch in enumerate(test,1):
        native,target,origin_y,hit=predict(batch,'test',step);cache_hits+=hit
        # Preserve original inverse-scaling semantics when enabled.
        for i in range(len(target)):
            n=native[i:i+1];y=target[i:i+1];outputs={'Native':n}
            for name,cal in [('Binned',original),('BinnedFiltered',robust)]:
                t=time.monotonic();outputs[name]=cal.transform(n);transform_seconds[name]+=time.monotonic()-t
            for name,values in outputs.items():
                assert np.isfinite(values).all()
                if model.invtrans_loss:
                    values=model.scaler.inverse_transform(torch.as_tensor(values)).cpu().numpy()
                    truth=origin_y[i:i+1].cpu().numpy()
                else:truth=y
                acc[name].update(values,truth)
        windows+=len(target);batches=step
        print('TEST',task['id'],step,len(test),'windows',windows,round(time.monotonic()-started,1),flush=True)
    assert batches==len(test) and windows==len(test.dataset),(batches,windows,len(test.dataset))
    result=dict(task=task,fit=fit_metadata,metrics={k:v.result() for k,v in acc.items()},
        full_test_split=True,original_fast_test_protocol=True,test_batches=batches,test_windows=windows,
        test_cache_hits=cache_hits,transform_seconds=transform_seconds,total_seconds=time.monotonic()-started,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
        note='Single-stage native calibration, no SPARC. Original and filtered binned share all native draws. Existing seed1 checkpoints; Table1 references are historical summaries.')
    temp=out/'results.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(out/'results.json')
    print('COMPLETE',task['id'],result['metrics'],flush=True)


if __name__=='__main__':main()
