"""Source-filtered controls on the same retained real forecast coordinates."""
import csv
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT=Path('/home/wyj/source_filtered_reproduction/auxiliary')
OLD=Path('/home/wyj/binned_mechanism/results')
CACHE=Path('/home/wyj/mechanism_interventions/results')
SIMPLE=Path('/home/wyj/simple_calibration_comparison/results')
sys.path.insert(0,'/home/wyj/mechanism_interventions/code')
sys.path.insert(0,'/home/wyj/three_seed_reproduction_20261031/code')
sys.path.insert(0,'/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf')
from intervention_stats import fit_model,transform,state_keys,score,bounded
from core import CommonMetrics
from sparc_adapters.posthoc_baselines import DistributionCalibrationCalibrator as Cls


def load(path):
    with np.load(path) as data: return {k:data[k] for k in data.files}


def fit(fr,fy,fh,H,bins=(4,4,4),shuffle_seed=None):
    model=fit_model(Cls,fr,fy,fh,H,bins=bins,mad=False,exclude_collapsed=True)
    # Shuffle only the surviving residual values, keeping the source filter fixed.
    if shuffle_seed is not None:
        mu=fr.mean(1);std=fr.std(1,ddof=1);kept=std>1.01e-6
        z=(fy-mu)/std.clip(1e-6)
        z[kept]=z[kept][np.random.default_rng(shuffle_seed).permutation(kept.sum())]
        keys=state_keys(model.cal,mu,std.clip(1e-6),fh)
        unique,inverse=np.unique(keys,axis=0,return_inverse=True)
        model.cal.bin_residuals={tuple(int(v) for v in key):bounded(z[(inverse==i)&kept])
            for i,key in enumerate(unique) if np.sum((inverse==i)&kept)>=128}
        model.cal.global_residuals=bounded(z[kept])
    # Publication records describe the actual source filter, without historical filter metadata.
    model.meta={k:v for k,v in model.meta.items() if k not in ('mad','filter')}
    return model


def run(task,context):
    dest=ROOT/task['id']/context;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'results.json').exists(): return
    frdata=load(OLD/task['id']/'train_coordinates.npz' if context=='train_to_test'
                else CACHE/task['id']/'validation_coordinates.npz')
    ev=load(CACHE/task['id']/'test_coordinates.npz')
    fr,fy,fh=frdata['rows'],frdata['targets'],frdata['horizon']
    vr,vy,vh=ev['rows'],ev['targets'],ev['horizon'];H=task['horizon']
    raw64=fit_model(Cls,fr,fy,fh,H,mad=False)
    keys=state_keys(raw64.cal,vr.mean(1),vr.std(1,ddof=1).clip(1e-6),vh)
    noncollapsed=vr.std(1,ddof=1)>1.01e-6
    scores={};meta={}
    def evaluate(name,model=None,mode='full'):
        output=vr if model is None else transform(model,vr,vh,mode)
        result,_=score(output,vy,keys,noncollapsed)
        common=CommonMetrics();common.update(output[:,:,None,None],vy[:,None,None])
        result.update(common.result())
        scores[name]=result
        if model is not None: meta[name]={k:v for k,v in model.meta.items() if k not in ('mad','filter')}
        return output
    evaluate('Native')
    for name,bins in [('GlobalTemplate',(1,1,1)),('Scale4',(1,1,4)),('DC16',(1,4,4)),('DC64',(4,4,4))]:
        model=fit(fr,fy,fh,H,bins);output=evaluate(name,model)
        if name=='DC64':
            moments=evaluate('LocationScale',model,'location_scale')
            if noncollapsed.any():
                assert np.allclose(moments[noncollapsed].mean(1),output[noncollapsed].mean(1),atol=1e-10,rtol=0)
                assert np.allclose(moments[noncollapsed].std(1,ddof=1),output[noncollapsed].std(1,ddof=1),atol=1e-10,rtol=0)
    for repeat in range(5): evaluate('Shuffled'+str(repeat),fit(fr,fy,fh,H,shuffle_seed=task['sampling_seed']+60000+repeat))
    raw16=fit_model(Cls,fr,fy,fh,H,bins=(1,4,4),mad=False)
    evaluate('DC16Unfiltered',raw16);evaluate('DC64Unfiltered',raw64)
    # These formulas do not depend on any residual-pool filter and remain identical.
    previous=json.loads((SIMPLE/task['id']/context/'results.json').read_text())
    assert abs(scores['Native']['crps']-previous['scores']['Native']['crps'])<1e-12
    for name in ['GlobalScaleCRPS','GlobalAffineCRPS','GaussianVariance2P','GaussianAffine4P']:
        scores[name]=previous['scores'][name]
        meta[name]=previous['models'][name]
    prior=json.loads((Path('/home/wyj/dc16_stability_comparison/results')/task['id']/context/'results.json').read_text())
    assert abs(scores['DC16']['CRPS']-prior['scores']['DC16DropCollapsed']['CRPS'])<1e-12
    z=(fy-fr.mean(1))/fr.std(1,ddof=1).clip(1e-6);collapsed=fr.std(1,ddof=1)<=1.01e-6;extreme=abs(z)>100
    result=dict(task=task,context=context,fit_coordinates=len(fr),evaluation_coordinates=len(vr),scores=scores,models=meta,
        source_filter=dict(excluded_sources=int(collapsed.sum()),retained_sources=int((~collapsed).sum()),
            extreme_sources=int(extreme.sum()),extreme_from_excluded=int((extreme&collapsed).sum()),threshold=1.01e-6),
        source_parity=dict(native=True,dc16_source_filter=True),
        fixed_diagnostic_partition=[4,4,4],evaluation_scope='uniform retained coordinates after full split traversal')
    (dest/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('COMPLETE',task['id'],context,flush=True)


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    tasks=json.loads((OLD/'manifest.json').read_text());(ROOT/'manifest.json').write_text(json.dumps(tasks,indent=2))
    failures=[]
    with ProcessPoolExecutor(max_workers=4) as executor:
        jobs={executor.submit(run,t,c):(t['id'],c) for t in tasks for c in ['train_to_test','validation_to_test']}
        for future in as_completed(jobs):
            try: future.result()
            except Exception as e: failures.append(dict(task=jobs[future],error=repr(e)));print('FAILED',jobs[future],repr(e),flush=True)
    (ROOT/'completion_audit.json').write_text(json.dumps(dict(complete=not failures,expected=32,failures=failures),indent=2))
    if failures: raise SystemExit(1)
