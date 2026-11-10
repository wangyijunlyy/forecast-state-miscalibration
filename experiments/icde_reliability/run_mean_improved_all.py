"""Complete paired evaluation of mean-only FSPC using frozen checkpoints."""
import argparse
import fcntl
import gzip
import inspect
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch
from mean_improved_all import VERSION,METHODS,fit_mean_models,valid_result,regular_origin_stride
from location_scale import replay_windows
from portable_dc import summaries,edges
from reliability_metrics import ReliabilityMetrics,state_ids,assert_traversal
from audit_diffusion_sampling import find_native_model,dataset_stride


def load(p):return json.loads(Path(p).read_text())

def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def guard_checkpoint(task):
    if task['backbone']=='DeepSTUQ':
        directory=Path('/home/wyj/icde_reliability_20261102/deepstuq/runs')/f"{task['dataset']}_seed{task['seed']}"
        r=load(directory/'training.json')
        assert r['training_complete'] and r['awa_epochs']==20
        assert (directory/'model.pt').is_file() and (directory/'temperature.json').is_file()
    else:
        r=load(Path(task['run_root'])/'training.json')
        assert r['training_complete'] and r['seed']==task['seed']
        assert Path(r['checkpoint']).exists()
    return r


def calibration(task,source,interface):
    path=source/'runs'/task['id'];record=load(path/'validation.json')
    with np.load(path/'calibration.npz') as d:
        rows,y,h,c,thresholds=[d[k] for k in ['rows','targets','horizon','channel','thresholds']]
    loader=interface['val'];H=task['horizon'];C=len(thresholds);n=record['calibration_windows']
    batch_size=loader.batch_size;sizes=[batch_size]*(n//batch_size)+([n%batch_size] if n%batch_size else [])
    w,hh,cc=replay_windows(n,H,C,50000,task['seed']+40000,sizes)
    np.testing.assert_array_equal(h,hh);np.testing.assert_array_equal(c,cc)
    assert record['calibration_seen_coordinates']==n*H*C and sum(sizes)==n
    cv=inspect.getclosurevars(interface['predict']).nonlocals
    name=task['backbone'];dataset=loader.dataset
    if name=='K2VAE':
        base=dataset.transformed_dataset.base_dataset
        origins=np.array([entry['start'].ordinal+entry['target'].shape[-1]-H for entry in base])
        assert len(origins)==record['validation_windows']
        stride=regular_origin_stride(origins) if len(origins)>1 else H
        def targets(batch):return cv['util'].normalized_batch(cv['m'],batch,torch.device('cuda:0')).future_target_cdf.detach().cpu().numpy()
        origin_source='raw ordered GluonTS entry start + target length - horizon; actual regular origin stride verified'
    elif name=='PPM':
        assert type(dataset).__name__ in ('Dataset_ETT_hour','Dataset_ETT_minute','Dataset_Custom','Dataset_Solar')
        stride=1;args=cv['args']
        def targets(batch):return cv['util'].batch_target(batch[1].float(),args).cpu().numpy()
        origin_source='chronological dataset index; author rolling stride one'
    elif name=='DeepSTUQ':
        assert type(dataset).__name__=='TensorDataset'
        stride=1
        def targets(batch):return cv['data'][3].inverse_transform(batch[1][..., :1]).squeeze(-1).cpu().numpy()
        origin_source='chronological tensor loader; author rolling stride one'
    else:
        model=find_native_model(interface);stride=dataset_stride(dataset)
        def targets(batch):return model._target(batch[1].to('cuda:0').float()).detach().cpu().numpy()
        origin_source='native dataset chronological origin stride'
    offset=0;seen=[];matched=0
    for step,batch in enumerate(loader,1):
        if step>record['calibration_fit_limit']:break
        yy=targets(batch);b=len(yy);assert yy.shape[1:]==(H,C)
        take=(w>=offset)&(w<offset+b)
        np.testing.assert_array_equal(yy[w[take]-offset,h[take],c[take]],y[take])
        matched+=int(take.sum());offset+=b;seen.append(b)
    assert offset==n and seen==sizes and matched==len(y)
    return rows,y,w,thresholds,dict(calibration_windows=n,origin_stride=stride,origin_source=origin_source,
        batch_sizes=sizes,priority_seed=task['seed']+40000,
        coordinate_provenance='exact reservoir replay; horizon/channel and actual normalized targets verified',
        fit_source=str(path/'calibration.npz'))


def score_rows(acc,rows,y,c,groups,thresholds,ordered):
    # Treat retained coordinates as channels, with their original event thresholds.
    m=ReliabilityMetrics(thresholds[c],n_states=36)
    m.update(rows.T[None,:,None,:],y[None,None,:],groups,ordered=ordered)
    return m.result()


def check_parity(got,expected):
    for k in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
        np.testing.assert_allclose(got[k],expected[k],atol=1e-9,rtol=1e-10)
    np.testing.assert_array_equal(got['state_coverage']['counts'],expected['state_coverage']['counts'])
    np.testing.assert_allclose(got['events']['brier'],expected['events']['brier'],atol=1e-10,rtol=1e-10)


def run(task,root,source,mean_root):
    if task['backbone']=='DiffusionTS':raise ValueError('Reuse completed unclipped results; do not regenerate')
    out=root/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
    lock=(out/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (out/'results.json').exists():
        if valid_result(load(out/'results.json'),task):return
        raise ValueError('Preserve incompatible result and select a fresh root')
    start=time.monotonic();atomic(out/'task.json',task)
    guard_checkpoint(task)
    sys.path.insert(0,str(source/'code'))
    import run_fsdc36 as original
    import run_task as legacy
    torch.set_num_threads(2)
    locks=root/'prepare_locks';locks.mkdir(exist_ok=True)
    with (locks/(task['dataset'].lower()+'.lock')).open('a') as dl:
        fcntl.flock(dl,fcntl.LOCK_EX);interface=original.prepare(task,out)
    rows,y,window,thresholds,provenance=calibration(task,source,interface)
    with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:
        train,ty=[d[k] for k in ['rows','targets']]
    mu,_,sigma=summaries(train);me=edges(mu,6);se=edges(np.log(sigma),6)
    models,fit=fit_mean_models(task,rows,y,window,train,ty,provenance['calibration_windows'],provenance['origin_stride'])
    with gzip.open(out/'models.pkl.gz','wb') as f:pickle.dump(models,f,protocol=4)
    atomic(out/'fit.json',fit);atomic(out/'frozen.json',dict(version=VERSION,task=task,fit=fit,provenance=provenance,
        checkpoint=interface['checkpoint'],backbone_retrained=False))
    print('MEAN_FROZEN',task['id'],fit['weight'],fit['split']['protocol'],flush=True)
    old=load(source/'runs'/task['id']/'results.json')
    anchor=load(mean_root/'runs'/task['id']/'results.json')['methods']['DC36ControlledMean']
    assert old['full_test_split']
    # A cache is eligible only when its row count equals the FULL source population.
    cache=source/'runs'/task['id']/'test_reservoir.npz'
    cached=None
    with np.load(cache) as d:
        if len(d['targets'])==old['methods']['Native']['coordinate_count']:
            cached=tuple(d[k] for k in ['rows','targets','channel'])
    if cached is not None:
        test,test_y,channels=cached;tm,_,ts=summaries(test)
        groups=6*np.searchsorted(me[1:-1],tm,side='right')+np.searchsorted(se[1:-1],np.log(ts),side='right')
        scores={'Native':score_rows(None,test,test_y,channels,groups,thresholds,False)}
        for name,model in models.items():scores[name]=score_rows(None,model.predict(test),test_y,channels,groups,thresholds,True)
        steps=old['test_batches'];windows=old['test_windows'];execution='all full-test coordinates replayed from exact existing Native cache'
    else:
        accum={m:ReliabilityMetrics(thresholds,n_states=36) for m in METHODS}
        legacy.seed(task['seed']+30000);windows=steps=0
        for step,batch in enumerate(interface['test'],1):
            with torch.no_grad():x,yy,_=interface['predict'](batch,'test',step)
            b,s,H,C=x.shape;flat=x.transpose(0,2,3,1).reshape(-1,s);groups=state_ids(x,me,se)
            accum['Native'].update(x,yy,groups)
            for name,model in models.items():
                q=model.predict(flat)
                accum[name].update(q.reshape(b,H,C,s).transpose(0,3,1,2),yy,groups,ordered=True)
            windows+=len(yy);steps=step
            if step==1 or step%10==0:
                print('MEAN_TEST',task['id'],step,old['test_batches'],flush=True)
                atomic(out/'progress.json',dict(batches=step,expected=old['test_batches'],elapsed=time.monotonic()-start))
        assert_traversal(interface['test'],steps,windows)
        scores={m:a.result() for m,a in accum.items()};execution='complete frozen Native test iterator, shared draws'
    assert windows==old['test_windows'] and steps==old['test_batches']
    check_parity(scores['Native'],old['methods']['Native']);check_parity(scores['FSPCRefit'],anchor)
    for m in METHODS:
        assert scores[m]['coordinate_count']==old['methods']['Native']['coordinate_count']
        np.testing.assert_array_equal(scores[m]['state_coverage']['counts'],scores['Native']['state_coverage']['counts'])
    result=dict(version=VERSION,task=task,full_test_split=True,test_batches=steps,test_windows=windows,
        checkpoint=interface['checkpoint'],backbone_retrained=False,methods=scores,fit=fit,provenance=provenance,
        native_parity=True,original_fspc_parity=True,evaluation_execution=execution,total_seconds=time.monotonic()-start)
    assert valid_result(result,task);atomic(out/'results.json',result)
    print('MEAN_COMPLETE',task['id'],{m:round(v['CRPS'],6) for m,v in scores.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','source','mean-root','task']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(load(a.task),a.root.resolve(),a.source.resolve(),a.mean_root.resolve())
