"""Re-evaluate numerically affected learned-PIT fits on original full test iterators.

Other methods' results remain paired and unchanged. Original records are retained
under implementation_checks/qian_stable_kernel before replacement.
"""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from qian_pit import SoftCRPSPIT
from portable_dc import edges,summaries
from reliability_metrics import ReliabilityMetrics,state_ids,assert_traversal
from run_external_baselines import atomic,load_json

NAME='Qian2025SoftCRPS'
REVISION='stable-local-kernel-20261109'


def repair(task,root,source):
    out=root/'runs'/task['id'];target=out/'results.json'
    with (out/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        original=load_json(target)
        if original['fit']['methods'][NAME].get('numerical_revision')==REVISION:
            print('ALREADY_STABLE',task['id'],flush=True);return
        begin=time.monotonic()
        with np.load(source/'runs'/task['id']/'calibration.npz') as d:
            rows=d['rows'];y=d['targets'];thresholds=d['thresholds']
        model=SoftCRPSPIT.fit(rows,y,task['seed']+71000)
        fit_seconds=time.monotonic()-begin;model.metadata['fit_seconds']=fit_seconds
        fit=load_json(out/'fit.json')
        # Verify exactly the original data, frozen fit, and task identities.
        assert original['task']==fit['task']==task
        assert fit['all_parameters_frozen_before_test_access']
        assert len(y)==fit['calibration_coordinates']
        with np.load(source/'train_calibration/runs'/task['id']/'calibration.npz') as d:
            mu,_,sigma=summaries(d['rows']);mean_edges=edges(mu,6);scale_edges=edges(np.log(sigma),6)
        sys.path.insert(0,str(source/'code'))
        import run_fsdc36 as native
        import run_task as legacy
        with (root/'data_prepare_locks'/(task['dataset'].lower()+'.lock')).open('a') as dl:
            fcntl.flock(dl,fcntl.LOCK_EX);interface=native.prepare(task,out)
        accum={n:ReliabilityMetrics(thresholds,n_states=36) for n in ['Native',NAME]}
        legacy.seed(task['seed']+30000);steps=windows=0;generation=transform=metric=0.
        for step,batch in enumerate(interface['test'],1):
            t=time.monotonic();torch.cuda.synchronize()
            with torch.no_grad():x,y,_=interface['predict'](batch,'test',step)
            torch.cuda.synchronize();generation+=time.monotonic()-t
            b,s,h,c=x.shape;rows=x.transpose(0,2,3,1).reshape(-1,s)
            groups=state_ids(x,mean_edges,scale_edges);accum['Native'].update(x,y,groups)
            t=time.monotonic();forecast=model.predict(rows);transform+=time.monotonic()-t
            assert forecast.shape==rows.shape and np.isfinite(forecast).all()
            assert np.all(np.diff(forecast,axis=1)>=-1e-10)
            t=time.monotonic();accum[NAME].update(forecast.reshape(b,h,c,s).transpose(0,3,1,2),y,groups,ordered=True)
            metric+=time.monotonic()-t;windows+=len(y);steps=step
            if step==1 or step%10==0:
                print('REPAIR_TEST',task['id'],step,windows,round(time.monotonic()-begin,1),flush=True)
        assert_traversal(interface['test'],steps,windows)
        assert steps==original['test_batches'] and windows==original['test_windows']
        scores={n:a.result() for n,a in accum.items()}
        for key in ['CRPS','QICE_percent','MSE','MAE','RMSE']:
            np.testing.assert_allclose(scores['Native'][key],original['methods']['Native'][key],rtol=1e-10,atol=1e-9)
        np.testing.assert_array_equal(scores[NAME]['state_coverage']['counts'],original['methods']['Native']['state_coverage']['counts'])
        archived=root/'implementation_checks/qian_stable_kernel'/task['id'];archived.mkdir(parents=True,exist_ok=True)
        for filename in ['results.json','fit.json',NAME+'.pkl']:
            if not (archived/filename).exists():shutil.copy2(out/filename,archived/filename)
        with gzip.open(out/(NAME+'.stable.tmp'),'wb',compresslevel=3) as f:pickle.dump(model,f,protocol=4)
        (out/(NAME+'.stable.tmp')).replace(out/(NAME+'.pkl'))
        old_score=original['methods'][NAME]
        original['methods'][NAME]=scores[NAME]
        fit['methods'][NAME]=model.metadata;fit['fit_seconds'][NAME]=fit_seconds
        original['fit']=fit;original['transform_seconds'][NAME]=transform
        original['metric_seconds'][NAME]=metric
        original['serialized_model_bytes'][NAME]=(out/(NAME+'.pkl')).stat().st_size
        original['stable_kernel_repair']=dict(revision=REVISION,native_full_test_parity=True,
            original_record=str(archived/'results.json'),stable_cdf_queries=model.stable_cdf_queries,
            stable_quantile_queries=model.stable_quantile_queries,fit_seconds=fit_seconds,
            native_generation_seconds=generation,total_seconds=time.monotonic()-begin,
            old_selected_k=json.loads((archived/'fit.json').read_text())['methods'][NAME]['selected_k'],
            new_selected_k=model.k,metric_changes={key:scores[NAME][key]-old_score[key] for key in ['CRPS','QICE_percent','MSE']})
        atomic(out/'fit.json',fit);atomic(target,original)
        print('REPAIR_COMPLETE',task['id'],json.dumps(original['stable_kernel_repair']),flush=True)


def controller(root,source,gpus):
    checks=load_json(root/'implementation_checks/qian_stable_kernel/conditioning_audit.json')
    tasks={t['id']:t for t in load_json(root/'manifest.json')}
    pending=[tasks[r['id']] for r in checks if (r['unstable_calibration_queries'] or r['unstable_window_midpoints'])
        and (root/'runs'/r['id']/'results.json').exists()
        and load_json(root/'runs'/r['id']/'results.json')['fit']['methods'][NAME].get('numerical_revision')!=REVISION]
    queues=[[] for _ in gpus]
    for i,t in enumerate(pending):queues[i%len(gpus)].append(t)
    def worker(item):
        gpu,queue=item;finished=[]
        for task in queue:
            logpath=root/'runs'/task['id']/'stable_kernel_repair.log'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2',WANDB_MODE='disabled',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
            with logpath.open('a') as log:
                p=subprocess.run([sys.executable,'-u',__file__,'--root',str(root),'--source',str(source),'--task',task['id']],env=env,stdout=log,stderr=subprocess.STDOUT)
            result=dict(task=task['id'],returncode=p.returncode,gpu=gpu);finished.append(result)
            print('REPAIR_WORKER',json.dumps(result),flush=True)
        return finished
    print('REPAIR_PENDING',len(pending),flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results=[r for result in pool.map(worker,zip(gpus,queues)) for r in result]
    atomic(root/'implementation_checks/qian_stable_kernel/repair_status.json',dict(revision=REVISION,expected=len(pending),results=results,all_passed=all(r['returncode']==0 for r in results)))
    if any(r['returncode'] for r in results):raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--task');parser.add_argument('--gpus',default='5,6,7')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.task:
        task=next(t for t in load_json(args.root/'manifest.json') if t['id']==args.task)
        repair(task,args.root,args.source)
    else:controller(args.root,args.source,[int(x) for x in args.gpus.split(',')])
