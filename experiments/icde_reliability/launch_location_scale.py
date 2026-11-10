"""Restartable isolated GPU queue for location/scale and no-clamp diagnosis."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_location_scale import atomic,load
from location_scale import VERSION,complete_result


def campaign_jobs(tasks,unclipped_scope='priority'):
    priority=['ETTm2','ExchangeRate','ETTh2']
    tasks=sorted([t for t in tasks if t['backbone']=='DiffusionTS'],
        key=lambda t:(priority.index(t['dataset']) if t['dataset'] in priority else 10,t['seed'],t['dataset']))
    return [dict(task=t,mode=mode,external=mode=='unclipped') for t in tasks
        for mode in (['original','unclipped'] if unclipped_scope=='all' or t['dataset'] in priority else ['original'])]


def main():
    p=argparse.ArgumentParser()
    for n in ['root','source','mean-root']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--unclipped-scope',choices=['priority','all'],default='priority');a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=campaign_jobs(load(a.source/'manifest.json'),a.unclipped_scope)
    atomic(root/'manifest.json',jobs)
    def done(job):
        path=root/job['mode']/'runs'/job['task']['id']/'results.json'
        if not path.exists():return False
        r=load(path)
        return complete_result(r,job['task'],job['mode'],job['external'])
    pending=[j for j in jobs if not done(j)];active={};failures=[]
    pilot=jobs[0]
    while pending or active:
        for gpu in list(active):
            proc,job,log=active[gpu]
            if proc.poll() is not None:
                log.close();del active[gpu]
                if proc.returncode or not done(job):
                    failures.append(dict(task=job['task']['id'],mode=job['mode'],returncode=proc.returncode))
                    print('LS_FAILURE',job['mode'],job['task']['id'],proc.returncode,flush=True)
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        for gpu,value in enumerate(used.splitlines()):
            if gpu in active or int(value)>=1000 or len(active)>=a.workers or not pending:continue
            # Verify one full original-sampling run before the rest of the campaign.
            if not done(pilot) and (active or failures):break
            job=pending.pop(0);task=job['task'];out=root/job['mode']/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
            atomic(out/'task.json',task);log=(out/'run.log').open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='2',WANDB_MODE='disabled',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,'-u',str(root/'code/run_location_scale.py'),'--root',str(root),
                '--source',str(a.source),'--mean-root',str(a.mean_root),'--task',str(out/'task.json'),'--mode',job['mode']]
            if job['external']:cmd.append('--external')
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            active[gpu]=proc,job,log;print('LS_START',gpu,job['mode'],task['id'],proc.pid,flush=True)
        status=dict(version=VERSION,pid=os.getpid(),expected=len(jobs),complete=sum(done(j) for j in jobs),pending=len(pending),
            failures=failures,active={str(g):dict(task=j['task']['id'],mode=j['mode'],pid=p.pid) for g,(p,j,_) in active.items()})
        atomic(root/'status.json',status)
        if failures and not done(pilot) and not active:break
        time.sleep(5)
    print('LS_CONTROLLER_EXIT',json.dumps(status),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
