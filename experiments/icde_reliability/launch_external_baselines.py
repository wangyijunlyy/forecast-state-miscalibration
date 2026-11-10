"""Restartable queue; one worker on each otherwise idle GPU, pilot first."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from external_baselines import VERSION,METHODS
from run_external_baselines import atomic,load_json


def main():
    p=argparse.ArgumentParser()
    for n in ['root','source','mean-root']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--pilot',default='K2VAE_etth1_h96_seed1')
    a=p.parse_args();root=a.root.resolve();root.mkdir(exist_ok=True)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=load_json(root/'manifest.json');reference=load_json(a.source/'manifest.json')
    assert tasks==reference and len(tasks)==249
    def done(t):
        f=root/'runs'/t['id']/'results.json'
        if not f.exists():return False
        r=load_json(f);assert r['version']==VERSION and r['task']==t and r['full_test_split']
        assert all(n in r['methods'] for n in METHODS)
        return True
    def cost(t):
        r=load_json(a.source/'runs'/t['id']/'results.json')
        return r['methods']['Native']['coordinate_count']
    pending=sorted([t for t in tasks if not done(t)],key=cost,reverse=True)
    pilot=next(t for t in tasks if t['id']==a.pilot)
    if pilot in pending:pending.remove(pilot);pending.insert(0,pilot)
    active={};failures=[];expected=len(tasks)
    while True:
        for gpu in list(active):
            proc,t,log=active[gpu]
            if proc.poll() is not None:
                log.close();del active[gpu]
                if proc.returncode or not done(t):
                    failures.append(dict(task=t['id'],returncode=proc.returncode))
                    print('EXTERNAL_FAILURE',t['id'],proc.returncode,flush=True)
        # Existing controller owns only these PIDs; never terminate unrelated jobs.
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        free=[i for i,s in enumerate(memory.splitlines()) if int(s)<1000]
        for gpu in free:
            if gpu in active or len(active)>=a.workers or not pending:continue
            if not done(pilot) and active:break
            task=pending.pop(0);out=root/'runs'/task['id'];out.mkdir(parents=True,exist_ok=True)
            atomic(out/'task.json',task);log=(out/'run.log').open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='1',NUMBA_NUM_THREADS='2',WANDB_MODE='disabled',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
            proc=subprocess.Popen([sys.executable,'-u',str(root/'code/run_external_baselines.py'),
                '--root',str(root),'--source',str(a.source),'--mean-root',str(a.mean_root),
                '--task',str(out/'task.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=proc,task,log;print('EXTERNAL_START',gpu,task['id'],proc.pid,flush=True)
        status=dict(version=VERSION,controller_pid=os.getpid(),expected=expected,complete=sum(done(t) for t in tasks),
            pending=len(pending),failures=failures,pilot_complete=done(pilot),
            active={str(g):dict(task=t['id'],pid=p.pid) for g,(p,t,_) in active.items()})
        atomic(root/'status.json',status)
        if status['complete']!=getattr(main,'last_aggregated',-1):
            from aggregate_external_baselines import aggregate
            aggregate(root);main.last_aggregated=status['complete']
        if not pending and not active:break
        time.sleep(5)
    print('EXTERNAL_CONTROLLER_EXIT',json.dumps(status),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
