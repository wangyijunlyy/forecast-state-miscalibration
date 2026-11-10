"""Restartable no-retraining queue; each backbone gated by a full pilot."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from mean_improved_all import VERSION,valid_result,reuse_diffusion_record,designated_pilots
from run_mean_improved_all import atomic,load


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def main():
    p=argparse.ArgumentParser()
    for n in ['root','source','mean-root','reuse-root']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=load(a.source/'manifest.json');assert len(tasks)==249 and len({(t['backbone'],t['dataset'],t['horizon']) for t in tasks})==83
    atomic(root/'manifest.json',tasks);reused=[]
    for t in tasks:
        if t['backbone']=='DiffusionTS':
            path=a.reuse_root/'unclipped/runs'/t['id']/'results.json'
            result=reuse_diffusion_record(load(path),t,path);atomic(root/'runs'/t['id']/'results.json',result)
            atomic(root/'runs'/t['id']/'fit.json',result['fit']);reused.append(dict(task=t['id'],source=str(path)))
    atomic(root/'reused_records.json',reused)
    def done(t):
        path=root/'runs'/t['id']/'results.json'
        return path.exists() and valid_result(load(path),t)
    pilots=load(root/'pilots.json') if (root/'pilots.json').exists() else {}
    pilot_tasks=designated_pilots(tasks,pilots)
    previous=load(root/'status.json') if (root/'status.json').exists() else {}
    running={v['task']:dict(v,gpu=int(g)) for g,v in previous.get('active',{}).items()}
    prior_failures=previous.get('prior_failures',[])+previous.get('failures',[])
    active={};pending=[];failures=[]
    for t in tasks:
        if done(t):continue
        resumed=running.get(t['id'])
        if resumed and alive(resumed['pid']):
            active[resumed['gpu']]=(None,t,None,resumed['pid']);continue
        pilot=pilots.get(t['backbone'])
        if pilot and pilot['task']==t['id'] and alive(pilot['pid']):
            active[pilot['gpu']]=(None,t,None,pilot['pid'])
        else:pending.append(t)
    blocked=set()
    def snapshot():
        return dict(version=VERSION,pid=os.getpid(),expected=len(tasks),complete=sum(done(t) for t in tasks),reused=len(reused),
            pending=len(pending),failures=failures,prior_failures=prior_failures,blocked_backbones=sorted(blocked),
            active={str(g):dict(task=t['id'],pid=pid) for g,(_,t,_,pid) in active.items()})
    status=snapshot();atomic(root/'status.json',status)
    while pending or active:
        for gpu in list(active):
            proc,t,log,pid=active[gpu]
            finished=(proc.poll() is not None) if proc else (done(t) or not alive(pid))
            if finished:
                if log:log.close()
                del active[gpu]
                if not done(t):
                    failures.append(dict(task=t['id'],backbone=t['backbone'],returncode=proc.returncode if proc else None))
                    if pilot_tasks.get(t['backbone'])==t:blocked.add(t['backbone'])
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        for gpu,value in enumerate(used.splitlines()):
            if gpu in active or int(value)>=1000 or len(active)>=a.workers:continue
            candidates=[t for t in pending if t['backbone'] not in blocked and
                (done(pilot_tasks[t['backbone']]) or t==pilot_tasks[t['backbone']])]
            if not candidates:continue
            # Spread backbone classes across workers before using the remaining slots.
            active_names={t['backbone'] for _,t,_,_ in active.values()}
            t=next((t for t in candidates if t['backbone'] not in active_names),candidates[0]);pending.remove(t)
            out=root/'runs'/t['id'];out.mkdir(parents=True,exist_ok=True);atomic(out/'task.json',t)
            log=(out/'run.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',MPLBACKEND='Agg')
            cmd=[sys.executable,'-u',str(root/'code/run_mean_improved_all.py'),'--root',str(root),'--source',str(a.source),
                '--mean-root',str(a.mean_root),'--task',str(out/'task.json')]
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            active[gpu]=(proc,t,log,proc.pid);print('MEAN_START',gpu,t['id'],proc.pid,flush=True)
        status=snapshot()
        atomic(root/'status.json',status)
        if not active and pending and all(t['backbone'] in blocked for t in pending):break
        time.sleep(5)
    print('MEAN_CONTROLLER_EXIT',status,flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
