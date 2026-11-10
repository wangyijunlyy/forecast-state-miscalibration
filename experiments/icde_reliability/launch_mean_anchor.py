"""Bounded two-workers-per-GPU campaign, with validated immutable reuse."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mean_anchor_campaign import validate_tasks, validate_record

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--source',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
a=p.parse_args();root=a.root
assert 1<=a.workers<=16
lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
tasks=json.loads((root/'manifest.json').read_text())
validate_tasks(tasks,reference=json.loads((a.source/'manifest.json').read_text()))
def done(t):
 path=root/'runs'/t['id']/'results.json'
 if not path.exists():return False
 validate_record(json.loads(path.read_text()),t)
 return True
if len(tasks)==249:
 reused=json.loads((root/'reused_records.json').read_text())
 assert {r['id'] for r in reused}=={t['id'] for t in tasks if t['backbone']=='DiffusionTS'}
 assert all(done(t) for t in tasks if t['backbone']=='DiffusionTS'), 'Pilot records must be reused before launching'
def cost(t):return json.loads((a.source/'runs'/t['id']/'results.json').read_text())['total_test_seconds']
pending=sorted([t for t in tasks if not done(t)],key=cost,reverse=True);active={};failures=[]
while True:
 for slot in list(active):
  proc,t,log=active[slot]
  if proc.poll() is not None:
   log.close();del active[slot]
   if proc.returncode or not done(t):failures.append(dict(task=t['id'],returncode=proc.returncode))
 for slot in range(a.workers):
  if slot in active or not pending:continue
  gpu=slot%8
  t=pending.pop(0);out=root/'runs'/t['id'];out.mkdir(parents=True,exist_ok=True)
  path=out/'task.json';path.write_text(json.dumps(t,indent=2)+'\n');log=(out/'run.log').open('a')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='2',WANDB_MODE='disabled',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
  proc=subprocess.Popen([sys.executable,'-u',str(root/'code/run_mean_anchor.py'),'--task',str(path),
                         '--root',str(root),'--source',str(a.source)],env=env,stdout=log,stderr=subprocess.STDOUT)
  active[slot]=proc,t,log;print('MEAN_ANCHOR_START',slot,gpu,t['id'],proc.pid,flush=True)
 status=dict(controller_pid=os.getpid(),expected=len(tasks),complete=sum(done(t) for t in tasks),pending=len(pending),
             failures=failures,active={str(g):dict(task=t['id'],pid=proc.pid,gpu=g%8) for g,(proc,t,_) in active.items()})
 tmp=root/'status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(root/'status.json')
 if not pending and not active:break
 time.sleep(5)
print('MEAN_ANCHOR_FINISHED',json.dumps(status),flush=True)
if failures:raise SystemExit(1)
