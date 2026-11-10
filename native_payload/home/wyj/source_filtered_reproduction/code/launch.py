"""Eight-card checkpoint reevaluation, persistent per-run logs and status."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/home/wyj/source_filtered_reproduction')
PYTHON='/home/wyj/miniconda3/envs/k2vae/bin/python'
tasks=json.loads((ROOT/'manifest.json').read_text())
# Start a representative changed model immediately, while other cards handle cheap parity checks.
tasks.sort(key=lambda t:(0 if t['backbone']=='DiffusionTS' and t['dataset']=='ETTh1' else 1,
                         t['seed'],t['backbone'],t['dataset'],t['horizon']))
pending=[t for t in tasks if not (ROOT/'runs'/t['id']/'results.json').exists()]
active={}; failures=[]
while pending or active:
    for gpu in list(active):
        proc,t,handle=active[gpu]
        code=proc.poll()
        if code is not None:
            handle.close();del active[gpu]
            if code: failures.append(dict(task=t['id'],exit_code=code))
    for gpu in range(8):
        if gpu in active or not pending: continue
        t=pending.pop(0);out=ROOT/'runs'/t['id'];out.mkdir(parents=True,exist_ok=True)
        taskfile=out/'task.json';taskfile.write_text(json.dumps(t,indent=2))
        log=(out/'run.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
        proc=subprocess.Popen([PYTHON,str(ROOT/'code/evaluate.py'),'--task',str(taskfile),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,env=env)
        active[gpu]=(proc,t,log)
        print('START',gpu,t['id'],proc.pid,flush=True)
    status=dict(pending=len(pending),active={str(g):dict(task=t['id'],pid=p.pid) for g,(p,t,_) in active.items()},
        complete=sum((ROOT/'runs'/t['id']/'results.json').exists() for t in tasks),failures=failures)
    (ROOT/'status.json').write_text(json.dumps(status,indent=2))
    if pending or active: time.sleep(10)
print('FINISHED',status,flush=True)
