"""Exclusive, resumable controller with one complete forecast evaluation per GPU."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))
PYTHON = '/home/wyj/miniconda3/envs/k2vae/bin/python'
ROOT.mkdir(parents=True, exist_ok=True)
lock = (ROOT/'controller.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
tasks = json.loads((ROOT/'manifest.json').read_text())
tasks.sort(key=lambda t: (0 if t['seed'] == 1 and t['horizon'] == 192 and t['dataset'].lower() == 'etth1' else 1,
                          t['seed'], t['backbone'], t['dataset'], t['horizon']))
pending = [t for t in tasks if not (ROOT/'runs'/t['id']/'results.json').exists()]
active = {}; failures = []
while pending or active:
    for gpu in list(active):
        proc, task, handle = active[gpu]
        code = proc.poll()
        if code is not None:
            handle.close(); del active[gpu]
            if code:
                failures.append(dict(task=task['id'], exit_code=code))
    for gpu in range(8):
        if gpu in active or not pending:
            continue
        task = pending.pop(0); out = ROOT/'runs'/task['id']; out.mkdir(parents=True, exist_ok=True)
        taskfile = out/'task.json'; taskfile.write_text(json.dumps(task, indent=2))
        handle = (out/'run.log').open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='1', ICDE_RUN_ROOT=str(ROOT))
        proc = subprocess.Popen([PYTHON, str(ROOT/'code/run_task.py'), '--task', str(taskfile)],
            stdout=handle, stderr=subprocess.STDOUT, env=env)
        active[gpu] = proc, task, handle
        print('START', gpu, task['id'], proc.pid, flush=True)
    status = dict(controller_pid=os.getpid(), pending=len(pending),
        active={str(g): dict(task=t['id'], pid=p.pid) for g, (p, t, _) in active.items()},
        complete=sum((ROOT/'runs'/t['id']/'results.json').exists() for t in tasks), failures=failures)
    temporary = ROOT/'status.tmp'; temporary.write_text(json.dumps(status, indent=2)); temporary.replace(ROOT/'status.json')
    if pending or active:
        time.sleep(10)
print('FINISHED', status, flush=True)
