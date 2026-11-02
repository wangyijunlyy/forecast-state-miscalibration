"""Drain full Gaussian-control evaluations onto cards released by DeepSTUQ."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))
PYTHON = '/home/wyj/miniconda3/envs/k2vae/bin/python'
lock = (ROOT/'gaussian.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
tasks = json.loads((ROOT/'manifest.json').read_text()); tasks.sort(key=lambda t:(t['seed'], t['backbone'], t['dataset'], t['horizon']))
pending = [t for t in tasks if not (ROOT/'gaussian_runs'/t['id']/'results.json').exists()]
active = {}; failures = []


def alive(pid, fragment):
    try:
        return fragment.encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except FileNotFoundError:
        return False


while pending or active:
    for gpu in list(active):
        process, task, handle = active[gpu]; code = process.poll()
        if code is not None:
            handle.close(); del active[gpu]
            if code:
                failures.append(dict(task=task['id'], exit_code=code))
    baseline = json.loads((ROOT/'deepstuq/status.json').read_text())
    reserved = {int(g) for g,v in baseline['active'].items() if alive(v['pid'], str(ROOT/'code/deepstuq_task.py'))}
    memory = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True)
    free = [g for g,line in enumerate(memory.splitlines()) if int(line.strip()) < 256 and g not in reserved]
    for gpu in free:
        if gpu in active or not pending or baseline['pending'] or len(failures) >= 3:
            continue
        task = pending.pop(0); output = ROOT/'gaussian_runs'/task['id']; output.mkdir(parents=True, exist_ok=True)
        handle = (output/'run.log').open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        process = subprocess.Popen([PYTHON,str(ROOT/'code/run_gaussian.py'),'--task',str(ROOT/'runs'/task['id']/'task.json')],
                                   stdout=handle,stderr=subprocess.STDOUT,env=env)
        active[gpu] = process, task, handle; print('GAUSSIAN_START', gpu, task['id'], process.pid, flush=True)
    status = dict(controller_pid=os.getpid(), pending=len(pending), failures=failures,
        active={str(g):dict(task=t['id'],pid=p.pid) for g,(p,t,_) in active.items()}, baseline_reserved_cards=sorted(reserved),
        complete=sum((ROOT/'gaussian_runs'/t['id']/'results.json').exists() for t in tasks))
    temporary=ROOT/'gaussian_status.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(ROOT/'gaussian_status.json')
    if len(failures) >= 3 and not active:
        raise RuntimeError('Three Gaussian failures; inspect actual traces before continuing')
    if pending or active:
        time.sleep(15)
print('GAUSSIAN_FINISHED',status,flush=True)
