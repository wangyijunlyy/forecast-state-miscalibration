"""Train full DeepSTUQ on cards drained by the primary paired benchmark."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))
PYTHON = '/home/wyj/miniconda3/envs/k2vae/bin/python'
BASE = ROOT/'deepstuq'; BASE.mkdir(exist_ok=True)
lock = (BASE/'controller.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
last = None

def alive(pid, fragment):
    try:
        return fragment.encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except FileNotFoundError:
        return False

while True:
    status = json.loads((ROOT/'status.json').read_text())
    main_live = alive(status['controller_pid'], str(ROOT/'code/launch.py'))
    live_workers = [r for r in status['active'].values() if alive(r['pid'], str(ROOT/'runs'/r['task']))]
    # Once the primary queue is empty, it cannot acquire another card. Its
    # remaining live cards stay reserved; full baseline training can use the
    # drained cards, with baseline inference still waiting for primary exit.
    ready = status['pending'] == 0 and not status['failures']
    current = (status['complete'], main_live, len(live_workers))
    if current != last:
        print('WAIT_PRIMARY', current, flush=True); last = current
    if ready:
        break
    (BASE/'status.json').write_text(json.dumps(dict(stage='waiting_for_primary', controller_pid=os.getpid(),
        primary_complete=status['complete'], primary_controller_live=main_live, primary_live_workers=len(live_workers)), indent=2))
    time.sleep(30)
pending = [(d, s) for s in [1,2,3] for d in ['PEMS08','PEMS04','PEMS03','PEMS07']
           if not (BASE/'runs'/f'{d}_seed{s}'/'results.json').exists()]
active = {}; failures = []
while pending or active:
    for gpu in list(active):
        process, identity, handle = active[gpu]
        code = process.poll()
        if code is not None:
            handle.close(); del active[gpu]
            if code:
                failures.append(dict(task=identity, exit_code=code))
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    primary = json.loads((ROOT/'status.json').read_text())
    reserved = {int(g) for g, v in primary['active'].items() if alive(v['pid'], str(ROOT/'runs'/v['task']))}
    free = [i for i, line in enumerate(memory.splitlines()) if int(line.strip()) < 256 and i not in reserved]
    for gpu in free:
        if gpu in active or not pending or len(failures) >= 3:
            continue
        dataset, seed = pending.pop(0); identity = f'{dataset}_seed{seed}'
        out = BASE/'runs'/identity; out.mkdir(parents=True, exist_ok=True)
        handle = (out/'run.log').open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        process = subprocess.Popen([PYTHON, str(ROOT/'code/deepstuq_task.py'), '--dataset', dataset, '--seed', str(seed)],
            stdout=handle, stderr=subprocess.STDOUT, env=env)
        active[gpu] = process, identity, handle
        print('START', gpu, identity, process.pid, flush=True)
    status = dict(stage='running', controller_pid=os.getpid(), pending=len(pending),
        active={str(g):dict(task=t, pid=p.pid) for g,(p,t,_) in active.items()}, failures=failures,
        complete=len(list((BASE/'runs').glob('*/results.json'))), primary_reserved_cards=sorted(reserved))
    (BASE/'status.json').write_text(json.dumps(status, indent=2))
    if len(failures) >= 3 and not active:
        raise RuntimeError('Three baseline failures; inspect actual traces before continuing')
    if pending or active:
        time.sleep(20)
print('FINISHED', status, flush=True)
