"""Resume the isolated 249-task 36-state campaign with one worker per GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--auxiliary',action='store_true',help='Process short pending tasks alongside the primary eight workers')
args = parser.parse_args(); root = args.root
prefix = 'auxiliary_' if args.auxiliary else ''
lock = (root/(prefix+'controller.lock')).open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
tasks = json.loads((root/'manifest.json').read_text())
def done(task):
    return (root/'runs'/task['id']/'results.json').exists()
def cost(task):
    p = Path(task['source36_root'])/'runs'/task['id']/'results.json'
    return json.loads(p.read_text())['total_test_seconds']
excluded = set()
if args.auxiliary:
    primary = json.loads((root/'status.json').read_text())
    excluded = {v['task'] for v in primary['active'].values()}
pending = sorted([t for t in tasks if not done(t) and t['id'] not in excluded], key=cost, reverse=not args.auxiliary)
active = {}; failures = []
while pending or active:
    for gpu in list(active):
        p, t, log = active[gpu]
        if p.poll() is not None:
            log.close(); del active[gpu]
            if p.returncode or not done(t):
                failures.append(dict(task=t['id'], returncode=p.returncode))
    for gpu in range(8):
        while pending and done(pending[0]):pending.pop(0)
        if gpu in active or not pending:
            continue
        task = pending.pop(0); out = root/'runs'/task['id']; out.mkdir(parents=True, exist_ok=True)
        taskfile = out/'task.json'; taskfile.write_text(json.dumps(task, indent=2)+'\n')
        log = (out/'run.log').open('a')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', WANDB_MODE='disabled',
            MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
        p = subprocess.Popen([sys.executable, '-u', str(root/'code/run_fsdc36.py'),
            '--task', str(taskfile), '--root', str(root)], stdout=log, stderr=subprocess.STDOUT, env=env)
        active[gpu] = p, task, log
        print('START36', gpu, task['id'], p.pid, flush=True)
    status = dict(controller_pid=os.getpid(), expected=249, complete=sum(done(t) for t in tasks),
        pending=len(pending), failures=failures,
        active={str(g):dict(pid=p.pid, task=t['id']) for g, (p,t,_) in active.items()})
    temp = root/(prefix+'status.tmp'); temp.write_text(json.dumps(status, indent=2)+'\n'); temp.replace(root/(prefix+'status.json'))
    if pending or active:
        time.sleep(10)
print('FSDC36_CAMPAIGN_FINISHED', json.dumps(status), flush=True)
