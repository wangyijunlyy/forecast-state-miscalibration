"""Restartable complete Train FSPC campaign with two workers per GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from mean_anchor_campaign import validate_tasks
from train_fspc import validate_record
from run_train_fspc import atomic


def main():
    parser = argparse.ArgumentParser()
    for name in ['root', 'source', 'mean-root']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(); root = args.root
    assert 1 <= args.workers <= 16
    lock = (root/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks = json.loads((root/'manifest.json').read_text())
    validate_tasks(tasks, json.loads((args.source/'manifest.json').read_text()))
    validate_tasks(tasks, json.loads((args.mean_root/'manifest.json').read_text()))
    def done(task):
        path = root/'runs'/task['id']/'results.json'
        if not path.exists(): return False
        validate_record(json.loads(path.read_text()), task); return True
    def cost(task):
        path = args.mean_root/'runs'/task['id']/'results.json'
        return json.loads(path.read_text())['total_seconds']
    pending = sorted([t for t in tasks if not done(t)], key=cost, reverse=True)
    active = {}; failures = []
    while True:
        for slot in list(active):
            proc, task, log = active[slot]
            if proc.poll() is not None:
                log.close(); del active[slot]
                if proc.returncode or not done(task):
                    failures.append(dict(task=task['id'], returncode=proc.returncode))
        for slot in range(args.workers):
            if slot in active or not pending: continue
            task = pending.pop(0); out = root/'runs'/task['id']; out.mkdir(parents=True, exist_ok=True)
            atomic(out/'task.json', task); log = (out/'run.log').open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot % 8), OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', WANDB_MODE='disabled',
                MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
            proc = subprocess.Popen([sys.executable, '-u', str(root/'code/run_train_fspc.py'),
                '--task', str(out/'task.json'), '--root', str(root), '--source', str(args.source),
                '--mean-root', str(args.mean_root)], env=env, stdout=log, stderr=subprocess.STDOUT)
            active[slot] = proc, task, log
            print('TRAIN_FSPC_START', slot, slot % 8, task['id'], proc.pid, flush=True)
        status = dict(controller_pid=os.getpid(), expected=len(tasks), complete=sum(done(t) for t in tasks),
            pending=len(pending), failures=failures,
            active={str(slot):dict(task=t['id'], pid=p.pid, gpu=slot % 8)
                    for slot, (p, t, _) in active.items()})
        atomic(root/'status.json', status)
        if not pending and not active: break
        time.sleep(5)
    print('TRAIN_FSPC_FINISHED', json.dumps(status), flush=True)
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
