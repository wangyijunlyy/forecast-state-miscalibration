"""Run complete-test component tasks once per available GPU; never retrain."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from full_test_evaluation import VERSION, load, atomic


def main():
    p = argparse.ArgumentParser()
    for key in ['root', 'source', 'means', 'aux', 'prior']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--only', default='')
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = load(args.means/'manifest.json')
    atomic(args.root/'manifest.json', tasks)
    if args.only:
        chosen = set(args.only.split(','))
        tasks = [t for t in tasks if t['id'] in chosen]
        assert {t['id'] for t in tasks} == chosen
    def done(task):
        path = args.root/'runs'/task['id']/'results.json'
        if not path.exists():
            return False
        r = load(path)
        return r['version'] == VERSION and r['full_test_split'] and r['native_parity'] and r['frozen_fspc_parity']
    pending = [t for t in tasks if not done(t)]
    active, failures, blocked = {}, [], set()
    while pending or active:
        for gpu in list(active):
            proc, task, stream = active[gpu]
            if proc.poll() is None:
                continue
            stream.close()
            del active[gpu]
            if not done(task):
                failures.append(dict(task=task['id'], returncode=proc.returncode))
                blocked.add(task['backbone'])
                print('FULL_TEST_FAILED', task['id'], proc.returncode, flush=True)
        hold_path = args.root/'held_tasks.json'
        held = set(load(hold_path)) if hold_path.exists() else set()
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).splitlines()
        for gpu, value in enumerate(memory):
            if gpu in active or int(value) >= 1000 or len(active) >= args.workers:
                continue
            ready = [t for t in pending if t['id'] not in held and t['backbone'] not in blocked]
            if not ready:
                continue
            busy = {t['backbone'] for _, t, _ in active.values()}
            task = next((t for t in ready if t['backbone'] not in busy), ready[0])
            pending.remove(task)
            out = args.root/'runs'/task['id']
            atomic(out/'task.json', task)
            stream = (out/'run.log').open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_MODE='disabled', MPLBACKEND='Agg')
            command = [sys.executable, '-u', str(args.root/'code/full_test_evaluation.py')]
            for key in ['root', 'source', 'means', 'aux', 'prior']:
                command.extend(['--'+key, str(getattr(args, key))])
            command.extend(['--task', str(out/'task.json')])
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            active[gpu] = proc, task, stream
            print('FULL_TEST_START', gpu, task['id'], proc.pid, flush=True)
        status = dict(version=VERSION, expected=len(tasks), complete=sum(done(t) for t in tasks),
            pending=len(pending), held_for_baseline_resolution=sorted(held),
            failures=failures, blocked_backbones=sorted(blocked),
            active={str(g): dict(task=t['id'], pid=p.pid) for g, (p, t, _) in active.items()})
        atomic(args.root/'status.json', status)
        if not active and pending:
            break
        if active:
            time.sleep(5)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
