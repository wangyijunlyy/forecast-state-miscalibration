"""Resume twelve native DeepSTUQ evaluations while other queues stay paused."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))
BASE = ROOT/'deepstuq_backbone'
PYTHON = '/home/wyj/miniconda3/envs/k2vae/bin/python'


def done(task):
    identity = task['id']
    return all((BASE/folder/'runs'/identity/'results.json').exists()
               for folder in ['state_controls', 'compiled_deployment']) and all(
                   (BASE/'runs'/identity/name).exists() for name in ['results.json', 'training_partition_comparison.json'])


def main():
    BASE.mkdir(exist_ok=True)
    lock = (BASE/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = json.loads((BASE/'manifest.json').read_text())
    first = json.loads((BASE/'first_task.json').read_text())
    first_task = next(t for t in tasks if t['id'] == first['task'])
    if not all((BASE/folder/'runs'/first_task['id']/'results.json').exists()
               for folder in ['state_controls', 'compiled_deployment']):
        raise ValueError('Complete and audit the first native integration task before dispatch')
    pending = [t for t in tasks if not done(t)]
    active = {}; failures = []
    while pending or active:
        for gpu in list(active):
            process, task, stream = active[gpu]
            code = process.poll()
            if code is not None:
                stream.close(); del active[gpu]
                if code or not done(task):
                    failures.append(dict(task=task['id'], exit_code=code))
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        for gpu, free in enumerate(memory.splitlines()):
            if gpu in active or int(free.strip()) < 7000 or not pending or failures:
                continue
            task = pending.pop(0); directory = BASE/'runs'/task['id']; directory.mkdir(parents=True, exist_ok=True)
            stream = (directory/'run.log').open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', ICDE_RUN_ROOT=str(ROOT))
            process = subprocess.Popen([PYTHON, '-u', str(ROOT/'code/deepstuq_backbone.py'),
                '--dataset', task['dataset'], '--seed', str(task['seed'])],
                stdout=stream, stderr=subprocess.STDOUT, env=env)
            active[gpu] = process, task, stream
            print('START', gpu, task['id'], process.pid, flush=True)
        status = dict(controller_pid=os.getpid(), pending=len(pending), complete=sum(done(t) for t in tasks),
            active={str(g):dict(pid=p.pid, task=t['id']) for g,(p,t,_) in active.items()}, failures=failures)
        temporary = BASE/'status.tmp'; temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(BASE/'status.json')
        if failures and not active:
            raise RuntimeError('Inspect failed native task logs before resuming')
        if pending or active:
            time.sleep(15)
    print('ALL_TWELVE_DEEPSTUQ_BACKBONE_TASKS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
