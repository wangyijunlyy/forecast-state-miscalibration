"""Run one complete relocated task after draining an owned scheduling queue."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--task-id',required=True);parser.add_argument('--gpu',type=int,required=True)
parser.add_argument('--controller-root',type=Path,required=True);args=parser.parse_args()
bundle=args.bundle.resolve();home=bundle/'payload/home/wyj';root=home/'icde_reliability_20261102'
status=json.loads((args.controller_root/'gaussian_status.json').read_text());pid=status['controller_pid']
command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
if str(args.controller_root/'code/launch_gaussian.py').encode() not in command:
    raise ValueError('Scheduling controller identity mismatch')
tasks=json.loads((root/'manifest.json').read_text());task=next(t for t in tasks if t['id']==args.task_id)
audit_directory=bundle/'replay_audits';audit_directory.mkdir(exist_ok=True)
taskfile=audit_directory/(task['id']+'.task.json');taskfile.write_text(json.dumps(task,indent=2)+'\n')
os.kill(pid,signal.SIGSTOP)
try:
    deadline=time.monotonic()+1800
    while True:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if int(output.splitlines()[args.gpu].strip())<256:
            break
        if time.monotonic()>deadline:
            raise TimeoutError('Requested card did not drain')
        time.sleep(5)
    env=dict(os.environ,ICDE_RUN_ROOT=str(root),CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(home/'miniconda3/envs/k2vae/lib/python3.10/site-packages'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    with (audit_directory/(task['id']+'.log')).open('w') as log:
        subprocess.run([os.sys.executable,str(root/'code/run_task.py'),'--task',str(taskfile)],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    actual=json.loads((root/'runs'/task['id']/'results.json').read_text())
    expected=json.loads((bundle/'reference_records/runs'/task['id']/'results.json').read_text())
    differences={}
    for method,metric in expected['methods'].items():
        for key in ['CRPS','MSE','MAE','RMSE','QICE_percent','MedianMAE']:
            delta=actual['methods'][method][key]-metric[key];differences[method+'/'+key]=delta
            if abs(delta)>1e-9:
                raise ValueError('Relocated quality parity mismatch')
    record=dict(task=task['id'],full_native_validation=True,full_native_test=True,
        test_windows=actual['test_windows'],test_batches=actual['test_batches'],
        maximum_metric_difference=max(abs(x) for x in differences.values()),
        metric_differences=differences,
        inputs='relocated source, actual model binaries and native data; no copied primary prediction/quality cache',
        dependencies='existing server Python environment')
    (bundle/'REPLAY_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
    (audit_directory/(task['id']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
finally:
    os.kill(pid,signal.SIGCONT)
