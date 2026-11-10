"""Persistent staged queue with safe small-job concurrency and live adoption."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from core import stage_ready
from dispatch import plan_dispatch
from validation_runtime import DTS_VALIDATION_VERSION


def live(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


class Adopted:
    def __init__(self,pid):self.pid=pid;self.returncode=None
    def poll(self):
        if live(self.pid):return None
        self.returncode=-999;return self.returncode


def file_for(t,stage):return Path(t['run_root'])/({'training':'training.json','validation':'validation.json','test':'results.json'}[stage])


def complete(t,stage):
    p=file_for(t,stage)
    if not p.exists():return False
    try:
        r=json.loads(p.read_text())
        if stage=='training':return r['training_complete'] and Path(r['checkpoint']).is_file() and r['seed']==t['seed'] and (t['backbone']!='DiffusionTS' or r.get('metadata',{}).get('validation_protocol')==DTS_VALIDATION_VERSION)
        if stage=='validation':return r['validation_complete'] and r['full_validation_split']
        return r['full_test_split'] and set(r['common_normalized'])=={'Native','DC16','DC64'}
    except (ValueError,KeyError):return False


def code_version(stage):
    files=['train_task.py','core.py','validation_runtime.py'] if stage=='training' else ['evaluate_task.py','interfaces.py','core.py']
    return hashlib.sha256(b''.join(Path(__file__).with_name(p).read_bytes() for p in files)).hexdigest()[:12]


def free_slot(gpu,running):
    i=0
    while f'{gpu}.{i}' in running:i+=1
    return f'{gpu}.{i}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--lowdim-per-gpu',type=int,default=3);a=p.parse_args()
    assert 1<=a.lowdim_per_gpu<=3
    lock=(a.root/'.scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads((a.root/'manifest.json').read_text());byid={t['id']:t for t in tasks};running={};failures={};gpuids=a.gpus.split(',')
    failure_path=a.root/'attempt_failures.json'
    if failure_path.exists():failures=json.loads(failure_path.read_text())
    # Adopt live top-level workers, excluding their forked DataLoader children.
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit() or not live(int(procdir.name)):continue
        try:
            cmd=procdir.joinpath('cmdline').read_bytes().decode().split('\0')
            if not any(str(Path(__file__).with_name(n)) in cmd for n in ('train_task.py','evaluate_task.py')):continue
            parent_pid=procdir.joinpath('stat').read_text().split()[3]
            parent_cmd=(Path('/proc')/parent_pid/'cmdline').read_bytes().decode().split('\0')
            if any(str(Path(__file__).with_name(n)) in parent_cmd for n in ('train_task.py','evaluate_task.py')):continue
            tid=Path(cmd[cmd.index('--task')+1]).stem
            if tid not in byid:continue
            env=dict(x.split('=',1) for x in procdir.joinpath('environ').read_bytes().decode().split('\0') if '=' in x)
            gpu=env['CUDA_VISIBLE_DEVICES'];stage='training' if str(Path(__file__).with_name('train_task.py')) in cmd else cmd[cmd.index('--stage')+1]
            slot=free_slot(gpu,running)
            running[slot]=(Adopted(int(procdir.name)),byid[tid],None,stage,code_version(stage),gpu);print('ADOPTED',tid,gpu,procdir.name,flush=True)
        except (FileNotFoundError,PermissionError):continue
    while True:
        for slot in list(running):
            proc,t,log,stage,version,gpu=running[slot]
            if proc.poll() is None:continue
            if log:log.close()
            running.pop(slot)
            if not complete(t,stage):
                key=stage+':'+t['id'];f=failures.setdefault(key,[])
                f.append(dict(pid=proc.pid,returncode=proc.returncode,code_version=version,time=time.time()))
                failure_path.write_text(json.dumps(failures,indent=2));print('FAILED',key,proc.returncode,flush=True)
            else:print('FINISHED',stage,t['id'],flush=True)
        stage=next((s for s in ('training','validation','test') if not all(complete(t,s) for t in tasks)),'complete')
        if stage=='complete':
            from report import summarize
            audit=summarize(a.root,tasks);assert audit['complete'];print('ALL_COMPLETE',flush=True);break
        if stage=='validation':
            train=[json.loads(file_for(t,'training').read_text()) for t in tasks];assert stage_ready(train,237)
        if stage=='test' and not (a.root/'frozen_gates.json').exists():
            from report import freeze
            freeze(a.root,tasks);print('GATES_FROZEN',flush=True)
        active={t['id'] for _,t,_,s,_,_ in running.values() if s==stage};version=code_version(stage)
        queue=[t for t in tasks if not complete(t,stage) and t['id'] not in active
               and not any(x['code_version']==version for x in failures.get(stage+':'+t['id'],[]))]
        owners={g:[t for _,t,_,_,_,gpu in running.values() if gpu==g] for g in gpuids}
        plans,remaining=plan_dispatch(queue,owners,gpuids,stage,a.lowdim_per_gpu)
        for gpu,t in plans:
            out=Path(t['run_root']);out.mkdir(parents=True,exist_ok=True);slot=free_slot(gpu,running)
            log=(out/('train.log' if stage=='training' else stage+'.log')).open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_MODE='disabled',MPLBACKEND='Agg',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            script='train_task.py' if stage=='training' else 'evaluate_task.py'
            cmd=[sys.executable,'-u',str(Path(__file__).with_name(script)),'--task',str(a.root/'tasks'/f'{t["id"]}.json')]
            if stage!='training':cmd+=['--stage',stage]
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);running[slot]=(proc,t,log,stage,version,gpu)
            print('STARTED',stage,t['id'],'gpu',gpu,'slot',slot,'pid',proc.pid,flush=True)
        status=dict(stage=stage,expected=237,complete={s:sum(complete(t,s) for t in tasks) for s in ('training','validation','test')},pending=len(remaining),
            running={slot:dict(gpu=gpu,id=t['id'],pid=proc.pid,stage=s,live=live(proc.pid)) for slot,(proc,t,_,s,_,gpu) in running.items()},
            failed_current_version=[key for key,v in failures.items() if key.startswith(stage+':') and any(x['code_version']==version for x in v)],
            lowdim_training_slots_per_gpu=a.lowdim_per_gpu,updated_unix=time.time())
        tmp=a.root/'stage_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(a.root/'stage_status.json');time.sleep(10)
if __name__=='__main__':main()
