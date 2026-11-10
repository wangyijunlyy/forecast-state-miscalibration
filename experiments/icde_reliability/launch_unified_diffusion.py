import argparse,os,subprocess,sys,time
from pathlib import Path
from run_mean_improved_all import load,atomic
from fspc_unified import complete_diffusion

def main():
 p=argparse.ArgumentParser()
 for n in ['root','source','prior']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
 tasks=[t for t in load(a.source/'manifest.json') if t['backbone']=='DiffusionTS']
 def done(t):
  path=a.root/'diffusion/runs'/t['id']/'results.json'
  return path.exists() and complete_diffusion(load(path))
 pending=[t for t in tasks if not done(t)];active={};failures=[]
 while pending or active:
  for gpu in list(active):
   proc,t,log=active[gpu]
   if proc.poll() is not None:
    log.close();del active[gpu]
    if proc.returncode or not done(t):failures.append(t['id'])
  memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  for gpu,used in enumerate(memory.splitlines()):
   if gpu in active or int(used)>1000 or not pending:continue
   t=pending.pop(0);out=a.root/'diffusion/runs'/t['id'];out.mkdir(parents=True,exist_ok=True);atomic(out/'task.json',t);log=(out/'run.log').open('a')
   cmd=[sys.executable,'-u',str(a.root/'code/run_unified_diffusion.py'),'--root',str(a.root),'--source',str(a.source),'--prior',str(a.prior),'--task',str(out/'task.json')]
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',MPLBACKEND='Agg')
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);active[gpu]=proc,t,log
  atomic(a.root/'diffusion_status.json',dict(expected=27,complete=sum(done(t) for t in tasks),pending=len(pending),failures=failures,active={str(g):dict(task=t['id'],pid=p.pid) for g,(p,t,_) in active.items()}))
  time.sleep(5)
 if failures or not all(done(t) for t in tasks):raise SystemExit(1)
if __name__=='__main__':main()
