"""Run the inventoried complete-test tasks with one process per GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--external',default='',help='Existing task gpu:pid pairs to wait for without duplicating them');a=p.parse_args()
    tasks=json.loads((a.root/'manifest.json').read_text())
    order={'TimeDiff':0,'NsDiff':1,'TMDM':2,'DiffusionTS':3}
    tasks.sort(key=lambda t:(0 if t['dataset']=='ILI' else 1,order[t['backbone']],t['dataset']))
    external={};external_ids=set()
    for pair in filter(None,a.external.split(',')):
        gpu,pid=pair.split(':');pid=int(pid)
        proc=Path('/proc')/str(pid)
        if proc.exists() and proc.joinpath('stat').read_text().split()[2]!='Z':
            cmd=proc.joinpath('cmdline').read_bytes().decode().split('\0')
            assert str(Path(__file__).with_name('run_task.py')) in cmd
            task_id=Path(cmd[cmd.index('--task')+1]).stem
            external[gpu]=pid;external_ids.add(task_id)
    queue=[t for t in tasks if t['id'] not in external_ids and not (a.root/t['id']/'results.json').exists()]
    running={};status=[]
    while queue or running or external:
        for gpu in a.gpus.split(','):
            if gpu in external:
                proc=Path('/proc')/str(external[gpu])
                if proc.exists() and proc.joinpath('stat').read_text().split()[2]!='Z':continue
                external.pop(gpu)
            if gpu in running and running[gpu][0].poll() is not None:
                process,task,log=running.pop(gpu);log.close();status.append(dict(id=task['id'],returncode=process.returncode,gpu=gpu))
                (a.root/'status.json').write_text(json.dumps(status,indent=2));print('FINISHED',task['id'],process.returncode,flush=True)
            if gpu not in running and queue:
                task=queue.pop(0);out=a.root/task['id'];out.mkdir(exist_ok=True);log=(out/'run.log').open('w')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    WANDB_MODE='disabled',MPLBACKEND='Agg',TF_CPP_MIN_LOG_LEVEL='3')
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('run_task.py')),'--task',str(a.root/'tasks'/f'{task["id"]}.json'),'--output',str(out)]
                process=subprocess.Popen(cmd,cwd=task['repo'],env=env,stdout=log,stderr=subprocess.STDOUT)
                running[gpu]=process,task,log;print('STARTED',task['id'],'gpu',gpu,'pid',process.pid,flush=True)
        time.sleep(3)
    print('ALL_FINISHED',len(status),flush=True)


if __name__=='__main__':main()
