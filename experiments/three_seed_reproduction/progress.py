"""Read actual markers and verify live worker handles without changing jobs."""
import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path
from dispatch import shareable
from validation_runtime import DTS_VALIDATION_VERSION


def observe(root):
    tasks=json.loads((root/'manifest.json').read_text());byid={t['id']:t for t in tasks}
    status=json.loads((root/'stage_status.json').read_text());counts=dict(training=0,validation=0,test=0)
    backbone=Counter();invalid=[];completed=set()
    for t in tasks:
        out=Path(t['run_root'])
        for phase,name in [('training','training'),('validation','validation'),('test','results')]:
            p=out/(name+'.json')
            if not p.exists():continue
            try:
                r=json.loads(p.read_text())
                if phase=='training':valid=r.get('training_complete') and r.get('task_id')==t['id'] and r.get('seed')==t['seed'] and Path(r['checkpoint']).is_file()
                elif phase=='validation':valid=r.get('validation_complete') and r.get('full_validation_split') and r.get('task_id')==t['id']
                else:valid=r.get('full_test_split') and r.get('task',{}).get('id')==t['id'] and set(r.get('common_normalized',{}))=={'Native','DC16','DC64'}
                if phase=='training' and t['backbone']=='DiffusionTS':valid=valid and r.get('metadata',{}).get('validation_protocol')==DTS_VALIDATION_VERSION
                if valid:
                    counts[phase]+=1;completed.add((t['id'],phase))
                    if phase=='training':backbone[t['backbone']]+=1
                else:invalid.append(t['id']+':'+phase)
            except (ValueError,KeyError):invalid.append(t['id']+':'+phase)
    workers=[];recently_ended=[];unaccounted=[];identity_issues=[]
    for slot,v in status['running'].items():
        tid=v['id'];proc=Path('/proc')/str(v['pid'])
        try:
            alive=proc.joinpath('stat').read_text().split()[2]!='Z'
            cmd=proc.joinpath('cmdline').read_bytes().decode().split('\0')
        except FileNotFoundError:alive=False;cmd=[]
        if not alive:
            (recently_ended if (tid,v['stage']) in completed else unaccounted).append(tid);continue
        task_path=str(root/'tasks'/(tid+'.json'))
        if '--task' not in cmd or task_path not in cmd:identity_issues.append(tid);continue
        task=byid[tid];name='train.log' if v['stage']=='training' else v['stage']+'.log';log=Path(task['run_root'])/name
        workers.append(dict(slot=slot,gpu=v.get('gpu',slot.split('.')[0]),id=tid,pid=v['pid'],stage=v['stage'],
            backbone=task['backbone'],dataset=task['dataset'],log_age_seconds=round(time.time()-log.stat().st_mtime,1)))
    owners=defaultdict(list)
    for w in workers:owners[w['gpu']].append(w)
    policy_issues=[]
    for gpu,group in owners.items():
        if len(group)>3 or (len(group)>1 and any(w['stage']!='training' or not shareable(byid[w['id']]) for w in group)):policy_issues.append(gpu)
    if len({w['id'] for w in workers})!=len(workers):policy_issues.append('duplicate active task')
    return dict(stage=status['stage'],complete=counts,total=len(tasks),training_by_backbone=dict(backbone),live_workers=len(workers),
        per_gpu={g:len(v) for g,v in owners.items()},workers=workers,recently_ended=recently_ended,terminal_not_yet_accounted=unaccounted,
        failed=status['failed_current_version'],invalid_markers=invalid,pid_identity_issues=identity_issues,concurrency_policy_issues=policy_issues,
        observed_unix=time.time())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch-seconds',type=int,default=0);p.add_argument('--save',action='store_true');p.add_argument('--details',action='store_true');a=p.parse_args()
    if not 0<=a.watch_seconds<=55:p.error('watch-seconds must be0..55')
    start=time.monotonic();previous=None
    while True:
        r=observe(a.root);key=(r['complete'],r['failed'],r['invalid_markers'])
        if key!=previous or time.monotonic()-start>=a.watch_seconds:
            shown=r if a.details else {k:v for k,v in r.items() if k!='workers'}
            print(json.dumps(shown,ensure_ascii=False),flush=True);previous=key
        if a.save:(a.root/'latest_verified_progress.json').write_text(json.dumps(r,indent=2))
        if time.monotonic()-start>=a.watch_seconds or r['failed']:break
        time.sleep(min(5,a.watch_seconds-(time.monotonic()-start)))
if __name__=='__main__':main()
