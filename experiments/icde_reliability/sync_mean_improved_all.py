"""Collect only small result metadata and refresh paired coverage/report quietly."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
from aggregate_mean_improved_all import aggregate


def sync_once(root,remote,host):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    command=' '.join(shlex.quote(x) for x in ['/home/wyj/icde_external_baselines_20261105/env/bin/python',
        remote+'/code/aggregate_mean_improved_all.py',remote])
    subprocess.run(['ssh','-o','ConnectTimeout=12',host,command],check=True,capture_output=True,text=True,timeout=120)
    includes=['results.json','fit.json','frozen.json','status.json','manifest.json','reused_records.json','summary.json','*.csv','REPORT.md']
    cmd=['rsync','-az','--prune-empty-dirs','--include=*/']+['--include='+n for n in includes]+[
        '--exclude=*',host+':'+remote+'/',str(root)+'/']
    subprocess.run(cmd,check=True,capture_output=True,timeout=120)
    summary=aggregate(root);status=json.loads((root/'status.json').read_text())
    (root/'sync_status.json').write_text(json.dumps(dict(complete=summary['coverage']['complete'],expected=status['expected'],
        last_sync=time.time(),failures=status.get('failures',[])),indent=2)+'\n')
    return summary,status


def main():
    p=argparse.ArgumentParser();p.add_argument('--local-root',type=Path,required=True);p.add_argument('--remote-root',required=True)
    p.add_argument('--host',default='4090-wyj');p.add_argument('--once',action='store_true');a=p.parse_args()
    root=a.local_root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'sync.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            summary,status=sync_once(root,a.remote_root,a.host)
            error=root/'sync_error.json'
            if error.exists():error.unlink()
            if a.once or summary['coverage']['complete']==status['expected']:return
            if not status.get('active') and (not status.get('pending') or status.get('blocked_backbones')):
                (root/'sync_error.json').write_text(json.dumps(dict(error='Controller stopped before complete coverage; inspect status and logs',time=time.time()),indent=2)+'\n')
                return
        except Exception as error:
            (root/'sync_error.json').write_text(json.dumps(dict(error=str(error),time=time.time()),indent=2)+'\n')
            if a.once:raise
        time.sleep(45)


if __name__=='__main__':main()
