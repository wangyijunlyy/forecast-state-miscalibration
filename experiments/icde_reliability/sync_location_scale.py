"""Quietly collect the running authorized campaign and refresh its local report."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import time
from report_location_scale import report


def main():
    p=argparse.ArgumentParser();p.add_argument('--local-root',type=Path,required=True)
    p.add_argument('--remote-root',required=True);p.add_argument('--host',default='4090-wyj');a=p.parse_args()
    root=a.local_root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'sync.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    remote_python='/home/wyj/icde_external_baselines_20261105/env/bin/python'
    includes=['results.json','fit.json','sampling_audit.json','frozen.json','status.json','manifest.json',
        'continuation.json','integrity_audit.json','*.csv','summary.json']
    while True:
        try:
            for script in ['aggregate_location_scale.py','audit_location_scale.py']:
                command=' '.join(shlex.quote(x) for x in [remote_python,a.remote_root+'/code/'+script,a.remote_root])
                result=subprocess.run(['ssh','-o','ConnectTimeout=12',a.host,command],capture_output=True,text=True,timeout=120)
                if result.returncode:raise RuntimeError(script+': '+result.stderr[-1000:])
            cmd=['rsync','-az','--prune-empty-dirs',"--include=*/"]
            cmd+=['--include='+name for name in includes]
            cmd+=['--exclude=*',a.host+':'+a.remote_root+'/',str(root)+'/']
            subprocess.run(cmd,check=True,timeout=120,capture_output=True)
            report(root)
            s=json.loads((root/'status.json').read_text())
            watcher=dict(complete=s['complete'],expected=s['expected'],failures=s['failures'],last_sync=time.time())
            (root/'sync_status.json').write_text(json.dumps(watcher,indent=2)+'\n')
            if s['complete']==s['expected'] and not s['active']:return
            if s['failures'] and not s['active'] and s['pending']==0:
                raise RuntimeError('Campaign ended with missing/failed tasks; see coverage.csv and status.json')
        except Exception as error:
            (root/'sync_error.json').write_text(json.dumps(dict(error=str(error),time=time.time()),indent=2)+'\n')
            print('SYNC_RETRY',str(error),flush=True)
        time.sleep(45)


if __name__=='__main__':main()
