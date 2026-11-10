"""Finish the approved normalization repair on every Diffusion-TS dataset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from run_location_scale import atomic


def main():
    p=argparse.ArgumentParser()
    for n in ['root','source','mean-root']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    while True:
        s=json.loads((root/'status.json').read_text())
        if s['expected']!=36:raise ValueError('Continuation expects the initial 36-job phase')
        if s['failures']:
            atomic(root/'continuation.json',dict(phase='stopped',reason='initial phase has failed tasks',failures=s['failures']));return
        if s['complete']==36 and not s['active']:break
        atomic(root/'continuation.json',dict(phase='waiting for initial full evaluation',initial_complete=s['complete'],next_expected=54))
        time.sleep(15)
    # Let the first controller release its lock after its final status write.
    time.sleep(15)
    atomic(root/'continuation.json',dict(phase='running complete normalization repair',expected=54))
    command=[sys.executable,'-u',str(root/'code/launch_location_scale.py'),'--root',str(root),
        '--source',str(a.source),'--mean-root',str(a.mean_root),'--unclipped-scope','all']
    result=subprocess.run(command)
    atomic(root/'continuation.json',dict(phase='finished' if result.returncode==0 else 'failed',returncode=result.returncode,expected=54))
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
