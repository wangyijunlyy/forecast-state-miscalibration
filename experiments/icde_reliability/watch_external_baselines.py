"""Regenerate audited reports as the independent server campaign progresses."""
import argparse,json,time
from pathlib import Path
from aggregate_external_baselines import aggregate
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
last=None
while True:
    state=json.loads((a.root/'status.json').read_text())
    key=(state['complete'],len(state['failures']))
    if key!=last:
        audit=aggregate(a.root);last=key
        if audit['errors']:raise SystemExit('Result audit failed')
    if not state['active'] and (state['failures'] or not state['pending']):break
    time.sleep(15)
