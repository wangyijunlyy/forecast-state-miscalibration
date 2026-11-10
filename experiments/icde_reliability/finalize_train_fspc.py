"""Wait for complete execution, then audit the frozen training/validation sources."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from aggregate_train_fspc import aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root','source','mean-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    while True:
        status = json.loads((args.root/'status.json').read_text())
        if not status['pending'] and not status['active']: break
        time.sleep(10)
    if status['complete']!=249 or status['failures']:
        raise RuntimeError('Incomplete Train FSPC campaign; inspect status.json')
    audit = aggregate(args.root,args.source,args.mean_root,selection_source=args.source)
    if not audit['all_passed']: raise RuntimeError('Source or score audit failed; inspect analysis/audit.json')
    record = dict(all_passed=True,full_test_tasks=249,configurations=83,backbones=7,
        existing_full_test_baseline_scores_reused=True,neural_training_performed=False,
        independent_raw_training_validation_audit=True,completed_utc=datetime.now(timezone.utc).isoformat())
    (args.root/'analysis/completion.json').write_text(json.dumps(record,indent=2)+'\n')
    print('TRAIN_FSPC_ALL_AUDITED',json.dumps(record),flush=True)
