"""Finish the one-shot server experiment with independent raw-source auditing."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from aggregate_mean_anchor_all import aggregate


def main(root, source):
    while True:
        status = json.loads((root/'status.json').read_text())
        if not status['active'] and not status['pending']:
            break
        time.sleep(10)
    if status['complete'] != 249 or status['failures']:
        raise RuntimeError('Campaign has incomplete or failed tasks; inspect status.json')
    audit = aggregate(root, source, selection_source=source)
    if not audit['all_passed'] or not audit['raw_train_validation_sources_independently_audited']:
        raise RuntimeError('Full campaign/source audit failed')
    reused=json.loads((root/'reused_records.json').read_text())
    assert len(reused)==27 and len({r['id'] for r in reused})==27
    record = dict(all_passed=True, full_test_tasks=249, configurations=83, backbones=7,
        reused_pilot_tasks=len(reused), newly_evaluated_tasks=249-len(reused),
        native_and_original_fsdc_records_match_reference=True,
        baseline_metrics_recomputed_tasks=audit['baseline_metrics_recomputed_tasks'],
        baseline_metrics_reused_tasks=audit['baseline_metrics_reused_tasks'],
        independent_train_validation_reconstruction=True, neural_training_performed=False,
        completed_utc=datetime.now(timezone.utc).isoformat())
    (root/'analysis/completion.json').write_text(json.dumps(record,indent=2)+'\n')
    print('MEAN_ANCHOR_ALL_AUDITED', json.dumps(record), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);args=parser.parse_args()
    main(args.root,args.source)
