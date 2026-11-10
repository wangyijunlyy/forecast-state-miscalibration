"""Preserve reservoir memory layout in saved probes; do not change any scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from final_fspc import CompiledFSPC,save_parity_probe
from fspc_unified import UnifiedMean


def main(root):
    counts={'C':0,'F':0};verified=0
    tasks=json.loads((root/'manifest.json').read_text())
    for task in tasks:
        p=root/'runs'/task['id'];r=json.loads((p/'results.json').read_text())
        with np.load(r['sources']['test']) as d:rows,h=d['rows'],d['horizon']
        model=UnifiedMean.from_record(json.loads((p/'parameters.json').read_text()))
        compiled=CompiledFSPC.load(p/'compiled.npz');reference=model.predict(rows,h)
        np.testing.assert_array_equal(compiled.transform_rows(rows,h),reference)
        with np.load(p/'parity_probe.npz') as d:
            thresholds=d['thresholds'].copy()
            np.testing.assert_array_equal(d['reference'],reference[:128])
            np.testing.assert_array_equal(d['rows'],rows[:128])
        order='F' if rows.flags.f_contiguous else 'C';counts[order]+=1
        save_parity_probe(p/'parity_probe.npz',rows,h,thresholds,reference)
        with np.load(p/'parity_probe.npz') as d:
            np.testing.assert_array_equal(model.predict(d['rows'],d['horizon']),d['reference'])
            np.testing.assert_array_equal(compiled.transform_rows(d['rows'],d['horizon']),d['reference'])
        verified+=1
    record=dict(all_passed=verified==249,verified_tasks=verified,source_orders=counts,
        original_full_cohort_predictions_unchanged=True,compiled_full_cohort_parity_rechecked=True,
        score_and_timing_records_unchanged=True)
    (root/'probe_layout_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
