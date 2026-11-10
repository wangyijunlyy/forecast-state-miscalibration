"""Recompute state-assignment and matched-moment controls for source-filtered DC16.

Uses the exact cached fitting/evaluation coordinates of the existing auxiliary
experiment. Historical DC64 controls remain in their original records.
"""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, '/home/wyj/source_filtered_reproduction/code')
from auxiliary import CACHE, OLD, ROOT, fit, load
from intervention_stats import score, state_keys, transform
from scipy.special import ndtri

DEST = Path('/home/wyj/source_filtered_reproduction/dc16_controls')


def run(task, context):
    previous = json.loads((ROOT / task['id'] / context / 'results.json').read_text())
    fitting_path = (OLD / task['id'] / 'train_coordinates.npz'
                    if context == 'train_to_test'
                    else CACHE / task['id'] / 'validation_coordinates.npz')
    evaluation_path = CACHE / task['id'] / 'test_coordinates.npz'
    fitting, evaluation = load(fitting_path), load(evaluation_path)
    rows, targets, horizon = (evaluation[k] for k in ['rows', 'targets', 'horizon'])
    noncollapsed = rows.std(1, ddof=1) > 1.01e-6
    scores, models = {}, {}
    args = [fitting['rows'], fitting['targets'], fitting['horizon'], task['horizon']]
    dc = fit(*args, bins=(1, 4, 4))
    keys = state_keys(dc.cal, rows.mean(1), rows.std(1, ddof=1).clip(1e-6), horizon)

    def evaluate(name, values):
        scores[name], _ = score(values, targets, keys, noncollapsed)

    evaluate('Native', rows)
    output = transform(dc, rows, horizon)
    evaluate('DC16', output)
    models['DC16'] = dc.meta
    moments = transform(dc, rows, horizon, mode='location_scale')
    mean_error = float(np.max(abs(moments[noncollapsed].mean(1) - output[noncollapsed].mean(1))))
    scale_error = float(np.max(abs(moments[noncollapsed].std(1, ddof=1) - output[noncollapsed].std(1, ddof=1))))
    assert max(mean_error, scale_error) < 1e-10
    evaluate('LocationScaleDC16', moments)
    for repeat in range(5):
        model = fit(*args, bins=(1, 4, 4),
                    shuffle_seed=task['sampling_seed'] + 60000 + repeat)
        name = 'ShuffledDC16_' + str(repeat)
        evaluate(name, transform(model, rows, horizon))
        models[name] = model.meta
    for name, bins in [('GlobalTemplate', (1, 1, 1)), ('Scale4', (1, 1, 4)), ('DC64', (4, 4, 4))]:
        model = fit(*args, bins=bins)
        evaluate(name, transform(model, rows, horizon))
        models[name] = model.meta
    gaussian = previous['models']['GaussianVariance2P']
    sigma = rows.std(1, ddof=1).clip(1e-6)
    scale = np.sqrt(np.maximum(gaussian['variance_floor'] + gaussian['variance_multiplier'] * sigma**2, 1e-12))
    evaluate('GaussianVariance2P', rows.mean(1)[:, None] + scale[:, None] * ndtri((np.arange(100) + .5) / 100))
    # The primary DC16 and Native results must reproduce their existing records.
    for method in ['Native', 'DC16', 'DC64', 'Scale4', 'GlobalTemplate', 'GaussianVariance2P']:
        assert abs(scores[method]['crps'] - previous['scores'][method]['crps']) < 1e-12
    result = dict(task=task, context=context, scores=scores, models=models,
                  fitting_coordinates=str(fitting_path), evaluation_coordinates=str(evaluation_path),
                  fit_points=len(fitting['rows']), evaluation_points=len(rows),
                  noncollapsed_evaluation_points=int(noncollapsed.sum()),
                  partition_order=['horizon', 'mean', 'scale'], bins=[1, 4, 4],
                  fixed_diagnostic_partition=[1, 4, 4],
                  source_filter_threshold=1.01e-6,
                  mean_matching_max_error=mean_error, scale_matching_max_error=scale_error,
                  native_and_dc16_parity=True,
                  global_template_crps=previous['scores']['GlobalTemplate']['crps'])
    dest = DEST / task['id'] / context
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('COMPLETE', task['id'], context, flush=True)


if __name__ == '__main__':
    tasks = json.loads((OLD / 'manifest.json').read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        jobs = {executor.submit(run, task, context): (task['id'], context)
                for task in tasks for context in ['train_to_test', 'validation_to_test']}
        for future in as_completed(jobs):
            try:
                future.result()
            except Exception as exc:
                failures.append(dict(task=jobs[future], error=repr(exc)))
    records = list(DEST.glob('*/*/results.json'))
    audit = dict(complete=len(records) == 32 and not failures,
                 expected_contexts=32, complete_contexts=len(records), failures=failures,
                 main_method='source-filtered DC16', bins=[1, 4, 4])
    (DEST / 'completion_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit), flush=True)
    if not audit['complete']:
        raise SystemExit(1)
