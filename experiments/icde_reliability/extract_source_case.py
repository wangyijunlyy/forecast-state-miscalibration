"""Measured source-amplification case from the primary experiment, without refitting a backbone."""
import json
import os
from pathlib import Path
import numpy as np
from portable_dc import EmpiricalCalibrator

ROOT = Path(os.environ.get('ICDE_RUN_ROOT', '/home/wyj/icde_reliability_20261102'))
manifest = json.loads((ROOT/'manifest.json').read_text())
def template_tail(task):
    with np.load(ROOT/'runs'/task['id']/'calibration.npz') as data:
        cal = EmpiricalCalibrator.fit_rows(data['rows'], data['targets'], data['horizon'], task['horizon'], source_filter=False)
    return float(np.max(abs(cal.compile().quantiles))), task['id']
# An illustrative source-failure case selected using fitting data alone.
IDENTITY = max((t for t in manifest if t['backbone'] == 'PPM'), key=template_tail)['id']
source = ROOT/'runs'/IDENTITY
output = ROOT/'figure_data'; output.mkdir(exist_ok=True)
with np.load(source/'calibration.npz') as d:
    fitting, y, h = (d[k] for k in ['rows', 'targets', 'horizon'])
mean = fitting.mean(1); raw = fitting.std(1, ddof=1); scale = raw.clip(1e-6)
residual = (y-mean)/scale; near = raw <= 1.01e-6; extreme = abs(residual) > 100
task = next(t for t in manifest if t['id'] == IDENTITY)
filtered = EmpiricalCalibrator.fit_rows(fitting, y, h, task['horizon'])
unfiltered = EmpiricalCalibrator.fit_rows(fitting, y, h, task['horizon'], source_filter=False)
with np.load(source/'test_reservoir.npz') as d:
    rows, targets, horizon, channels = (d[k] for k in ['rows', 'targets', 'horizon', 'channel'])
a = unfiltered.compile().transform_rows(rows, horizon)
b = filtered.compile().transform_rows(rows, horizon)
mu = rows.mean(1); sigma = rows.std(1, ddof=1).clip(1e-6)
# Selection depends on forecasts, never on the corresponding test target.
eligible = (sigma >= .01)&(abs(mu) < 10)
index = int(np.argmax(np.where(eligible, np.max(abs(a), axis=1), -np.inf)))
np.savez_compressed(output/'source_case.npz', raw_scale=raw, scale=scale, residual=residual,
    probabilities=(np.arange(100)+.5)/100, native=np.sort(rows[index]),
    filtered=b[index], unfiltered=a[index])
record = dict(task=IDENTITY, task_selection='largest absolute unfiltered midpoint residual-template quantile among PPM validation fits; no test targets',
    fitting_segment='validation', fitting_coordinates=len(y),
    near_floor_sources=int(near.sum()), near_floor_percent=100*near.mean(), extreme_sources=int(extreme.sum()),
    near_floor_extreme_sources=int((near&extreme).sum()),
    near_floor_extreme_percent=float(100*(near&extreme).sum()/extreme.sum()) if extreme.any() else None,
    residual_min=float(residual.min()), residual_max=float(residual.max()),
    test_retained_coordinates=len(rows), coordinate_index=index, horizon_id=int(horizon[index]), channel_id=int(channels[index]),
    selection='largest absolute unfiltered reconstruction among retained forecasts with native scale >=0.01 and |mean|<10; test targets unused',
    native_mean=float(mu[index]), native_scale=float(sigma[index]), target=float(targets[index]),
    unfiltered_max_absolute=float(abs(a[index]).max()), filtered_max_absolute=float(abs(b[index]).max()))
(output/'source_case.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
