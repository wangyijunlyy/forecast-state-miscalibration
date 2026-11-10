"""Select and export a measured source-amplification example for 36 states."""
import argparse
import json
from pathlib import Path
import numpy as np
from granularity import fit_partition

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
tasks=[t for t in json.loads((a.root/'manifest.json').read_text()) if t['backbone']=='PPM']
def fit(task,filtered):
    source=Path(task['source36_root'])/'runs'/task['id']
    with np.load(source/'calibration.npz') as d:
        rows,y,h=[d[k] for k in ['rows','targets','horizon']]
    cal,_=fit_partition(rows,y,h,task['horizon'],6,1,source_filter=filtered)
    return cal
task=max(tasks,key=lambda t:float(np.max(abs(fit(t,False).compile().quantiles))))
source=Path(task['source36_root'])/'runs'/task['id'];output=a.root/'figure_data';output.mkdir(exist_ok=True)
with np.load(source/'calibration.npz') as d:fitting,y,h=[d[k] for k in ['rows','targets','horizon']]
mean=fitting.mean(1);raw=fitting.std(1,ddof=1);scale=raw.clip(1e-6)
residual=(y-mean)/scale;near=raw<=1.01e-6;extreme=abs(residual)>100
filtered=fit(task,True).compile();unfiltered=fit(task,False).compile()
with np.load(source/'test_reservoir.npz') as d:rows,targets,horizon,channels=[d[k] for k in ['rows','targets','horizon','channel']]
f=filtered.transform_rows(rows,horizon);u=unfiltered.transform_rows(rows,horizon)
mu=rows.mean(1);sigma=rows.std(1,ddof=1).clip(1e-6)
eligible=(sigma>=.01)&(abs(mu)<10)
index=int(np.argmax(np.where(eligible,np.max(abs(u),axis=1),-np.inf)))
np.savez_compressed(output/'source_case.npz',raw_scale=raw,scale=scale,residual=residual,
    probabilities=(np.arange(100)+.5)/100,native=np.sort(rows[index]),filtered=f[index],unfiltered=u[index])
record=dict(task=task['id'],mean_bins=6,scale_bins=6,
    task_selection='largest unfiltered fitting-template absolute quantile among PPM validation fits; targets of test forecasts unused',
    fitting_segment='validation',fitting_coordinates=len(y),near_floor_sources=int(near.sum()),
    near_floor_percent=float(100*near.mean()),extreme_sources=int(extreme.sum()),
    near_floor_extreme_sources=int((near&extreme).sum()),
    near_floor_extreme_percent=float(100*(near&extreme).sum()/extreme.sum()) if extreme.any() else None,
    residual_min=float(residual.min()),residual_max=float(residual.max()),
    test_retained_coordinates=len(rows),coordinate_index=index,horizon_id=int(horizon[index]),channel_id=int(channels[index]),
    native_mean=float(mu[index]),native_scale=float(sigma[index]),target=float(targets[index]),
    unfiltered_max_absolute=float(abs(u[index]).max()),filtered_max_absolute=float(abs(f[index]).max()),
    selection='largest absolute unfiltered forecast with native scale >=0.01 and |mean|<10; test targets unused')
(output/'source_case.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
