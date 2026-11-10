"""Stream measured source/forecast arrays from existing caches; no model inference."""
import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
import numpy as np


def crps(rows,target):
    s=rows.shape[1];weights=(2*np.arange(1,s+1)-s-1)/(s*s)
    return float(np.mean(np.abs(rows-target[:,None]))-np.mean(np.sort(rows,axis=1)@weights))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--mean-bins',type=int,choices=[4,6],help='Require the intended source-record partition')
    args=parser.parse_args()
    sys.path.insert(0,str(args.root/'code'))
    from portable_dc import EmpiricalCalibrator
    from granularity import fit_partition
    cases=json.loads(args.cases.read_text())
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
        for case in cases:
            identity=case['id']
            candidates=[args.root/'runs'/identity,args.root/'deepstuq_backbone/runs'/identity,
                        args.root.parent/'nsdiff_h192_correction_20261102/runs'/identity]
            source=next((p for p in candidates if (p/'calibration.npz').exists()),None)
            if source is None:raise FileNotFoundError(identity)
            record=json.loads((source/'results.json').read_text())
            assert record['task']['id']==identity and record['task']['horizon']==case['horizon']
            with np.load(source/'calibration.npz') as d:
                fitting=np.asarray(d['rows'],dtype=float);y=np.asarray(d['targets'],dtype=float);h=d['horizon']
            mu=fitting.mean(1);raw=fitting.std(1,ddof=1);z=(y-mu)/raw.clip(1e-6);near=raw<=1.01e-6
            bins=record.get('mean_bins',4)
            if args.mean_bins is not None and bins != args.mean_bins:
                raise ValueError('Source case partition differs from the requested default: '+identity)
            if bins == 6:
                if record['scale_bins']!=6 or record['horizon_bins']!=1 or record['version']!='icde-fsdc36-v1-20261103':
                    raise ValueError('Source case is not the canonical 36-state record: '+identity)
                filtered,_=fit_partition(fitting,y,h,case['horizon'],6,1)
                unfiltered,_=fit_partition(fitting,y,h,case['horizon'],6,1,source_filter=False)
            else:
                filtered=EmpiricalCalibrator.fit_rows(fitting,y,h,case['horizon'])
                unfiltered=EmpiricalCalibrator.fit_rows(fitting,y,h,case['horizon'],source_filter=False)
            with np.load(source/'test_reservoir.npz') as d:
                rows=np.asarray(d['rows'],dtype=float);targets=np.asarray(d['targets'],dtype=float)
                horizon=d['horizon'];channels=d['channel']
            a=unfiltered.compile().transform_rows(rows,horizon);b=filtered.compile().transform_rows(rows,horizon)
            control_path=args.root/'cohort/runs'/identity/'results.json' if bins==6 else source/'controls.json'
            controls=json.loads(control_path.read_text())
            errors={}
            for name,values,key in [('Native',rows,'Native'),('FSDC',b,'b6_h1_n50000' if bins==6 else 'Budget50000'),('Unfiltered',a,'UnfilteredDC36' if bins==6 else 'UnfilteredDC16')]:
                if key=='Native':continue
                observed=crps(values,targets);reference=controls['methods'][key]['CRPS']
                errors[name]=abs(observed-reference)
                if not np.isclose(observed,reference,rtol=1e-7,atol=1e-8):
                    raise ValueError('Cached reconstruction does not match recorded control: '+identity+'/'+name+': '+str((observed,reference)))
            center=rows.mean(1);scale=rows.std(1,ddof=1)
            eligible=scale>=max(float(np.quantile(scale,.25)),1e-6)
            amplification=np.max(abs(a-center[:,None]),axis=1)/np.maximum(np.max(abs(rows-center[:,None]),axis=1),1e-6)
            index=int(np.argmax(np.where(eligible,amplification,-np.inf)))
            if not near.any():assert np.array_equal(a,b)
            buffer=io.BytesIO()
            np.savez_compressed(buffer,raw_scale=raw,residual=z,probabilities=(np.arange(100)+.5)/100,
                                native=np.sort(rows[index]),filtered=b[index],unfiltered=a[index])
            metadata=dict(case,mean_bins=bins,scale_bins=bins,default_states=bins*bins,
                          source_version=record['version'],fitting_coordinates=len(y),near_floor_sources=int(near.sum()),
                          extreme_sources=int((abs(z)>100).sum()),near_floor_extreme_sources=int((near&(abs(z)>100)).sum()),
                          retained_test_coordinates=len(rows),coordinate_index=index,target=float(targets[index]),
                          native_mean=float(center[index]),native_scale=float(scale[index]),
                          horizon_id=int(horizon[index]),channel_id=int(channels[index]),
                          amplification_factor=float(amplification[index]),
                          paired_metric_absolute_differences=errors,
                          filtered_equals_unfiltered=bool(np.array_equal(a,b)),
                          coordinate_selection='maximum unfiltered/native deviation ratio among forecasts with scale at least the test-cohort 25th percentile; no targets')
            for name,data in [(identity+'.npz',buffer.getvalue()),(identity+'.json',json.dumps(metadata,indent=2).encode())]:
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
            print('Extracted '+identity,file=sys.stderr,flush=True)


if __name__=='__main__':main()
