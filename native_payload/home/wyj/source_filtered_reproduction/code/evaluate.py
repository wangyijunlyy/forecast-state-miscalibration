"""Reconstruct fitting sources and evaluate the fixed source-filtered method."""
import argparse
import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

OLD = Path('/home/wyj/three_seed_reproduction_20261031')
sys.path.insert(0, str(OLD / 'code'))
sys.path.insert(0, '/home/wyj/SPARC/K2VAE_dacu_k2vae_ltsf')
sys.path.insert(0, '/home/wyj/distribution_binned_diffusion/code')
from interfaces import prepare
from core import CommonMetrics, atomic
from reservoir import CalibrationReservoir
from sparc_adapters.posthoc_baselines import DistributionCalibrationCalibrator as Calibrator


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def equal(a, b):
    return (np.array_equal(a.mean_edges, b.mean_edges) and np.array_equal(a.scale_edges, b.scale_edges)
            and np.array_equal(a.global_residuals, b.global_residuals)
            and a.bin_residuals.keys() == b.bin_residuals.keys()
            and all(np.array_equal(a.bin_residuals[k], b.bin_residuals[k]) for k in a.bin_residuals))


def bounded(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    return values if len(values) <= 20000 else values[np.linspace(0, len(values)-1, 20000).astype(int)]


def fit(t, out, interface):
    seed(t['seed'] + 10000); reservoir = CalibrationReservoir(50000, t['seed']); windows = 0
    with torch.no_grad():
        for step, batch in enumerate(interface['train'], 1):
            if step > interface['fit_limit']: break
            x, y, _ = interface['predict'](batch, 'calibration', step)
            reservoir.update(x, y); windows += len(y)
            if step % 20 == 0: print('FIT', t['id'], step, flush=True)
    rows, y, h, _ = reservoir.arrays()
    mu = rows.mean(1); std = rows.std(1, ddof=1); sigma = std.clip(1e-6)
    z = (y-mu)/sigma; kept = std > 1.01e-6
    if not kept.any(): raise ValueError('No noncollapsed fitting sources: '+t['id'])
    with (Path(t['run_root'])/'calibrators.pkl').open('rb') as handle: previous = pickle.load(handle)
    models = {}; parity = {}
    for bins, nh in [(16, 1), (64, 4)]:
        raw = Calibrator.fit_summaries(z, mu, sigma, h, t['horizon'], n_mean_bins=4,
                  n_scale_bins=4, n_horizon_bins=nh, min_bin_size=128, max_pool_size=20000)
        parity[str(bins)] = equal(raw, previous[str(bins)+'_raw'])
        cal = copy.deepcopy(raw)
        if not kept.all():
            keys = np.stack([np.minimum(h*nh//t['horizon'], nh-1),
                np.searchsorted(cal.mean_edges[1:-1], mu, side='right'),
                np.searchsorted(cal.scale_edges[1:-1], np.log(sigma), side='right')], axis=1)
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            cal.bin_residuals = {tuple(int(v) for v in key): bounded(z[(inverse==i)&kept])
                for i, key in enumerate(unique) if np.sum((inverse==i)&kept) >= 128}
            cal.global_residuals = bounded(z[kept])
        models['DC'+str(bins)] = cal
    with (out/'calibrators.pkl').open('wb') as handle: pickle.dump(models, handle)
    meta = dict(fit_split='train', fit_windows=windows, retained_points=len(z), seen_points=reservoir.seen_points,
        retained_sources=int(kept.sum()), excluded_sources=int((~kept).sum()), source_scale_threshold=1.01e-6,
        original_raw_pool_exact_parity=parity, boundary_source='all retained native fitting coordinates',
        filtering_scope='residual sources only; every test coordinate retained', seed=t['seed']+10000,
        configurations={n:dict(n_horizon_bins=c.n_horizon_bins, local_pools=len(c.bin_residuals)) for n,c in models.items()})
    atomic(out/'fit_metadata.json', meta)
    return models, meta


def run(t, out):
    if (out/'results.json').exists(): return
    interface = prepare(t, out)
    if (out/'fit_metadata.json').exists():
        meta = json.loads((out/'fit_metadata.json').read_text())
        with (out/'calibrators.pkl').open('rb') as handle: models = pickle.load(handle)
    else: models, meta = fit(t, out, interface)
    old = json.loads((Path(t['run_root'])/'results.json').read_text())
    reusable = (meta['excluded_sources'] == 0 and all(meta['original_raw_pool_exact_parity'].values())
                and not any(old['mad_trigger'].values()))
    if reusable:
        result = {k:v for k,v in old.items() if k != 'mad_trigger'}
        result.update(source_filter=meta, evaluation_provenance='exact unchanged calibrator; original complete-test scores',
                      original_results=str(Path(t['run_root'])/'results.json'))
        atomic(out/'results.json', result)
        print('EXACT_REUSE', t['id'], flush=True); return
    names = ['Native','DC16','DC64']; common = {n:CommonMetrics() for n in names}
    native = {n:interface['native_factory']() for n in names}
    original = {n:CommonMetrics() for n in names} if t['backbone']=='K2VAE' else None
    seed(t['seed']+30000); windows=0; steps=0; start=time.monotonic()
    with torch.no_grad():
        for step,batch in enumerate(interface['test'],1):
            x,y,origin=interface['predict'](batch,'test',step)
            outputs={'Native':x, **{n:c.transform(x) for n,c in models.items()}}
            for name, values in outputs.items():
                common[name].update(values,y); interface['native_update'](native[name],values,y,origin)
                if original: original[name].update(interface['denormalize'](values),origin)
            windows+=len(y); steps=step
            if step%100==0: print('TEST',t['id'],step,round(time.monotonic()-start,1),flush=True)
    assert windows==old['test_windows'] and steps==old['test_batches']
    assert windows>0
    scores={n:a.result() for n,a in common.items()}
    parity={k: scores['Native'][k]-old['common_normalized']['Native'][k]
            for k in ('CRPS','MSE','QICE_percent')}
    atomic(out/'results.json', dict(task=t,checkpoint=interface['checkpoint'],full_test_split=True,
        test_windows=windows,test_batches=steps,common_normalized=scores,
        native_metrics={n:a.result() for n,a in native.items()},
        common_original_units={n:a.result() for n,a in original.items()} if original else None,
        source_filter=meta,test_seed=t['seed']+30000,total_seconds=time.monotonic()-start,
        evaluation_provenance='fresh paired complete native test iterator',native_score_deltas=parity))
    print('COMPLETE',t['id'],scores,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    run(json.loads(args.task.read_text()),args.out)
