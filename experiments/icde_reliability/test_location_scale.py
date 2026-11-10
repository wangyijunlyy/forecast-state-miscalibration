"""Critical-path mathematical and provenance tests for the new calibration."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
try:
    from location_scale import (variance_objective, select_blend, replay_windows,
        fit_location, EmpiricalLocationScale, chronological_split)
except ImportError:
    variance_objective = select_blend = replay_windows = None
    fit_location = EmpiricalLocationScale = chronological_split = None

try:
    from location_scale import complete_result, VERSION
except ImportError:
    complete_result=None
    VERSION='missing'


class LocationScaleTests(unittest.TestCase):
    def test_crps_variance_gradient_matches_independent_pairwise_definition(self):
        self.assertIsNotNone(variance_objective)
        z=np.array([[-1.5,-.5,.5,1.5],[-1.,-.2,.3,.9]])
        mu=np.array([0.,1.]); sigma2=np.array([.2,1.3]);y=np.array([.3,-.4]);p=np.array([.7,.4])
        loss,gradient=variance_objective(p,mu,sigma2,y,z)
        a=mu[:,None]+np.sqrt(p[0]*sigma2+p[1])[:,None]*z
        expected=np.mean(abs(a-y[:,None]))-.5*np.mean(abs(a[:,:,None]-a[:,None,:]))
        self.assertAlmostEqual(loss,expected,places=12)
        numeric=[]
        for j in range(2):
            d=np.zeros(2);d[j]=1e-6
            numeric.append((variance_objective(p+d,mu,sigma2,y,z)[0]-variance_objective(p-d,mu,sigma2,y,z)[0])/2e-6)
        np.testing.assert_allclose(gradient,numeric,atol=1e-7,rtol=1e-6)

    def test_selector_rejects_crps_gain_with_excess_mean_error(self):
        self.assertIsNotNone(select_blend)
        base=np.tile(np.linspace(-5,5,100),(6,1));candidate=np.ones_like(base)
        weight,meta=select_blend(base,candidate,np.zeros(6))
        self.assertLess(weight,1e-6)
        self.assertLessEqual(meta['selected_mse'],meta['reference_mse']+1e-12)
        # A spread correction with the same mean must remain available.
        weight,meta=select_blend(base,np.zeros_like(base),np.zeros(6))
        self.assertAlmostEqual(weight,1.,places=6)
        self.assertLess(meta['selected_crps'],meta['reference_crps'])

    def test_affine_mean_can_correct_slope_and_recover_unseen_sparse_states(self):
        self.assertIsNotNone(fit_location)
        mu=np.linspace(-2,2,400);g=np.zeros(400,dtype=int)
        p=fit_location(mu,2+3*mu,g,n_states=3)
        np.testing.assert_allclose(p['intercept']+p['slope']*np.array([-4.,4.])+p['state_shift'][[1,2]],[-10.,14.],atol=1e-10)

    def test_collapsed_ensembles_recover_spread_and_preserve_sorted_output_mean(self):
        self.assertIsNotNone(EmpiricalLocationScale)
        rows=np.zeros((600,100));y=np.tile(np.linspace(-2,2,100),6)
        model=EmpiricalLocationScale.fit(rows,y,new_mean=True,new_scale=True)
        out=model.predict(np.zeros((3,100)))
        self.assertGreater(float(out.std(1).min()),.7)
        self.assertTrue(np.all(np.diff(out,axis=1)>=0))
        np.testing.assert_allclose(out.mean(1),model.location(np.zeros((3,100))),atol=1e-12)
        self.assertEqual(model.metadata['source_rows_used'],600)

    def test_priority_replay_matches_independent_full_priority_selection(self):
        self.assertIsNotNone(replay_windows)
        n,h,c,capacity,seed=5,3,2,7,23
        windows,hh,cc=replay_windows(n,h,c,capacity,seed)
        priorities=np.random.default_rng(seed).random(n*h*c)
        # Exact retained set; order is checked by attached channel/horizon IDs.
        actual=set((windows*h*c+hh*c+cc).tolist())
        expected=set(np.argsort(priorities)[-capacity:].tolist())
        self.assertEqual(actual,expected)
        windows,hh,cc=replay_windows(2,2,2,100,1)
        np.testing.assert_array_equal(windows,[0,0,0,0,1,1,1,1])
        np.testing.assert_array_equal(hh,[0,0,1,1,0,0,1,1])
        np.testing.assert_array_equal(cc,[0,1,0,1,0,1,0,1])

    def test_chronological_split_purges_shared_targets_and_keeps_window_groups(self):
        self.assertIsNotNone(chronological_split)
        window=np.repeat(np.arange(12),2)
        fit,selection,meta=chronological_split(window,12,horizon=4,stride=1)
        self.assertLess(int(window[fit].max())+4,int(window[selection].min())+1)
        self.assertEqual(set(window[fit])&set(window[selection]),set())
        self.assertEqual(int(fit.sum())%2,0)
        self.assertGreater(meta['purged_windows'],0)

    def test_two_nonoverlapping_validation_windows_use_independent_fit_and_selection(self):
        from run_location_scale import fit_models
        rng=np.random.default_rng(7)
        rows=rng.normal(size=(600,100));targets=rows.mean(1)+rng.normal(size=600)
        with tempfile.TemporaryDirectory() as directory:
            _,meta=fit_models(dict(id='two_windows',horizon=1),Path(directory),rows,targets,
                np.repeat([0,1],300),rows,targets,2,2)
        self.assertTrue(meta['split']['protocol'].startswith('chronological'))
        self.assertEqual(meta['split']['fit_coordinates'],300)
        self.assertEqual(meta['split']['selection_coordinates'],300)

    def test_completion_rejects_missing_required_external_methods(self):
        self.assertIsNotNone(complete_result)
        r=dict(version=VERSION,task=dict(id='x'),mode='unclipped',full_test_split=True,
            external_baselines_refitted=False,methods={n:{} for n in ['Native','FSPCBlocked','FSPCRefit','FSPCMean','FSPCScale','FSPCJoint',
                'FSPCMeanBlocked','FSPCScaleBlocked','FSPCJointBlocked']})
        self.assertFalse(complete_result(r,r['task'],'unclipped',external=True))
        r['external_baselines_refitted']=True
        self.assertFalse(complete_result(r,r['task'],'unclipped',external=True))
        r['methods'].update({n:{} for n in ['EMOS','Qian2025SoftCRPS','DistributionRecalibration2022','QRF2016']})
        self.assertTrue(complete_result(r,r['task'],'unclipped',external=True))

    def test_aggregation_reports_missing_jobs_instead_of_silent_subset_mean(self):
        import json
        from aggregate_location_scale import aggregate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            jobs=[dict(task=dict(id='a',dataset='A',seed=1),mode='original',external=False),
                  dict(task=dict(id='b',dataset='A',seed=2),mode='original',external=False)]
            (root/'manifest.json').write_text(json.dumps(jobs))
            aggregate(root)
            summary=json.loads((root/'analysis/summary.json').read_text())
        self.assertEqual(summary.get('expected_jobs'),2)
        self.assertEqual(summary.get('missing_jobs'),2)
        self.assertEqual(summary.get('comparison_status'),'provisional')

    def test_full_extension_keeps_all_original_jobs_and_refits_every_unclipped_baseline(self):
        import launch_location_scale as module
        function=getattr(module,'campaign_jobs',None)
        self.assertIsNotNone(function)
        tasks=[dict(id='a',backbone='DiffusionTS',dataset='ETTm2',seed=1),
               dict(id='b',backbone='DiffusionTS',dataset='SolarEnergy',seed=1),
               dict(id='c',backbone='Other',dataset='SolarEnergy',seed=1)]
        initial=function(tasks,'priority');full=function(tasks,'all')
        self.assertEqual({(j['task']['id'],j['mode']) for j in initial},{('a','original'),('a','unclipped'),('b','original')})
        self.assertEqual({(j['task']['id'],j['mode']) for j in full},{('a','original'),('a','unclipped'),('b','original'),('b','unclipped')})
        self.assertTrue(all(j['external'] for j in full if j['mode']=='unclipped'))


if __name__=='__main__':unittest.main()
