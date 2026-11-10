import unittest
import numpy as np
from mean_improved_all import split_coordinates, replay_provenance, fit_mean_models, valid_result, coverage, reuse_diffusion_record, regular_origin_stride, designated_pilots
from location_scale import EmpiricalLocationScale, select_blend

class MeanAllTests(unittest.TestCase):
 def test_priority_replay_with_actual_batches_and_targets(self):
  from location_scale import replay_windows
  targets=np.arange(13*4*3,dtype=np.float64).reshape(13,4,3)
  sizes=[5,5,3];w,h,c=replay_windows(13,4,3,37,99,sizes)
  got=replay_provenance(targets,sizes,h,c,targets[w,h,c],37,99)
  np.testing.assert_array_equal(got,w)
  with self.assertRaises(AssertionError):replay_provenance(targets,[4,4,5],h,c,targets[w,h,c],37,99)
 def test_overlapping_short_calibration_uses_labeled_fallback(self):
  f,s,meta=split_coordinates(np.repeat(np.arange(100),300),100,192,1)
  self.assertIsNone(f);self.assertIsNone(s);self.assertEqual(meta['protocol'],'training forecasts fit; entire validation selects')
 def test_two_nonoverlapping_windows_split_without_fallback(self):
  f,s,meta=split_coordinates(np.repeat(np.arange(2),200),2,192,360)
  self.assertEqual((int(f.sum()),int(s.sum())),(200,200));self.assertEqual(meta['purged_windows'],0)
 def test_mean_only_matches_existing_protocol_and_preserves_shape(self):
  rng=np.random.default_rng(31);rows=rng.normal(size=(1800,100))+np.linspace(-2,2,1800)[:,None]
  y=.7*rows.mean(1)+.4+rng.normal(size=1800)*.7
  train=rows[:800];ty=y[:800];w=np.repeat(np.arange(9),200)
  models,meta=fit_mean_models({'horizon':1},rows,y,w,train,ty,9,1)
  f,s,_=split_coordinates(w,9,1,1)
  base=EmpiricalLocationScale.baseline(rows[f],y[f],train,ty,1)
  candidate=EmpiricalLocationScale.fit(rows[f],y[f],True,False,base)
  eta,_=select_blend(base.predict(rows[s]),candidate.predict(rows[s]),y[s])
  self.assertEqual(eta,models['FSPCMean'].weight)
  np.testing.assert_allclose(np.diff(models['FSPCRefit'].predict(rows),axis=1),np.diff(models['FSPCMean'].predict(rows),axis=1),atol=1e-14)
  self.assertFalse(meta['test_targets_used'])
 def test_completion_checks_full_population_and_methods(self):
  task={'id':'A'}
  r={'version':'fspc-mean-all-v1-20261110','task':task,'full_test_split':True,'backbone_retrained':False,'methods':{'Native':{},'FSPCRefit':{},'FSPCMean':{}}}
  self.assertFalse(valid_result(r,task))
  for m in r['methods'].values():m['coordinate_count']=100
  r.update(test_windows=2,test_batches=2,fit={'test_targets_used':False,'parameters_frozen_before_test':True})
  self.assertTrue(valid_result(r,task))
  r['methods']['FSPCMean']['coordinate_count']=99
  self.assertFalse(valid_result(r,task))
 def test_coverage_does_not_silently_drop_missing(self):
  tasks=[{'id':'a'},{'id':'b'},{'id':'c'}]
  out=coverage(tasks,{'a':True,'b':False})
  self.assertEqual(out['complete'],1);self.assertEqual(out['invalid'],['b']);self.assertEqual(out['missing'],['c'])

 def test_reuse_requires_exact_corrected_protocol_and_task(self):
  from location_scale import VERSION as OLD_VERSION
  task={'id':'D','backbone':'DiffusionTS'}
  names=['Native','FSPCBlocked','FSPCRefit','FSPCMean','FSPCScale','FSPCJoint','FSPCMeanBlocked','FSPCScaleBlocked','FSPCJointBlocked','EMOS','Qian2025SoftCRPS','DistributionRecalibration2022','QRF2016']
  r=dict(version=OLD_VERSION,task=task,mode='unclipped',full_test_split=True,external_baselines_refitted=True,
    test_windows=1,test_batches=1,checkpoint='frozen',methods={m:{'coordinate_count':100} for m in names},
    fit={'test_targets_used':False,'parameters_frozen_before_test':True,'selection':{'FSPCMean':{'weight':.6,'training':{'new_mean':True,'new_scale':False}}}})
  result=reuse_diffusion_record(r,task,'prior/results.json')
  self.assertTrue(valid_result(result,task));self.assertEqual(result['fit']['weight'],.6)
  r['mode']='original'
  with self.assertRaises(ValueError):reuse_diffusion_record(r,task,'wrong')

 def test_aggregation_requires_all_three_paired_runs_per_configuration(self):
  from aggregate_mean_improved_all import summarize_records
  tasks=[dict(id=str(i),backbone='B',dataset='D',horizon=1,seed=i) for i in [1,2,3]]
  def record(t):
   v=dict(coordinate_count=10,MSE=1.,CRPS=.4,QICE_percent=2.,state_coverage={'state_error':.03},events={'brier':[.1,.2,.3]})
   return dict(version='fspc-mean-all-v1-20261110',task=t,full_test_split=True,backbone_retrained=False,
     methods={m:dict(v) for m in ['Native','FSPCRefit','FSPCMean']},test_windows=1,test_batches=1,
     fit={'test_targets_used':False,'parameters_frozen_before_test':True,'weight':.5,'split':{'protocol':'chronological'}})
  records={t['id']:record(t) for t in tasks[:2]}
  out=summarize_records(tasks,records)
  self.assertEqual(out['coverage']['complete'],2);self.assertEqual(out['configurations'],[])
  records['3']=record(tasks[2]);out=summarize_records(tasks,records)
  self.assertEqual(len(out['configurations']),3)
  self.assertEqual(out['backbones'][0]['MSE_std'],0.)
  self.assertEqual(out['backbones'][0]['completed_configurations'],1)

 def test_k2_horizon_does_not_determine_rolling_stride(self):
  self.assertEqual(regular_origin_stride(np.array([1000,1096,1192,1288])),96)
  f,s,meta=split_coordinates(np.repeat(np.arange(28),200),28,192,regular_origin_stride(np.arange(28)*96))
  self.assertEqual(meta['purged_windows'],1)
  self.assertLessEqual(meta['fit_last_window']*96+191,meta['cut_window']*96-1)

 def test_fresh_queue_designates_each_backbone_pilot(self):
  tasks=[dict(id=f'{b}{i}',backbone=b) for b in ['K2VAE','PPM','DiffusionTS'] for i in [1,2]]
  pilots=designated_pilots(tasks,{})
  self.assertEqual({b:t['id'] for b,t in pilots.items()},{'K2VAE':'K2VAE1','PPM':'PPM1'})
  with self.assertRaises(ValueError):designated_pilots(tasks,{'K2VAE':{'task':'PPM1'}})
 def test_completed_queue_restart_exits_and_writes_final_status(self):
  import json,subprocess,sys,tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'source';source.mkdir();tasks=[]
   for config in range(83):
    for seed in [1,2,3]:
     t=dict(id=f'c{config}s{seed}',backbone='K2VAE',dataset=f'D{config}',horizon=1,seed=seed);tasks.append(t)
     path=root/'runs'/t['id'];path.mkdir(parents=True)
     r=dict(version='fspc-mean-all-v1-20261110',task=t,full_test_split=True,backbone_retrained=False,
      methods={m:{'coordinate_count':100} for m in ['Native','FSPCRefit','FSPCMean']},test_windows=1,test_batches=1,
      fit={'test_targets_used':False,'parameters_frozen_before_test':True})
     (path/'results.json').write_text(json.dumps(r))
   (source/'manifest.json').write_text(json.dumps(tasks))
   cmd=[sys.executable,str(Path(__file__).with_name('launch_mean_improved_all.py')),'--root',str(root),
    '--source',str(source),'--mean-root',str(root/'unused'),'--reuse-root',str(root/'unused')]
   result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
   self.assertEqual(result.returncode,0,result.stderr)
   self.assertEqual(json.loads((root/'status.json').read_text())['complete'],249)

if __name__=='__main__':unittest.main()
