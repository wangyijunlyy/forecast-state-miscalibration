import unittest
import numpy as np
from fspc_unified import fit_unified,UnifiedMean,complete_diffusion
from mean_improved_all import fit_mean_models

class UnifiedTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(76)
  self.x=rng.normal(size=(1800,100))+np.linspace(-2,2,1800)[:,None]
  self.y=.7*self.x.mean(1)+.2+rng.normal(size=1800)*.7
  self.h=np.arange(1800)%4;self.w=np.repeat(np.arange(9),200)
 def test_default_exactly_matches_completed_mean_method(self):
  old,meta=fit_mean_models({'horizon':4},self.x,self.y,self.w,self.x[:800],self.y[:800],9,8)
  new,fit=fit_unified(self.x[:800],self.y[:800],self.h[:800],self.x,self.y,self.h,self.w,9,8,4)
  np.testing.assert_allclose(new.predict(self.x,self.h),old['FSPCMean'].predict(self.x),atol=1e-13,rtol=1e-13)
  self.assertAlmostEqual(fit['weight'],meta['weight'],places=12)
 def test_partition_extension_and_training_templates_keep_shape(self):
  for split in ['validation','train']:
   model,fit=fit_unified(self.x[:800],self.y[:800],self.h[:800],self.x,self.y,self.h,self.w,9,8,4,bins=3,horizon_groups=2,template_split=split)
   self.assertEqual(len(model.bias),18)
   q=model.predict(self.x,self.h);base=model.predict(self.x,self.h,weight=0.)
   np.testing.assert_allclose(np.diff(q,axis=1),np.diff(base,axis=1),atol=1e-14)
   self.assertFalse(fit['test_targets_used']);self.assertEqual(fit['template_split'],split)
 def test_serialized_state_reconstructs_same_forecast(self):
  model,_=fit_unified(self.x[:800],self.y[:800],self.h[:800],self.x,self.y,self.h,self.w,9,8,4)
  import json
  saved=json.loads(json.dumps(model.to_record()))
  restored=UnifiedMean.from_record(saved)
  np.testing.assert_array_equal(model.predict(self.x,self.h),restored.predict(self.x,self.h))
 def test_corrected_internal_controls_require_uncentered_default(self):
  from fspc_unified import VERSION
  r={'version':VERSION,'full_test_split':True,'original_corrected_scores_reproduced':True,'methods':{m:{} for m in ['Native','FSPCMean','GlobalDC','GlobalPIT','ConditionalPIT','DC36']}}
  self.assertTrue(complete_diffusion(r));r['methods'].pop('DC36');self.assertFalse(complete_diffusion(r))

 def test_partial_internal_record_refits_only_missing_comparators(self):
  from fspc_unified import missing_internal_controls
  self.assertEqual(missing_internal_controls(None),['GlobalDC','GlobalPIT','ConditionalPIT'])
  partial={'methods':{'Native':{},'GlobalDC':{},'DC36':{}}}
  self.assertEqual(missing_internal_controls(partial),['GlobalPIT','ConditionalPIT'])
  complete={'methods':dict(partial['methods'],GlobalPIT={},ConditionalPIT={})}
  self.assertEqual(missing_internal_controls(complete),[])

 def test_interrupted_npz_write_preserves_existing_complete_cache(self):
  import tempfile
  from pathlib import Path
  from unittest.mock import patch
  import fspc_unified
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'cache.npz';np.savez_compressed(path,x=np.array([7]))
   def interrupted(target,**arrays):
    Path(target).write_bytes(b'partial zip');raise RuntimeError('interrupted producer')
   with patch.object(fspc_unified.np,'savez_compressed',side_effect=interrupted):
    with self.assertRaises(RuntimeError):fspc_unified.atomic_npz(path,x=np.array([9]))
   with np.load(path) as d:self.assertEqual(d['x'][0],7)

 def test_external_table_rounding_matches_canonical_half_up(self):
  # Use the shared canonical table formatter.
  from render_tables import cell
  self.assertEqual(cell(2.9625,.0045,False),r'\stat{2.963}{0.005}')

if __name__=='__main__':unittest.main()
