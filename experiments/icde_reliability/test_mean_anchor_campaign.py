"""Scope and immutable pilot-reuse checks for the full campaign."""
import copy,json,unittest
from pathlib import Path
try:
 from mean_anchor_campaign import validate_tasks,validate_record,version_for
except ImportError:
 validate_tasks=validate_record=version_for=None

class CampaignTests(unittest.TestCase):
 def test_current_registry_requires_complete_three_seed_configurations(self):
  self.assertIsNotNone(validate_tasks,'Campaign scope validation is not implemented')
  path=Path(__file__).parent/'results/fsdc36/manifest.json';tasks=json.loads(path.read_text())
  summary=validate_tasks(tasks);self.assertEqual(summary['tasks'],249);self.assertEqual(summary['configurations'],83)
  with self.assertRaises(ValueError):validate_tasks(tasks[:-1])
  duplicate=copy.deepcopy(tasks);duplicate[-1]=duplicate[0]
  with self.assertRaises(ValueError):validate_tasks(duplicate)
  wrong_horizon=copy.deepcopy(tasks)
  for task in wrong_horizon:
   if task['backbone']=='NsDiff':task['horizon']=96
  with self.assertRaises(ValueError):validate_tasks(wrong_horizon,reference=tasks)
 def test_pilot_reuse_rejects_relabelling_and_parameter_changes(self):
  self.assertIsNotNone(validate_record,'Safe pilot reuse is not implemented')
  results=Path(__file__).parent/'results'
  pilot=results/'mean_anchor_diffusionts/runs'
  p=next(pilot.glob('*/results.json')) if pilot.exists() else next(
   (results/'mean_anchor_all/runs').glob('DiffusionTS*/results.json'))
  record=json.loads(p.read_text());validate_record(record,record['task'])
  altered=copy.deepcopy(record);altered['fit']['pseudo_count']=64
  with self.assertRaises(ValueError):validate_record(altered,record['task'])
  other=copy.deepcopy(record);other['task']['backbone']='DeepSTUQ';other['fit']['task']['backbone']='DeepSTUQ'
  with self.assertRaises(ValueError):validate_record(other,other['task'])
  from mean_anchor_campaign import source_proof_matches
  tasks=[record['task']];fits={record['task']['id']:record['fit']}
  proof=dict(all_passed=True,tasks=1,manifest=tasks,fits=copy.deepcopy(fits))
  self.assertTrue(source_proof_matches(proof,tasks,fits))
  changed=copy.deepcopy(fits);changed[record['task']['id']]['state_bias'][0]+=.5
  self.assertFalse(source_proof_matches(proof,tasks,changed))
 def test_ratios_weight_backbones_equally_and_pair_each_seed(self):
  from aggregate_mean_anchor_all import normalized_ratios
  rows=[]
  for backbone,scale in [('small',1.),('large',10000.)]:
   for seed in [1,2,3]:
    for i in range(1 if backbone=='small' else 2):
     for name,factor in [('Native',1.),('candidate',.5 if backbone=='small' else 1.)]:
      rows.append(dict(backbone=backbone,dataset=str(i),horizon=1,seed=seed,
                       method=name,MSE=scale*seed*factor))
  summary,paired=normalized_ratios(rows,['MSE'])
  result=next(r for r in summary if r['method']=='candidate')
  self.assertAlmostEqual(result['MSE_mean'],.75)
  self.assertAlmostEqual(result['MSE_std'],0.)
  self.assertEqual(len(paired),12)

if __name__=='__main__':unittest.main()
