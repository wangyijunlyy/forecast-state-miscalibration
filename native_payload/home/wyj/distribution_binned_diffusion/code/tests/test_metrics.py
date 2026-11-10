import unittest
import numpy as np
from metrics import ScoreAccumulator


class NativeMetricTests(unittest.TestCase):
    def test_crps_is_empirical_sample_score_and_folds_qice_tails(self):
        # Two samples {0,2}; CRPS(y=1)=.5 and CRPS(y=3)=1.5.
        samples=np.array([[[[0.]],[[2.]]],[[[0.]],[[2.]]]])
        acc=ScoreAccumulator();acc.update(samples,np.array([[[1.]],[[3.]]]))
        self.assertAlmostEqual(acc.result()['CRPS'],1.)
        # y=1 is in the middle, y=3 belongs to the outermost bin.
        self.assertAlmostEqual(acc.result()['QICE'],.16)

    def test_qice_counts_both_outside_tails_in_outer_bins(self):
        samples=np.broadcast_to(np.arange(100.)[None,:,None,None],(2,100,1,1))
        acc=ScoreAccumulator();acc.update(samples,np.array([[[-10.]],[[110.]]]))
        self.assertAlmostEqual(acc.result()['QICE'],.16)
        self.assertEqual(acc.result()['coordinate_count'],2)

    def test_uniform_decile_membership_has_zero_qice(self):
        samples=np.broadcast_to(np.arange(100.)[None,:,None,None],(10,100,1,1))
        acc=ScoreAccumulator();acc.update(samples,np.arange(5.,100.,10).reshape(10,1,1))
        self.assertAlmostEqual(acc.result()['QICE'],0.)

    def test_batch_partition_does_not_change_metric_weights(self):
        samples=np.broadcast_to(np.arange(100.)[None,:,None,None],(3,100,1,1))
        target=np.array([-1.,50.,110.]).reshape(3,1,1)
        full=ScoreAccumulator();full.update(samples,target)
        split=ScoreAccumulator();split.update(samples[:2],target[:2]);split.update(samples[2:],target[2:])
        self.assertAlmostEqual(full.result()['CRPS'],split.result()['CRPS'])
        self.assertAlmostEqual(full.result()['QICE'],split.result()['QICE'])


if __name__=='__main__':unittest.main()
