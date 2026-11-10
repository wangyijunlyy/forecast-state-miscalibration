import unittest
import torch
from timediff_batching import sample_timediff_batched


class WindowIndependentModel:
    def __call__(self,x,xm,y,ym,*unused):
        return x[:,None,-2:,:],None,None,None,None


class TimeDiffBatchingTests(unittest.TestCase):
    def test_keeps_each_window_separate_across_sample_chunks(self):
        x=torch.tensor([[[1.],[2.],[3.]],[[10.],[20.],[30.]]])
        result=sample_timediff_batched(WindowIndependentModel(),x,x,x,x,5,2)
        self.assertEqual(tuple(result.shape),(2,5,2,1))
        torch.testing.assert_close(result[0],torch.tensor([[[2.],[3.]]]).repeat(5,1,1))
        torch.testing.assert_close(result[1],torch.tensor([[[20.],[30.]]]).repeat(5,1,1))


if __name__=='__main__':unittest.main()
