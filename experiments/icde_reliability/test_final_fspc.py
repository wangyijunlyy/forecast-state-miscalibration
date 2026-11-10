"""Critical numerical contracts of the full FSPC factorial/deployment study."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from portable_dc import CompiledCalibrator, summaries
from fspc_unified import UnifiedMean
from final_fspc import CompiledFSPC, factorial_forecasts, save_parity_probe


def fixture(weight=.6):
    rng=np.random.default_rng(17)
    rows=rng.normal(size=(73,100))*rng.uniform(.05,3,(73,1))+rng.normal(size=(73,1))
    rows[0]=2.;rows[1]=0.
    edge=np.array([-np.inf,-2.,-1.,0.,1.,2.,np.inf])
    grid=(np.arange(100)+.5)/100
    q=np.array([3*grid-1,4*grid**2-1,.3+np.log(grid/(1-grid))])
    cal=CompiledCalibrator(edge,edge,q,np.arange(36)%3,192,1)
    loc=dict(intercept=.23,slope=.87,state_shift=np.linspace(-.3,.4,36),counts=np.arange(36))
    model=UnifiedMean(cal,np.linspace(-.5,.7,36),.4,loc,weight)
    return model,rows,np.arange(len(rows))%192


class FinalFSPCTest(unittest.TestCase):
    def test_four_arms_isolate_mean_and_shape(self):
        model,rows,h=fixture();arms=factorial_forecasts(model,rows,h)
        self.assertEqual(set(arms),{'Native','MeanOnly','ShapeOnly','FSPC'})
        np.testing.assert_array_equal(arms['Native'],rows)
        np.testing.assert_array_equal(arms['FSPC'],model.predict(rows,h))
        np.testing.assert_allclose(arms['ShapeOnly'].mean(1),rows.mean(1),atol=1e-13)
        np.testing.assert_allclose(arms['MeanOnly'].mean(1),arms['FSPC'].mean(1),atol=1e-13)
        np.testing.assert_allclose(np.diff(arms['MeanOnly'],axis=1),np.diff(rows,axis=1),atol=1e-13)
        np.testing.assert_allclose(arms['FSPC']-arms['FSPC'].mean(1,keepdims=True),
                                   arms['ShapeOnly']-arms['ShapeOnly'].mean(1,keepdims=True),atol=1e-13)

    def test_full_reference_is_bitwise_preserved_at_blend_endpoints(self):
        for weight in [0.,.6,1.]:
            model,rows,h=fixture(weight);compiled=CompiledFSPC.from_model(model)
            np.testing.assert_array_equal(compiled.transform_rows(rows,h),model.predict(rows,h))

    def test_direct_queries_preserve_strict_threshold_ties(self):
        for weight in [0.,.6,1.]:
            model,rows,h=fixture(weight);compiled=CompiledFSPC.from_model(model)
            q=model.predict(rows,h);thresholds=q[:,[0,49,99]]
            expected=(q[:,:,None]>thresholds[:,None,:]).mean(1)
            np.testing.assert_array_equal(compiled.probabilities_rows(rows,h,thresholds),expected)
            mean,_,scale=summaries(rows)
            np.testing.assert_array_equal(compiled.probabilities_summaries(mean,scale,h,thresholds),expected)

    def test_reloaded_state_keeps_full_mean_parameters_and_outputs(self):
        model,rows,h=fixture();compiled=CompiledFSPC.from_model(model)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fspc.npz';compiled.save(path);loaded=CompiledFSPC.load(path)
            np.testing.assert_array_equal(loaded.transform_rows(rows,h),model.predict(rows,h))
            self.assertEqual(loaded.weight,model.weight)
            self.assertEqual(loaded.strength,model.strength)

    def test_saved_probe_preserves_fortran_reduction_order(self):
        model,rows,h=fixture()
        rows=np.asfortranarray(np.tile(rows,(5,1)));h=np.tile(h,5)
        reference=model.predict(rows,h)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'probe.npz'
            save_parity_probe(path,rows,h,reference[:,[0,49,99]],reference)
            with np.load(path) as d:
                self.assertTrue(d['rows'].flags.f_contiguous)
                np.testing.assert_array_equal(model.predict(d['rows'],d['horizon']),d['reference'])


if __name__=='__main__':unittest.main()
