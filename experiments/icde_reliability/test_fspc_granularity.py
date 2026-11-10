"""Budget isolation and horizon-sensitive mean control for the FSPC sweep."""
import unittest
import numpy as np
from mean_anchor import centered_forecasts
try:
    from fspc_granularity import fit_variant
except ImportError:
    fit_variant = None


class FSPCGranularityTests(unittest.TestCase):
    def arrays(self):
        rows=np.tile(np.linspace(-1,1,100),(1600,1))
        horizon=np.repeat([0,3],800)
        train_target=np.repeat([2.,-2.],800)
        validation_target=np.repeat([.25,-.5],800)
        return rows,horizon,train_target,validation_target

    def test_horizon_states_use_train_bias_and_one_validation_coefficient(self):
        self.assertIsNotNone(fit_variant,'FSPC partition fitting is not implemented')
        rows,h,train_y,val_y=self.arrays()
        cal,bias,strength,meta=fit_variant(rows,train_y,h,rows,val_y,h,4,1,2,256,19)
        expected_bias=1600./928.
        np.testing.assert_allclose(bias,[expected_bias,-expected_bias],atol=1e-12)
        take=np.random.default_rng(19).permutation(len(val_y))[:256]
        shifts=np.where(h[take]==0,expected_bias,-expected_bias)
        expected_strength=np.dot(val_y[take],shifts)/np.dot(shifts,shifts)
        self.assertAlmostEqual(strength,expected_strength,places=12)
        out=centered_forecasts(cal,rows,h,bias,strength)
        np.testing.assert_allclose(out.mean(1),np.where(h==0,1,-1)*expected_strength*expected_bias,atol=1e-12)
        np.testing.assert_allclose(np.diff(out,axis=1),np.diff(cal.transform_rows(rows,h),axis=1),atol=1e-12)
        self.assertEqual(meta['training_coordinates'],1600)
        self.assertEqual(meta['calibration_coordinates'],256)

    def test_unused_validation_targets_do_not_affect_any_fitted_parameter(self):
        self.assertIsNotNone(fit_variant,'FSPC partition fitting is not implemented')
        rows,h,train_y,val_y=self.arrays()
        first=fit_variant(rows,train_y,h,rows,val_y,h,4,2,1,256,19)
        take=np.random.default_rng(19).permutation(len(val_y))[:256]
        altered=val_y.copy();outside=np.ones(len(val_y),bool);outside[take]=False
        altered[outside]+=10000
        second=fit_variant(rows,train_y,h,rows,altered,h,4,2,1,256,19)
        np.testing.assert_array_equal(first[0].quantiles,second[0].quantiles)
        np.testing.assert_array_equal(first[1],second[1])
        self.assertEqual(first[2],second[2])


if __name__=='__main__':unittest.main()
