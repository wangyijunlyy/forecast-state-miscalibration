"""Critical mathematical checks for the isolated mean-anchor ablations."""
import unittest
import numpy as np
from portable_dc import CompiledCalibrator
try:
    from mean_anchor import centered_forecasts, fit_bias, select_strength
except ImportError:
    centered_forecasts = fit_bias = select_strength = None

class MeanAnchorTests(unittest.TestCase):
    def calibrator(self):
        return CompiledCalibrator(np.array([-np.inf,np.inf]),np.array([-np.inf,np.inf]),
            (np.linspace(-1,2,100)**3+3)[None,:],np.array([0]),192,1)

    def test_centering_preserves_actual_output_mean_mse_and_spacings(self):
        self.assertIsNotNone(centered_forecasts,'Mean-anchor reconstruction is not implemented')
        cal=self.calibrator();rows=np.arange(500,dtype=float).reshape(5,100)/100-2
        h=np.arange(5);target=np.array([-1.,2.,-3.,4.,5.])
        original=cal.transform_rows(rows,h);out=centered_forecasts(cal,rows,h)
        np.testing.assert_allclose(out.mean(1),rows.mean(1),atol=2e-14,rtol=0)
        np.testing.assert_allclose(np.diff(out,axis=1),np.diff(original,axis=1),atol=2e-14,rtol=0)
        self.assertAlmostEqual(np.mean((out.mean(1)-target)**2),np.mean((rows.mean(1)-target)**2),places=12)
        self.assertGreater(abs(original.mean()-rows.mean()),.1)

    def test_bias_fit_uses_raw_errors_and_shrinks_sparse_states_to_zero(self):
        self.assertIsNotNone(fit_bias,'Train-only state bias fitting is not implemented')
        cal=self.calibrator();rows=np.tile(np.linspace(-1,1,100),(4,1));h=np.zeros(4,dtype=int)
        b,n=fit_bias(cal,rows,np.full(4,2.),h,pseudo_count=4,min_count=4)
        np.testing.assert_array_equal(n,[4]);np.testing.assert_allclose(b,[1.])
        b,n=fit_bias(cal,rows,np.full(4,2.),h,pseudo_count=4,min_count=5)
        np.testing.assert_array_equal(b,[0.])

    def test_independent_validation_selects_strength_with_native_fallback(self):
        self.assertIsNotNone(select_strength,'Validation-only strength selection is not implemented')
        cal=self.calibrator();rows=np.tile(np.linspace(-1,1,100),(4,1));h=np.zeros(4,dtype=int)
        alpha,meta=select_strength(cal,rows,np.full(4,.25),h,np.array([1.]))
        self.assertAlmostEqual(alpha,.25);self.assertLess(meta['selected_mse'],1e-25)
        self.assertLessEqual(meta['selected_mse'],meta['native_mse'])
        alpha,_=select_strength(cal,rows,np.full(4,-1.),h,np.array([1.]));self.assertEqual(alpha,0.)
        alpha,_=select_strength(cal,rows,np.full(4,2.),h,np.array([1.]));self.assertEqual(alpha,1.)
        out=centered_forecasts(cal,rows,h,np.array([1.]),strength=.25)
        np.testing.assert_allclose(out.mean(1),.25,atol=1e-14)

if __name__=='__main__':unittest.main()
