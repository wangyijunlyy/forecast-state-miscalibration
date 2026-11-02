import unittest
import numpy as np


class NativeDistributionTests(unittest.TestCase):
    def test_native_nll_matches_author_pytorch_default_including_variance_floor(self):
        import torch
        from deepstuq_math import native_gaussian_nll
        mean = np.array([0., 2., -1.]); target = np.array([1., 0., -1.001])
        variance = np.array([2., 4., 1e-10])
        reference = torch.nn.GaussianNLLLoss(reduction='none')(
            torch.tensor(mean), torch.tensor(target), torch.tensor(variance)).numpy()
        np.testing.assert_allclose(native_gaussian_nll(mean, target, variance), reference, rtol=1e-13)

    def test_temperature_changes_aleatoric_but_not_epistemic_variance(self):
        from deepstuq_math import marginal_parameters
        means = np.array([0., 2.]).reshape(2, 1, 1, 1)
        log_variances = np.full_like(means, np.log(4.))
        mu, variance = marginal_parameters(means, log_variances, np.log(2.))
        np.testing.assert_allclose(mu, 1.)
        np.testing.assert_allclose(variance, 4.)
        _, raw = marginal_parameters(means, log_variances, 0.)
        np.testing.assert_allclose(raw, 6.)


if __name__ == '__main__':
    unittest.main()
