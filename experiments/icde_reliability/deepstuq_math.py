"""The Gaussian marginal approximation used in the author combined_test routine."""
import numpy as np


def native_gaussian_nll(mean, target, variance):
    """Author GaussianNLLLoss(full=False, eps=1e-6), per coordinate."""
    v = np.maximum(np.asarray(variance, dtype=np.float64), 1e-6)
    return .5*(np.log(v)+(np.asarray(target)-np.asarray(mean))**2/v)


def marginal_parameters(mc_means, mc_log_variances, log_temperature=0.):
    means = np.asarray(mc_means, dtype=np.float64)
    log_variances = np.asarray(mc_log_variances, dtype=np.float64)
    if means.shape != log_variances.shape or means.shape[0] < 2:
        raise ValueError('Expected matching multiple-dropout estimates')
    mu = means.mean(0)
    variance = means.var(0, ddof=1)+np.exp(log_variances.mean(0)-float(log_temperature))
    if not np.isfinite(mu).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('Invalid native DeepSTUQ predictive marginal')
    return mu, variance
