# ---------------------------------------------------------------------------------
# Portions of this file are derived from gluonts
# - Source: https://github.com/awslabs/gluonts
# - Paper: GluonTS: Probabilistic and Neural Time Series Modeling in Python
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from typing import Optional
import numpy as np
from gluonts.time_feature import get_seasonality


def mse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)
    """
    return np.mean(np.square(target - forecast))


def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        abs\_error = sum(|Y - \hat{Y}|)
    """
    return np.sum(np.abs(target - forecast))


def abs_target_sum(target) -> float:
    r"""
    .. math::

        abs\_target\_sum = sum(|Y|)
    """
    return np.sum(np.abs(target))


def abs_target_mean(target) -> float:
    r"""
    .. math::

        abs\_target\_mean = mean(|Y|)
    """
    return np.mean(np.abs(target))


def mase(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_error: np.ndarray,
) -> float:
    r"""
    .. math::

        mase = mean(|Y - \hat{Y}|) / seasonal\_error

    See [HA21]_ for more details.
    """
    diff = np.mean(np.abs(target - forecast), axis=1)
    mase = diff / seasonal_error
    # if seasonal_error is 0, set mase to 0
    mase = mase.filled(0)  
    return np.mean(mase)

def calculate_seasonal_error(
    past_data: np.ndarray,
    freq: Optional[str] = None,
):
    r"""
    .. math::

        seasonal\_error = mean(|Y[t] - Y[t-m]|)

    where m is the seasonal frequency. See [HA21]_ for more details.
    """
    seasonality = get_seasonality(freq)

    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        # revert to freq=1

        # logging.info('The seasonal frequency is larger than the length of the
        # time series. Reverting to freq=1.')
        forecast_freq = 1
        
    y_t = past_data[:, :-forecast_freq]
    y_tm = past_data[:, forecast_freq:]

    mean_diff = np.mean(np.abs(y_t - y_tm), axis=1)
    mean_diff = np.expand_dims(mean_diff, axis=1)

    return mean_diff



def mape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))

    See [HA21]_ for more details.
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    See [HA21]_ for more details.
    """
    return 2 * np.mean(
        np.abs(target - forecast) / (np.abs(target) + np.abs(forecast))
    )


def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    r"""
    .. math::

        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * ((Y <= \hat{Y}) - q)|)
    """
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)))


def coverage(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        coverage = mean(Y < \hat{Y})
    """
    return np.mean(target < forecast)


def _rankdata(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(flat), dtype=float)
    return ranks.reshape(values.shape)


def spearman_corr(error: np.ndarray, uncertainty: np.ndarray) -> float:
    error_rank = _rankdata(np.asarray(error))
    uncertainty_rank = _rankdata(np.asarray(uncertainty))
    error_centered = error_rank - error_rank.mean()
    uncertainty_centered = uncertainty_rank - uncertainty_rank.mean()
    denom = np.sqrt(np.sum(error_centered ** 2) * np.sum(uncertainty_centered ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(error_centered * uncertainty_centered) / denom)


def ause(error: np.ndarray, uncertainty: np.ndarray, steps: int = 20) -> float:
    error_flat = np.asarray(error).reshape(-1)
    uncertainty_flat = np.asarray(uncertainty).reshape(-1)
    if error_flat.size == 0:
        return 0.0

    uncertainty_order = np.argsort(uncertainty_flat)
    oracle_order = np.argsort(error_flat)
    fractions = np.linspace(0.0, 1.0, steps + 1)
    sparse_curve = []
    oracle_curve = []
    for frac in fractions:
        keep = max(1, int((1.0 - frac) * error_flat.size))
        sparse_curve.append(np.mean(error_flat[uncertainty_order[:keep]]))
        oracle_curve.append(np.mean(error_flat[oracle_order[:keep]]))
    return float(np.trapz(np.abs(np.asarray(sparse_curve) - np.asarray(oracle_curve)), fractions))
