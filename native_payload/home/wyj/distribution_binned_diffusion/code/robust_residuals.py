"""Training-only robust winsorization before binned residual-pool fitting."""
import numpy as np


def filter_residuals(residuals, multiplier=5.0):
    """Cap standardized residuals at median +/- 5 robust standard deviations.

    Bounds depend only on the calibration residual array. No test targets,
    test scores, per-dataset thresholds, or output clipping are used.
    """
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    estimator = '1.4826*MAD'
    if scale == 0:
        q25, q75 = np.quantile(values, [.25, .75])
        scale = float((q75 - q25) / 1.349)
        estimator = 'IQR/1.349 (MAD zero)'
    lower = center - multiplier * scale
    upper = center + multiplier * scale
    filtered = np.clip(values, lower, upper)
    metadata = dict(method='global calibration-residual winsorization',
        threshold_source='calibration split only', multiplier=float(multiplier),
        median=center, mad=mad, robust_scale=scale, scale_estimator=estimator,
        lower_bound=lower, upper_bound=upper, total_points=len(values),
        clipped_lower=int(np.sum(values < lower)),
        clipped_upper=int(np.sum(values > upper)))
    return filtered, metadata
