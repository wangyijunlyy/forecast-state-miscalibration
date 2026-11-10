"""Official PPM-style CRPS/QICE evaluation for sample forecasts."""

from __future__ import annotations

import numpy as np

from utils.metrics import metric


def _crps_vector(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Compute equally weighted sample CRPS for rows of samples.

    This is algebraically equivalent to CRPS.CRPS(...).compute()[0] used by
    the official PPM evaluator, but avoids millions of Python-level calls.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    sample_count = pred.shape[1]
    sorted_pred = np.sort(pred, axis=1)
    term_observation = np.mean(np.abs(pred - true[:, None]), axis=1)
    weights = (2 * np.arange(1, sample_count + 1, dtype=np.float64) - sample_count - 1) / (sample_count**2)
    term_pairwise = sorted_pred @ weights
    return term_observation - term_pairwise


def calculate_batch_crps(pred: np.ndarray, true: np.ndarray) -> float:
    batch_size, sample_count, _horizon, _target_dim = pred.shape
    all_res = []
    for i in range(batch_size):
        p_in = pred[i].transpose(1, 2, 0).reshape(-1, sample_count)
        t_in = true[i].reshape(-1)
        all_res.append(_crps_vector(p_in, t_in))
    all_res = np.array(all_res)
    return np.mean(all_res, axis=1).sum()


def compute_qice_bin_count(n_bin: int, all_true_y: np.ndarray, all_generated_y: np.ndarray) -> np.ndarray:
    quantile_list = np.arange(n_bin + 1) * (100 / n_bin)
    y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
    y_true = all_true_y.T
    quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
    y_true_quantile_membership = quantile_membership_array.sum(axis=0)
    y_true_quantile_bin_count = np.array([(y_true_quantile_membership == v).sum() for v in np.arange(n_bin + 2)])
    return y_true_quantile_bin_count[1:-1]


def evaluate_ppm_samples(preds: list[np.ndarray], trues: list[np.ndarray]) -> dict[str, float | np.ndarray]:
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0.0
    sum_crps = 0.0
    bin_count = np.zeros(10)
    qice_samples = 0

    for pred, true in zip(preds, trues):
        batch_size, sample_count, _horizon, _target_dim = pred.shape
        q_pred = pred.transpose(0, 2, 3, 1).reshape(-1, sample_count)
        q_true = true.reshape(-1, 1)
        bin_count += compute_qice_bin_count(10, q_true, q_pred)
        qice_samples += q_pred.shape[0]
        sum_crps += calculate_batch_crps(pred, true)

        pred_mean = np.mean(pred, axis=1)
        mae, mse, _rmse, _mape, _mspe = metric(pred_mean, true)
        total_mse += mse * pred_mean.shape[0]
        total_mae += mae * pred_mean.shape[0]
        total_samples += pred_mean.shape[0]

    qice_coverage_ratio = bin_count / qice_samples
    qice = np.absolute(np.ones(10) / 10 - qice_coverage_ratio).mean()
    return {
        "CRPS": sum_crps / total_samples,
        "MSE": total_mse / total_samples,
        "MAE": total_mae / total_samples,
        "QICE": qice * 100,
        "QICE_bins": qice_coverage_ratio,
        "total_samples": total_samples,
    }
