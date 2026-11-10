"""Efficient marginal and joint-path metrics for state-aware VolDy-VAE."""

import math

import torch

from probts.utils.evaluator import Evaluator


def joint_path_metrics(targets, forecasts, patch_len):
    """Trajectory scores on dataset-standardized values; lower is better.

    PathES averages channel-wise temporal energy scores. The first expectation
    uses every forecast draw; the second uses disjoint draw pairs, avoiding an
    O(S^2) distance matrix. PathVS is a p=0.5 temporal variogram score over
    lags 1, P, 2P, and 4P. All reductions average over cases and channels.
    """
    if targets.ndim != 3 or forecasts.ndim != 4:
        raise ValueError("Expected targets [B,H,C] and forecasts [B,S,H,C]")
    batch, samples, horizon, channels = forecasts.shape
    if targets.shape != (batch, horizon, channels) or samples < 2:
        raise ValueError("Incompatible target/forecast shapes or too few samples")

    chunk = 10
    first_sum = targets.new_zeros((batch, channels))
    for start in range(0, samples, chunk):
        draw = forecasts[:, start:start + chunk]
        first_sum += torch.linalg.vector_norm(draw - targets[:, None], dim=2).sum(dim=1)
    first = first_sum / (samples * math.sqrt(horizon))

    pairs = samples // 2
    pair_sum = targets.new_zeros((batch, channels))
    for start in range(0, pairs, chunk):
        left = forecasts[:, start:start + chunk]
        right = forecasts[:, pairs + start:pairs + start + chunk]
        pair_sum += torch.linalg.vector_norm(left - right, dim=2).sum(dim=1)
    second = pair_sum / (pairs * math.sqrt(horizon))
    path_es = (first - 0.5 * second).mean()

    scores = []
    for lag in (1, patch_len, 2 * patch_len, 4 * patch_len):
        if lag >= horizon:
            continue
        observed = (targets[:, lag:] - targets[:, :-lag]).abs().sqrt()
        expected = torch.zeros_like(observed)
        for start in range(0, samples, chunk):
            draw = forecasts[:, start:start + chunk]
            expected += (draw[:, :, lag:] - draw[:, :, :-lag]).abs().sqrt().sum(dim=1)
        expected /= samples
        scores.append((observed - expected).square().mean())
    path_vs = torch.stack(scores).mean()
    return {"PathES": float(path_es.detach().cpu()),
            "PathVS": float(path_vs.detach().cpu())}


class StateEvaluator(Evaluator):
    """Compute all marginal quantiles from one sort per evaluator call."""

    def _torch_sequence_metrics(self, targets, forecasts, seasonal_error=None,
                                samples_dim=1, loss_weights=None):
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
        if forecasts.dim() == 3:
            forecasts = forecasts.unsqueeze(-1)
        if samples_dim != 1:
            raise ValueError("StateEvaluator expects sample dimension 1")

        reduce_dims = tuple(range(1, targets.dim()))
        mean_forecasts = forecasts.mean(dim=1)
        # torch.quantile with individual q values sorts this very large tensor
        # repeatedly. Sorting once yields the same linear interpolation rule.
        sorted_forecasts = forecasts.sort(dim=1).values
        levels = torch.as_tensor(
            [0.5, *map(float, self.quantiles)],
            dtype=forecasts.dtype, device=forecasts.device,
        )
        positions = levels * (forecasts.shape[1] - 1)
        low_idx = positions.floor().long()
        high_idx = positions.ceil().long()
        fraction = (positions - low_idx).view(1, -1, 1, 1)
        quantiles = (sorted_forecasts.index_select(1, low_idx)
                     + fraction * (sorted_forecasts.index_select(1, high_idx)
                                   - sorted_forecasts.index_select(1, low_idx)))
        median_forecasts = quantiles[:, 0]

        mse_values = torch.mean(torch.square(targets - mean_forecasts), dim=reduce_dims)
        abs_error_values = torch.sum(torch.abs(targets - median_forecasts), dim=reduce_dims)
        abs_target_sum_values = torch.sum(torch.abs(targets), dim=reduce_dims).clamp_min(1e-8)
        abs_target_mean_values = torch.mean(torch.abs(targets), dim=reduce_dims).clamp_min(1e-8)
        metrics = {
            "MSE": mse_values.mean(),
            "abs_error": abs_error_values.mean(),
            "abs_target_sum": abs_target_sum_values.mean(),
            "abs_target_mean": abs_target_mean_values.mean(),
            "RMSE": torch.sqrt(mse_values).mean(),
            "NRMSE": (torch.sqrt(mse_values) / abs_target_mean_values).mean(),
            "ND": (abs_error_values / abs_target_sum_values).mean(),
        }
        if seasonal_error is not None:
            if seasonal_error.dim() == 2:
                seasonal_error = seasonal_error.unsqueeze(-1)
            diff = torch.mean(torch.abs(targets - median_forecasts), dim=1)
            mase_values = diff / seasonal_error.squeeze(1).clamp_min(1e-8)
            mase_values = torch.nan_to_num(mase_values, nan=0.0, posinf=0.0, neginf=0.0)
            metrics["MASE"] = mase_values.mean()
        if loss_weights is not None:
            weights = loss_weights.detach().to(targets.device).view(1, -1, 1)
            nd = torch.abs(targets - mean_forecasts) / abs_target_sum_values.view(-1, 1, 1)
            metrics["weighted_ND"] = torch.sum(weights * nd)
        else:
            metrics["weighted_ND"] = metrics["ND"]

        quantile_losses = []
        weighted_quantile_losses = []
        coverages = []
        for i, q in enumerate(self.quantiles, start=1):
            q_value = float(q)
            q_forecasts = quantiles[:, i]
            q_loss_values = 2 * torch.sum(
                torch.abs((q_forecasts - targets)
                          * ((targets <= q_forecasts).to(targets.dtype) - q_value)),
                dim=reduce_dims,
            )
            weighted_q_loss_values = q_loss_values / abs_target_sum_values
            coverage_values = torch.mean((targets < q_forecasts).to(targets.dtype), dim=reduce_dims)
            metrics[self.loss_name(q)] = q_loss_values.mean()
            metrics[self.weighted_loss_name(q)] = weighted_q_loss_values.mean()
            metrics[self.coverage_name(q)] = coverage_values.mean()
            quantile_losses.append(metrics[self.loss_name(q)])
            weighted_quantile_losses.append(metrics[self.weighted_loss_name(q)])
            coverages.append(coverage_values)

        metrics["mean_absolute_QuantileLoss"] = torch.stack(quantile_losses).mean()
        metrics["CRPS"] = torch.stack(weighted_quantile_losses).mean()
        metrics["MAE_Coverage"] = torch.stack([
            torch.abs(metrics[self.coverage_name(q)] - float(q)) for q in self.quantiles
        ]).mean()
        coverages_tensor = torch.stack(coverages, dim=1)
        zeros = torch.zeros((targets.shape[0], 1), device=targets.device, dtype=targets.dtype)
        ones = torch.ones((targets.shape[0], 1), device=targets.device, dtype=targets.dtype)
        observed = torch.diff(torch.cat([zeros, coverages_tensor, ones], dim=1), dim=1)
        metrics["QICE"] = (observed - 1.0 / (len(self.quantiles) + 1)).abs().mean()
        return metrics
