import numpy as np
import torch
from gluonts.time_feature import get_seasonality
from .metrics import *


class Evaluator:
    
    def __init__(self, quantiles_num=10, smooth=False):
        self.quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
        self.ignore_invalid_values = True
        self.smooth = smooth

    def loss_name(self, q):
        return f"QuantileLoss[{q}]"

    def weighted_loss_name(self, q):
        return f"wQuantileLoss[{q}]"

    def coverage_name(self, q):
        return f"Coverage[{q}]"

    def get_sequence_metrics(self, targets, forecasts, seasonal_error=None, samples_dim=1,loss_weights=None):
        mean_forecasts = forecasts.mean(axis=samples_dim)
        median_forecasts = np.quantile(forecasts, 0.5, axis=samples_dim)
        metrics = {
            "MSE": mse(targets, mean_forecasts),
            "abs_error": abs_error(targets, median_forecasts),
            "abs_target_sum": abs_target_sum(targets),
            "abs_target_mean": abs_target_mean(targets),
            "MAPE": mape(targets, median_forecasts),
            "sMAPE": smape(targets, median_forecasts),
        }
        
        if seasonal_error is not None:
            metrics["MASE"] = mase(targets, median_forecasts, seasonal_error)
        
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]
        
        # calculate weighted loss
        if loss_weights is not None:
            nd = np.abs(targets - mean_forecasts) / np.sum(np.abs(targets), axis=(1, 2))
            loss_weights = loss_weights.detach().unsqueeze(0).unsqueeze(-1).numpy()
            weighted_ND = loss_weights * nd
            metrics['weighted_ND'] = np.sum(weighted_ND)
        else:
            metrics['weighted_ND'] = metrics["ND"]

        for q in self.quantiles:
            q_forecasts = np.quantile(forecasts, q, axis=samples_dim)
            metrics[self.loss_name(q)] = quantile_loss(targets, q_forecasts, q)
            metrics[self.weighted_loss_name(q)] = \
                metrics[self.loss_name(q)] / metrics["abs_target_sum"]
            metrics[self.coverage_name(q)] = coverage(targets, q_forecasts)
        
        metrics["mean_absolute_QuantileLoss"] = np.mean(
            [metrics[self.loss_name(q)] for q in self.quantiles]
        )
        metrics["CRPS"] = np.mean(
            [metrics[self.weighted_loss_name(q)] for q in self.quantiles]
        )
        metrics["MAE_Coverage"] = np.mean(
            [
                np.abs(metrics[self.coverage_name(q)] - np.array([q]))
                for q in self.quantiles
            ]
        )

        # Calculate QICE
        # We assume self.quantiles are sorted and correspond to boundaries of intervals
        # For M equal-sized intervals, we expect M-1 quantiles dividing the space.
        # But here self.quantiles has `quantiles_num` entries (e.g. 10).
        # Typically quantiles are 0.1, 0.2, ... 0.9.
        # This creates 10 intervals: [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0].
        # The coverage at q is P(y < y_q).
        # We need the proportion of data in each interval.
        # Interval m (m=1..M): (q_{m-1}, q_m].
        # Proportion r_m = coverage(q_m) - coverage(q_{m-1}).
        # We pad coverage with 0 at the start and 1 at the end?
        # self.quantiles usually doesn't include 0 or 1.
        # If quantiles are [0.1, ..., 0.9], we have 9 cut points, defining 10 intervals.
        
        # Let's get coverages for all quantiles
        coverages = [metrics[self.coverage_name(q)] for q in self.quantiles]
        # Prepend 0 and append 1 to coverages to represent the full probability range
        # Note: coverages are empirical, so they should be monotonic but not strictly.
        # However, we are partitioning the *predicted* distribution's quantiles.
        # The true data falls into these bins. 
        # r_m is the fraction of true data in the m-th bin.
        # Bin m corresponds to quantile range [ (m-1)/M, m/M ].
        # So r_m = coverage(q=m/M) - coverage(q=(m-1)/M).
        
        # We need to ensure we have the right quantiles for this. The constructor sets:
        # self.quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
        # If quantiles_num=10, quantiles=[0.1, 0.2, ... 0.9].
        # So we have coverages at 0.1, ..., 0.9.
        # We implicitly have coverage at 0.0 is 0 and coverage at 1.0 is 1.
        
        all_coverages = [0.0] + coverages + [1.0]
        observed_proportions = np.diff(all_coverages)
        
        # Ideal proportion is 1/M. M = quantiles_num.
        # In current code, len(self.quantiles) = quantiles_num - 1?
        # self.quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
        # len is 10-1 = 9. So M=10.
        
        # M = len(self.quantiles) + 1
        M = len(self.quantiles) + 1
        optimal_proportion = 1.0 / M
        
        metrics["QICE"] = np.mean(np.abs(observed_proportions - optimal_proportion))

        return metrics

    def get_metrics(self, targets, forecasts, seasonal_error=None, samples_dim=1, loss_weights=None):
        metrics = {}
        seq_metrics = {}
        
        # Calculate metrics for each sequence
        for i in range(targets.shape[0]):
            single_seq_metrics = self.get_sequence_metrics(
                np.expand_dims(targets[i], axis=0),
                np.expand_dims(forecasts[i], axis=0),
                np.expand_dims(seasonal_error[i], axis=0) if seasonal_error is not None else None,
                samples_dim,
                loss_weights
            )
            for metric_name, metric_value in single_seq_metrics.items():
                if metric_name not in seq_metrics:
                    seq_metrics[metric_name] = []
                seq_metrics[metric_name].append(metric_value)
        
        for metric_name, metric_values in seq_metrics.items():
            metrics[metric_name] = np.mean(metric_values)
        return metrics

    @property
    def selected_metrics(self):
        return [ "ND",'weighted_ND', 'CRPS', "NRMSE", "MSE", "MASE", "QICE"]

    def _torch_seasonal_error(self, past_data, freq):
        seasonality = get_seasonality(freq)
        forecast_freq = seasonality if seasonality < past_data.shape[1] else 1
        y_t = past_data[:, :-forecast_freq]
        y_tm = past_data[:, forecast_freq:]
        return torch.mean(torch.abs(y_t - y_tm), dim=1, keepdim=True)

    def _torch_sequence_metrics(self, targets, forecasts, seasonal_error=None, samples_dim=1, loss_weights=None):
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
        if forecasts.dim() == 3:
            forecasts = forecasts.unsqueeze(-1)

        reduce_dims = tuple(range(1, targets.dim()))
        mean_forecasts = forecasts.mean(dim=samples_dim)
        median_forecasts = torch.quantile(forecasts, 0.5, dim=samples_dim)

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
        for q in self.quantiles:
            q_value = float(q)
            q_forecasts = torch.quantile(forecasts, q_value, dim=samples_dim)
            q_loss_values = 2 * torch.sum(
                torch.abs((q_forecasts - targets) * ((targets <= q_forecasts).to(targets.dtype) - q_value)),
                dim=reduce_dims,
            )
            weighted_q_loss_values = q_loss_values / abs_target_sum_values
            coverage_values = torch.mean((targets < q_forecasts).to(targets.dtype), dim=reduce_dims)

            q_loss = q_loss_values.mean()
            weighted_q_loss = weighted_q_loss_values.mean()
            coverage_value = coverage_values.mean()

            metrics[self.loss_name(q)] = q_loss
            metrics[self.weighted_loss_name(q)] = weighted_q_loss
            metrics[self.coverage_name(q)] = coverage_value
            quantile_losses.append(q_loss)
            weighted_quantile_losses.append(weighted_q_loss)
            coverages.append(coverage_values)

        metrics["mean_absolute_QuantileLoss"] = torch.stack(quantile_losses).mean()
        metrics["CRPS"] = torch.stack(weighted_quantile_losses).mean()
        metrics["MAE_Coverage"] = torch.mean(
            torch.stack(
                [
                    torch.abs(metrics[self.coverage_name(q)] - float(q))
                    for q in self.quantiles
                ]
            )
        )

        coverages_tensor = torch.stack(coverages, dim=1)
        zeros = torch.zeros((targets.shape[0], 1), device=targets.device, dtype=targets.dtype)
        ones = torch.ones((targets.shape[0], 1), device=targets.device, dtype=targets.dtype)
        observed_proportions = torch.diff(torch.cat([zeros, coverages_tensor, ones], dim=1), dim=1)
        optimal_proportion = 1.0 / (len(self.quantiles) + 1)
        metrics["QICE"] = torch.mean(torch.abs(observed_proportions - optimal_proportion))

        return metrics

    def _torch_metrics(self, targets, forecasts, past_data, freq, loss_weights=None):
        if self.ignore_invalid_values:
            targets = torch.nan_to_num(targets)
            forecasts = torch.nan_to_num(forecasts)
            past_data = torch.nan_to_num(past_data)

        seasonal_error = self._torch_seasonal_error(past_data, freq)
        metrics = self._torch_sequence_metrics(
            targets,
            forecasts,
            seasonal_error=seasonal_error,
            samples_dim=1,
            loss_weights=loss_weights,
        )
        metrics_sum = self._torch_sequence_metrics(
            targets.sum(dim=-1),
            forecasts.sum(dim=-1),
            samples_dim=1,
        )

        output_metrics = {}
        for k in self.selected_metrics:
            output_metrics[k] = float(metrics[k].detach().cpu())
            if k in metrics_sum:
                output_metrics[f"{k}-Sum"] = float(metrics_sum[k].detach().cpu())
        return output_metrics

    def __call__(self, targets, forecasts, past_data, freq, loss_weights=None):
        """

        Parameters
        ----------
        targets
            groundtruth in (batch_size, prediction_length, target_dim)
        forecasts
            forecasts in (batch_size, num_samples, prediction_length, target_dim)
        Returns
        -------
        Dict[String, float]
            metrics
        """
        if isinstance(targets, torch.Tensor) and isinstance(forecasts, torch.Tensor):
            return self._torch_metrics(targets, forecasts, past_data, freq, loss_weights=loss_weights)

        targets = targets.cpu().detach().numpy()
        forecasts = forecasts.cpu().detach().numpy()
        past_data = past_data.cpu().detach().numpy()
        if self.ignore_invalid_values:
            targets = np.ma.masked_invalid(targets)
            forecasts = np.ma.masked_invalid(forecasts)
        
        seasonal_error = calculate_seasonal_error(past_data, freq)

        metrics = self.get_metrics(targets, forecasts, seasonal_error=seasonal_error, samples_dim=1, loss_weights=loss_weights)
        metrics_sum = self.get_metrics(targets.sum(axis=-1), forecasts.sum(axis=-1), samples_dim=1)
        
        # select output metrics
        output_metrics = dict()
        for k in self.selected_metrics:
            output_metrics[k] = metrics[k]
            if k in metrics_sum:
                output_metrics[f"{k}-Sum"] = metrics_sum[k]
        return output_metrics
