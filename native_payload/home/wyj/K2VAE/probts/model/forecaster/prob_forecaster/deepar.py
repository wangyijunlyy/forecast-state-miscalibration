import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.data import ProbTSBatchData
from probts.model.forecaster import Forecaster
from probts.utils import repeat


class DeepARModel(Forecaster):
    """Gaussian DeepAR baseline for continuous long-horizon forecasting.

    The implementation follows the DeepAR teacher-forcing / autoregressive
    decoding idea while using a Gaussian likelihood so that it matches the
    ProbTS LTSF evaluation protocol.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        min_scale: float = 1e-4,
        **kwargs,
    ):
        super().__init__(autoregressive=True, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.min_scale = min_scale

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mu_proj = nn.Linear(hidden_size, self.target_dim)
        self.scale_proj = nn.Linear(hidden_size, self.target_dim)

    def _distribution_args(self, hidden):
        mu = self.mu_proj(hidden)
        scale = F.softplus(self.scale_proj(hidden)) + self.min_scale
        return mu, scale

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, mode="all")
        outputs, _ = self.rnn(inputs)

        # Teacher forcing: output after the last observed value predicts the
        # first future value; output after future t predicts future t+1.
        start = self.max_context_length - 1
        end = start + self.max_prediction_length
        pred_hidden = outputs[:, start:end, :]
        mu, scale = self._distribution_args(pred_hidden)

        target = batch_data.future_target_cdf
        nll = torch.log(scale) + 0.5 * ((target - mu) / scale).pow(2)
        nll = nll + 0.5 * math.log(2.0 * math.pi)

        if batch_data.future_observed_values is not None:
            weights = batch_data.future_observed_values
            nll = torch.where(weights > 0, nll * weights, torch.zeros_like(nll))
            denom = torch.clamp(weights.sum(), min=1.0)
            return nll.sum() / denom
        return nll.mean()

    def encode(self, batch_data):
        inputs = self.get_inputs(batch_data, mode="encode")
        _, states = self.rnn(inputs)
        return states

    def decode(self, batch_data, states):
        inputs = self.get_inputs(batch_data, mode="decode")
        outputs, states = self.rnn(inputs, states)
        return outputs, states

    def forecast(self, batch_data, num_samples=None):
        if num_samples is None:
            num_samples = 1

        states = self.encode(batch_data)

        repeated_target_dimension_indicator = repeat(
            batch_data.target_dimension_indicator, num_samples
        )
        repeated_past_target_cdf = repeat(batch_data.past_target_cdf, num_samples)
        repeated_future_time_feat = repeat(batch_data.future_time_feat, num_samples)
        repeated_states = repeat(states, num_samples, dim=1)

        future_samples = []
        for k in range(self.max_prediction_length):
            repeated_batch_data = ProbTSBatchData(
                {
                    "target_dimension_indicator": repeated_target_dimension_indicator,
                    "past_target_cdf": repeated_past_target_cdf,
                    "future_time_feat": repeated_future_time_feat[:, k : k + 1, :],
                },
                device=batch_data.device,
            )

            outputs, repeated_states = self.decode(
                repeated_batch_data, repeated_states
            )
            mu, scale = self._distribution_args(outputs)
            new_samples = mu + torch.randn_like(mu) * scale
            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, new_samples), dim=1
            )

        forecasts = torch.cat(future_samples, dim=1)
        forecasts = forecasts.reshape(
            -1, num_samples, self.max_prediction_length, self.target_dim
        )
        return forecasts
