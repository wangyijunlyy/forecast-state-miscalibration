"""
Forecasting-adapted VHVM diagnostic baseline for the ProbTS interface.

The original VHVM (Yin and Barucca, 2022) targets time-varying covariance and
volatility estimation for financial returns.  This wrapper adapts the same
variational recurrent volatility idea to long-horizon probabilistic forecasting:
the look-back window is encoded by a recurrent variational state, future latent
volatility states are rolled forward recurrently, and each future step emits a
Gaussian location-scale distribution.

Use this as a diagnostic "VHVM-adapted" baseline rather than as the original
financial covariance model unchanged.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from probts.model.forecaster.forecaster import Forecaster


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies
        in_dim = 1 + 2 * num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        freqs = torch.arange(
            1,
            self.num_frequencies + 1,
            device=t.device,
            dtype=t.dtype,
        )
        angles = 2.0 * math.pi * t * freqs.view(1, 1, -1)
        feats = torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(feats)


class VHVMForecastNet(nn.Module):
    def __init__(
        self,
        target_dim,
        context_length,
        prediction_length,
        hidden_dim=128,
        latent_dim=64,
        dropout=0.1,
        min_scale=1e-4,
        logvar_min=-10.0,
        logvar_max=5.0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.min_scale = min_scale
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.time_emb = TimeEmbedding(hidden_dim)
        self.value_proj = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encoder = nn.GRU(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.posterior = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.posterior_mu = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)

        self.prior_cell = nn.GRUCell(hidden_dim + latent_dim, hidden_dim)
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, target_dim)
        self.scale_head = nn.Linear(hidden_dim, target_dim)

    def _time_grid(self, batch_size, length, start_index, total_length, device, dtype):
        steps = torch.arange(
            start_index,
            start_index + length,
            device=device,
            dtype=dtype,
        )
        denom = max(total_length - 1, 1)
        t = (steps / denom).view(1, length, 1)
        return t.expand(batch_size, -1, -1)

    def _time_embedding(self, batch_size, length, start_index, device, dtype):
        total_length = self.context_length + self.prediction_length
        t = self._time_grid(
            batch_size,
            length,
            start_index=start_index,
            total_length=total_length,
            device=device,
            dtype=dtype,
        )
        return self.time_emb(t)

    @staticmethod
    def reparameterize(mu, logvar, training=True):
        if training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    @staticmethod
    def gaussian_nll(target, mean, logvar, reduction="mean"):
        loss = 0.5 * (logvar + (target - mean).pow(2) * torch.exp(-logvar))
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def gaussian_kl(q_mu, q_logvar, p_mu, p_logvar):
        kl = 0.5 * (
            p_logvar
            - q_logvar
            + (q_logvar.exp() + (q_mu - p_mu).pow(2)) * torch.exp(-p_logvar)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def _decode_step(self, state, z, time_feat):
        h = self.decoder(torch.cat([state, z, time_feat], dim=-1))
        mean = self.mean_head(h)
        raw_scale = self.scale_head(h)
        scale = F.softplus(raw_scale) + self.min_scale
        logvar = (2.0 * torch.log(scale)).clamp(self.logvar_min, self.logvar_max)
        return mean, logvar

    def encode_context(self, x):
        batch_size, length, _ = x.shape
        time_feat = self._time_embedding(
            batch_size,
            length,
            start_index=0,
            device=x.device,
            dtype=x.dtype,
        )
        tokens = torch.cat([self.value_proj(x), time_feat], dim=-1)
        h_seq, h_last = self.encoder(tokens)
        post = self.posterior(h_seq)
        q_mu = self.posterior_mu(post)
        q_logvar = self.posterior_logvar(post).clamp(self.logvar_min, self.logvar_max)
        return h_seq, h_last[-1], q_mu, q_logvar, time_feat

    def reconstruct_context(self, h_seq, q_mu, q_logvar, time_feat):
        z = self.reparameterize(q_mu, q_logvar, self.training)
        means = []
        logvars = []
        batch_size = h_seq.shape[0]
        prior_state = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=h_seq.device,
            dtype=h_seq.dtype,
        )
        prev_z = torch.zeros(
            batch_size,
            self.latent_dim,
            device=h_seq.device,
            dtype=h_seq.dtype,
        )
        p_mus = []
        p_logvars = []
        for t in range(h_seq.shape[1]):
            prior_input = torch.cat([time_feat[:, t, :], prev_z], dim=-1)
            prior_state = self.prior_cell(prior_input, prior_state)
            p_mu = self.prior_mu(prior_state)
            p_logvar = self.prior_logvar(prior_state).clamp(
                self.logvar_min,
                self.logvar_max,
            )
            mean, logvar = self._decode_step(h_seq[:, t, :], z[:, t, :], time_feat[:, t, :])
            means.append(mean)
            logvars.append(logvar)
            p_mus.append(p_mu)
            p_logvars.append(p_logvar)
            prev_z = z[:, t, :]

        return (
            torch.stack(means, dim=1),
            torch.stack(logvars, dim=1),
            torch.stack(p_mus, dim=1),
            torch.stack(p_logvars, dim=1),
            z[:, -1, :],
        )

    def rollout_future(self, init_state, prev_z, num_samples=1):
        batch_size = init_state.shape[0]
        device = init_state.device
        dtype = init_state.dtype
        time_feat = self._time_embedding(
            batch_size,
            self.prediction_length,
            start_index=self.context_length,
            device=device,
            dtype=dtype,
        )

        all_samples = []
        for _ in range(num_samples):
            state = init_state
            z_prev = prev_z
            preds = []
            means = []
            logvars = []
            for t in range(self.prediction_length):
                prior_input = torch.cat([time_feat[:, t, :], z_prev], dim=-1)
                state = self.prior_cell(prior_input, state)
                p_mu = self.prior_mu(state)
                p_logvar = self.prior_logvar(state).clamp(
                    self.logvar_min,
                    self.logvar_max,
                )
                z_t = self.reparameterize(p_mu, p_logvar, self.training)
                mean, logvar = self._decode_step(state, z_t, time_feat[:, t, :])
                scale = torch.exp(0.5 * logvar)
                y_t = mean + torch.randn_like(scale) * scale
                preds.append(y_t)
                means.append(mean)
                logvars.append(logvar)
                z_prev = z_t

            all_samples.append(torch.stack(preds, dim=1))
            if num_samples == 1:
                return (
                    all_samples[0],
                    torch.stack(means, dim=1),
                    torch.stack(logvars, dim=1),
                )

        return torch.stack(all_samples, dim=1)

    def forward(self, x):
        h_seq, h_last, q_mu, q_logvar, time_feat = self.encode_context(x)
        x_mu, x_logvar, p_mu, p_logvar, last_z = self.reconstruct_context(
            h_seq,
            q_mu,
            q_logvar,
            time_feat,
        )
        _, y_mu, y_logvar = self.rollout_future(h_last, last_z, num_samples=1)
        return x_mu, y_mu, q_mu, q_logvar, p_mu, p_logvar, x_logvar, y_logvar

    def sample(self, x, num_samples):
        _, h_last, q_mu, q_logvar, _ = self.encode_context(x)
        batch_size = h_last.shape[0]
        last_mu = q_mu[:, -1, :]
        last_logvar = q_logvar[:, -1, :]
        last_std = torch.exp(0.5 * last_logvar)

        expanded_state = (
            h_last.unsqueeze(1)
            .expand(batch_size, num_samples, self.hidden_dim)
            .reshape(batch_size * num_samples, self.hidden_dim)
        )
        expanded_mu = (
            last_mu.unsqueeze(1)
            .expand(batch_size, num_samples, self.latent_dim)
            .reshape(batch_size * num_samples, self.latent_dim)
        )
        expanded_std = (
            last_std.unsqueeze(1)
            .expand(batch_size, num_samples, self.latent_dim)
            .reshape(batch_size * num_samples, self.latent_dim)
        )
        expanded_z = expanded_mu + torch.randn_like(expanded_std) * expanded_std

        y, _, _ = self.rollout_future(expanded_state, expanded_z, num_samples=1)
        return y.reshape(
            batch_size,
            num_samples,
            self.prediction_length,
            self.target_dim,
        )


class VHVMModel(Forecaster):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=64,
        dropout=0.1,
        weight_beta=0.01,
        min_scale=1e-4,
        logvar_min=-10.0,
        logvar_max=5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weight_beta = weight_beta
        self.model = VHVMForecastNet(
            target_dim=self.input_size,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            min_scale=min_scale,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

    def forward(self, input):
        return self.model(input)

    def loss(self, batch_data, **kwargs):
        self.model.train()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        target = batch_data.future_target_cdf

        x_mu, y_mu, q_mu, q_logvar, p_mu, p_logvar, x_logvar, y_logvar = self.forward(x)
        rec_loss = self.model.gaussian_nll(x, x_mu, x_logvar, reduction="mean")
        pred_loss = self.model.gaussian_nll(target, y_mu, y_logvar, reduction="mean")
        kld_loss = self.model.gaussian_kl(q_mu, q_logvar, p_mu, p_logvar)

        print(
            f"rec_nll: {rec_loss:.4f}, pred_nll: {pred_loss:.4f}, "
            f"kld_loss: {kld_loss:.4f}"
        )
        return rec_loss + pred_loss + self.weight_beta * kld_loss

    def forecast(self, batch_data, num_samples=None):
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]

        with torch.no_grad():
            if num_samples is not None:
                outputs = self.model.sample(x, num_samples)
            else:
                outputs = self.forward(x)
                outputs = outputs[1].unsqueeze(1)
        return outputs
