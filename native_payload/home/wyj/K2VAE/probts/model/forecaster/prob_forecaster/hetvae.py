"""
HeTVAE-style baseline adapted to the ProbTS forecasting interface.

The original HeTVAE code is designed for irregular time-series interpolation:
observed context points are encoded and target query points are decoded with a
heteroscedastic Gaussian likelihood.  This wrapper converts the same idea to
long-term forecasting by treating the lookback window as context observations
and the prediction horizon as future query points.
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


class HeTVAEForecastNet(nn.Module):
    def __init__(
        self,
        target_dim,
        context_length,
        prediction_length,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
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
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.posterior_mu = nn.Linear(hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, latent_dim)

        dec_in_dim = hidden_dim + hidden_dim + latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
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

    def encode(self, x):
        batch_size, length, _ = x.shape
        total_length = self.context_length + self.prediction_length
        t = self._time_grid(
            batch_size,
            length,
            start_index=0,
            total_length=total_length,
            device=x.device,
            dtype=x.dtype,
        )
        tokens = torch.cat([self.value_proj(x), self.time_emb(t)], dim=-1)
        _, h = self.encoder(tokens)
        context = self.context_proj(h[-1])
        mu = self.posterior_mu(context)
        logvar = self.posterior_logvar(context).clamp(-8.0, 5.0)
        return context, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

    def decode(self, context, z, query_length, start_index):
        batch_size = context.shape[0]
        total_length = self.context_length + self.prediction_length
        dtype = context.dtype
        t = self._time_grid(
            batch_size,
            query_length,
            start_index=start_index,
            total_length=total_length,
            device=context.device,
            dtype=dtype,
        )
        time_feat = self.time_emb(t)
        context_feat = context.unsqueeze(1).expand(-1, query_length, -1)
        z_feat = z.unsqueeze(1).expand(-1, query_length, -1)
        h = self.decoder(torch.cat([time_feat, context_feat, z_feat], dim=-1))
        mean = self.mean_head(h)
        raw_scale = self.scale_head(h)
        scale = F.softplus(raw_scale) + self.min_scale
        logvar = (2.0 * torch.log(scale)).clamp(self.logvar_min, self.logvar_max)
        return mean, logvar

    def forward(self, x):
        context, mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_mu, x_logvar = self.decode(
            context,
            z,
            query_length=self.context_length,
            start_index=0,
        )
        y_mu, y_logvar = self.decode(
            context,
            z,
            query_length=self.prediction_length,
            start_index=self.context_length,
        )
        return x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar

    def sample(self, x, num_samples):
        context, mu_z, logvar_z = self.encode(x)
        std_z = torch.exp(0.5 * logvar_z)
        samples = []
        for _ in range(num_samples):
            z = mu_z + torch.randn_like(std_z) * std_z
            y_mu, y_logvar = self.decode(
                context,
                z,
                query_length=self.prediction_length,
                start_index=self.context_length,
            )
            y_scale = torch.exp(0.5 * y_logvar)
            y = y_mu + torch.randn_like(y_scale) * y_scale
            samples.append(y)
        return torch.stack(samples, dim=1)

    @staticmethod
    def gaussian_nll(target, mean, logvar, reduction="mean"):
        loss = 0.5 * (logvar + (target - mean).pow(2) * torch.exp(-logvar))
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def kl_divergence(mu, logvar):
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=-1).mean()


class HeTVAEModel(Forecaster):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        dropout=0.1,
        weight_beta=0.01,
        min_scale=1e-4,
        logvar_min=-10.0,
        logvar_max=5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weight_beta = weight_beta
        self.model = HeTVAEForecastNet(
            target_dim=self.input_size,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
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

        x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar = self.forward(x)
        rec_loss = self.model.gaussian_nll(x, x_mu, x_logvar, reduction="mean")
        pred_loss = self.model.gaussian_nll(target, y_mu, y_logvar, reduction="mean")
        kld_loss = self.model.kl_divergence(mu_z, logvar_z)

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
