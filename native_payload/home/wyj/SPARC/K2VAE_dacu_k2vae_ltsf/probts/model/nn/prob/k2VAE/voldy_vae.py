import math

import torch
import torch.nn as nn

from probts.model.nn.prob.k2VAE.RevIN import RevIN
from probts.model.nn.prob.k2VAE.koopman import MLP


class VolatilityDynamics(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, xi=1e-6):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()
        self.xi = xi

    def forward(self, z: torch.Tensor, h0: torch.Tensor = None):
        gru_out, h_n = self.gru(z, h0)
        sigma = self.softplus(self.linear(gru_out)) + self.xi
        return sigma, h_n


class VolDy_VAE(nn.Module):
    """Location-scale VAE with recurrent volatility dynamics."""

    def __init__(self, config):
        super().__init__()
        self.input_len = config.seq_len
        self.patch_len = config.patch_len
        self.dynamic_dim = config.dynamic_dim
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars
        self.freq = math.ceil(self.input_len / self.patch_len)
        self.pred_len = config.pred_len
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len

        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim * 2,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.future_proj = MLP(
            f_in=self.freq * self.dynamic_dim,
            f_out=self.step * self.dynamic_dim,
            activation="relu",
            hidden_dim=max(1, self.freq * self.dynamic_dim // 2),
            hidden_layers=1,
        )
        self.decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.volatility_dynamics = VolatilityDynamics(
            input_dim=self.dynamic_dim,
            hidden_dim=getattr(config, "vol_hidden_dim", self.dynamic_dim),
            output_dim=self.patch_len * self.enc_in,
        )
        self.revin = RevIN(self.enc_in)

    def encode(self, x: torch.Tensor):
        batch_size, length, _ = x.shape
        x_norm = self.revin(x, "norm")
        if self.padding_len > 0:
            padded = torch.cat([x_norm[:, length - self.padding_len :, :], x_norm], dim=1)
        else:
            padded = x_norm
        patches = padded.chunk(self.freq, dim=1)
        patches = torch.stack(patches, dim=1).reshape(batch_size, self.freq, -1)
        encoded = self.encoder(patches)
        mu_z, logvar_z = torch.chunk(encoded, 2, dim=-1)
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        z = mu_z + eps * std_z if self.training else mu_z
        return mu_z, logvar_z, z

    def decode(self, z: torch.Tensor):
        batch_size = z.shape[0]
        x_mu_raw = self.decoder(z).reshape(batch_size, self.freq, self.patch_len, self.enc_in)
        x_mu_raw = x_mu_raw.reshape(batch_size, -1, self.enc_in)[:, : self.input_len]
        x_mu = self.revin(x_mu_raw, "denorm")

        x_sigma_patches, h_vol = self.volatility_dynamics(z)
        x_sigma = x_sigma_patches.reshape(batch_size, self.freq, self.patch_len, self.enc_in)
        x_sigma = x_sigma.reshape(batch_size, -1, self.enc_in)[:, : self.input_len]

        z_flat = z.reshape(batch_size, -1)
        z_future = self.future_proj(z_flat).reshape(batch_size, self.step, self.dynamic_dim)
        y_mu_raw = self.decoder(z_future).reshape(batch_size, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(batch_size, -1, self.enc_in)[:, : self.pred_len]
        y_mu = self.revin(y_mu_raw, "denorm")

        y_sigma_patches, _ = self.volatility_dynamics(z_future, h0=h_vol)
        y_sigma = y_sigma_patches.reshape(batch_size, self.step, self.patch_len, self.enc_in)
        y_sigma = y_sigma.reshape(batch_size, -1, self.enc_in)[:, : self.pred_len]
        return x_mu, y_mu, x_sigma, y_sigma

    def forward(self, x: torch.Tensor):
        mu_z, logvar_z, z = self.encode(x)
        x_mu, y_mu, x_sigma, y_sigma = self.decode(z)
        return x_mu, y_mu, mu_z, logvar_z, x_sigma, y_sigma

    def sample(self, x: torch.Tensor, num_samples: int = 100):
        mu_z, logvar_z, _ = self.encode(x)
        std_z = torch.exp(0.5 * logvar_z)
        sample_list = []
        for _ in range(num_samples):
            z_sample = mu_z + torch.randn_like(std_z) * std_z
            _, y_mu, _, y_sigma = self.decode(z_sample)
            sample_list.append(y_mu + torch.randn_like(y_mu) * y_sigma)
        return torch.stack(sample_list, dim=1)

    def kl_divergence(self, mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    @staticmethod
    def gaussian_nll(x, mu, sigma, reduction: str = "mean"):
        sigma = torch.clamp(sigma, min=1e-6, max=1e3)
        nll = torch.log(sigma) + (x - mu).pow(2) / (2.0 * sigma.pow(2))
        nll = nll.sum(dim=[1, 2]) / (x.shape[1] * x.shape[2])
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        return nll
