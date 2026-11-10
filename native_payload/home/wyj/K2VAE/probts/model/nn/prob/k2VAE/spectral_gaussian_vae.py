import math

import torch
import torch.nn as nn

from probts.model.nn.prob.k2VAE.koopman import MLP


class IdentityRevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()

    def forward(self, x, mode: str):
        return x


class SpectralConditioner(nn.Module):
    """Encode the normalized input spectrum into a compact conditioning vector."""

    def __init__(self, seq_len: int, n_vars: int, d_condition: int):
        super().__init__()
        n_freqs = seq_len // 2 + 1
        self.net = nn.Sequential(
            nn.Linear(n_freqs * n_vars, d_condition),
            nn.GELU(),
            nn.Linear(d_condition, d_condition),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        spectrum = torch.fft.rfft(x, dim=1).abs()
        return self.net(spectrum.reshape(bsz, -1))


class SpectralGaussianVAE(nn.Module):
    """
    Incremental frequency-aware GaussianVAE.

    The latent path and mean decoder stay unchanged.
    Only the variance path is conditioned on spectral statistics extracted
    from the input sequence, which is a lower-risk way to inject
    frequency awareness.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.dynamic_dim = config.dynamic_dim
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars
        self.spectral_cond_dim = getattr(config, "spectral_cond_dim", 64)

        self.freq = math.ceil(self.input_len / self.patch_len)
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len

        self.future_proj = MLP(
            f_in=self.freq * self.dynamic_dim,
            f_out=self.step * self.dynamic_dim,
            activation="relu",
            hidden_dim=max(self.freq * self.dynamic_dim // 2, self.dynamic_dim),
            hidden_layers=1,
        )
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim * 2,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        self.decoder_mu = MLP(
            f_in=self.dynamic_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.decoder_logvar = MLP(
            f_in=self.dynamic_dim + self.spectral_cond_dim,
            f_out=self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.spectral_conditioner = SpectralConditioner(
            self.input_len, self.enc_in, self.spectral_cond_dim
        )
        self.revin = IdentityRevIN(self.enc_in)

    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if self.padding_len > 0:
            x = torch.cat([x[:, seq_len - self.padding_len :, :], x], dim=1)
        patches = x.chunk(self.freq, dim=1)
        patches = torch.stack(patches, dim=1)
        return patches.reshape(bsz, self.freq, -1)

    def encode(self, x: torch.Tensor):
        x_norm = self.revin(x, "norm")
        patches = self._create_patches(x_norm)
        encoded = self.encoder(patches)
        mu_z, logvar_z = torch.chunk(encoded, 2, dim=-1)
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        z = mu_z + eps * std_z if self.training else mu_z
        spectral_cond = self.spectral_conditioner(x_norm)
        return x_norm, mu_z, logvar_z, z, spectral_cond

    def _decode_variance(self, z: torch.Tensor, spectral_cond: torch.Tensor, n_steps: int, out_len: int):
        bsz = z.size(0)
        cond = spectral_cond.unsqueeze(1).expand(bsz, n_steps, self.spectral_cond_dim)
        logvar = self.decoder_logvar(torch.cat([z, cond], dim=-1))
        logvar = logvar.reshape(bsz, n_steps, self.patch_len, self.enc_in)
        logvar = logvar.reshape(bsz, -1, self.enc_in)[:, :out_len, :]
        return torch.clamp(logvar, min=-10.0, max=10.0)

    def decode(self, z: torch.Tensor, spectral_cond: torch.Tensor):
        bsz = z.size(0)

        x_mu = self.decoder_mu(z)
        x_mu = x_mu.reshape(bsz, self.freq, self.patch_len, self.enc_in)
        x_mu = x_mu.reshape(bsz, -1, self.enc_in)[:, : self.input_len, :]
        x_logvar = self._decode_variance(z, spectral_cond, self.freq, self.input_len)
        x_mu = self.revin(x_mu, "denorm")

        z_future = self.future_proj(z.reshape(bsz, -1)).reshape(bsz, self.step, self.dynamic_dim)
        y_mu = self.decoder_mu(z_future)
        y_mu = y_mu.reshape(bsz, self.step, self.patch_len, self.enc_in)
        y_mu = y_mu.reshape(bsz, -1, self.enc_in)[:, : self.pred_len, :]
        y_logvar = self._decode_variance(z_future, spectral_cond, self.step, self.pred_len)
        y_mu = self.revin(y_mu, "denorm")

        return x_mu, y_mu, x_logvar, y_logvar

    def forward(self, x: torch.Tensor):
        x_norm, mu_z, logvar_z, z, spectral_cond = self.encode(x)
        x_mu, y_mu, x_logvar, y_logvar = self.decode(z, spectral_cond)
        return x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar, spectral_cond, x_norm

    def sample(self, x: torch.Tensor, num_samples: int = 100):
        bsz = x.size(0)
        _, mu_z, logvar_z, _, spectral_cond = self.encode(x)
        std_z = torch.exp(0.5 * logvar_z)

        sample_list = []
        for _ in range(num_samples):
            z_sample = mu_z + torch.randn_like(std_z) * std_z
            z_future = self.future_proj(z_sample.reshape(bsz, -1)).reshape(bsz, self.step, self.dynamic_dim)
            y_mu = self.decoder_mu(z_future)
            y_mu = y_mu.reshape(bsz, self.step, self.patch_len, self.enc_in)
            y_mu = y_mu.reshape(bsz, -1, self.enc_in)[:, : self.pred_len, :]
            y_mu = self.revin(y_mu, "denorm")

            y_logvar = self._decode_variance(z_future, spectral_cond, self.step, self.pred_len)
            std_y = torch.exp(0.5 * y_logvar)
            y_sample = y_mu + torch.randn_like(std_y) * std_y
            sample_list.append(y_sample)

        return torch.stack(sample_list, dim=1)

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    @staticmethod
    def gaussian_nll(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean"):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        nll = 0.5 * (math.log(2 * math.pi) + logvar + (x - mu) ** 2 / logvar.exp())
        nll = nll.sum(dim=[1, 2])
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        return nll
