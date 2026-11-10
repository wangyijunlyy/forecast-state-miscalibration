import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.koopman import MLP


class IdentityRevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()

    def forward(self, x, mode: str):
        return x


class DisentangledVolatilityVAE(nn.Module):
    """
    Trend-volatility disentangled VAE for probabilistic forecasting.

    Main components:
      1. Separate trend and volatility latent dynamics.
      2. Future volatility propagation from residual regime and latent state.
      3. Calibration-aware losses exposed as helper methods for the forecaster.
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
        self.regime_dim = int(getattr(config, "regime_dim", 32))
        self.vol_hidden_dim = int(getattr(config, "vol_hidden_dim", self.dynamic_dim))
        self.trend_kernel = int(getattr(config, "trend_kernel", 25))
        self.calibration_temperature = float(
            getattr(config, "calibration_temperature", 0.1)
        )

        self.freq = math.ceil(self.input_len / self.patch_len)
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len

        patch_dim = self.patch_len * self.enc_in

        self.trend_encoder = MLP(
            f_in=patch_dim,
            f_out=2 * self.dynamic_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.vol_encoder = MLP(
            f_in=patch_dim,
            f_out=2 * self.dynamic_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        self.trend_future_proj = MLP(
            f_in=self.freq * self.dynamic_dim,
            f_out=self.step * self.dynamic_dim,
            activation="relu",
            hidden_dim=max(self.dynamic_dim, self.freq * self.dynamic_dim // 2),
            hidden_layers=1,
        )
        self.trend_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=patch_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        self.past_vol_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=patch_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.regime_encoder = MLP(
            f_in=3 * self.enc_in,
            f_out=self.regime_dim,
            activation="relu",
            hidden_dim=max(self.hidden_dim // 2, self.regime_dim),
            hidden_layers=1,
        )
        self.vol_init = MLP(
            f_in=self.freq * self.dynamic_dim + self.regime_dim,
            f_out=self.vol_hidden_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=1,
        )
        self.vol_gru = nn.GRUCell(self.dynamic_dim + self.regime_dim, self.vol_hidden_dim)
        self.vol_out = MLP(
            f_in=self.vol_hidden_dim,
            f_out=patch_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=1,
        )

        self.revin = IdentityRevIN(self.enc_in)

    def moving_average(self, x):
        kernel = min(self.trend_kernel, x.shape[1])
        if kernel % 2 == 0:
            kernel -= 1
        if kernel <= 1:
            return x

        pad = kernel // 2
        x_ch = x.transpose(1, 2)
        x_pad = F.pad(x_ch, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=kernel, stride=1)
        return trend.transpose(1, 2)

    def create_patches(self, x):
        B, L, _ = x.shape
        if self.padding_len > 0:
            x = torch.cat([x[:, L - self.padding_len :, :], x], dim=1)
        patches = x.chunk(self.freq, dim=1)
        patches = torch.stack(patches, dim=1)
        return patches.reshape(B, self.freq, -1)

    def unpatch(self, patch_values, length, steps):
        B = patch_values.shape[0]
        values = patch_values.reshape(B, steps, self.patch_len, self.enc_in)
        return values.reshape(B, -1, self.enc_in)[:, :length]

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x):
        x_norm = self.revin(x, "norm")
        x_trend = self.moving_average(x_norm)
        x_vol = x_norm - x_trend

        trend_encoded = self.trend_encoder(self.create_patches(x_trend))
        vol_encoded = self.vol_encoder(self.create_patches(x_vol))

        trend_mu, trend_logvar = torch.chunk(trend_encoded, 2, dim=-1)
        vol_mu, vol_logvar = torch.chunk(vol_encoded, 2, dim=-1)
        z_trend = self.reparameterize(trend_mu, trend_logvar)
        z_vol = self.reparameterize(vol_mu, vol_logvar)
        return trend_mu, trend_logvar, z_trend, vol_mu, vol_logvar, z_vol

    def residual_regime(self, x_norm, x_mu_raw):
        residual = x_norm - x_mu_raw
        stats = torch.cat(
            [
                residual.abs().mean(dim=1),
                residual.std(dim=1, unbiased=False),
                residual.abs().amax(dim=1),
            ],
            dim=-1,
        )
        return self.regime_encoder(stats)

    def decode(self, z_trend, z_vol, x):
        B = z_trend.shape[0]
        x_norm = self.revin(x, "norm")

        x_mu_raw = self.unpatch(self.trend_decoder(z_trend), self.input_len, self.freq)
        x_mu = self.revin(x_mu_raw, "denorm")

        z_trend_future = self.trend_future_proj(z_trend.reshape(B, -1))
        z_trend_future = z_trend_future.reshape(B, self.step, self.dynamic_dim)
        y_mu_raw = self.unpatch(
            self.trend_decoder(z_trend_future), self.pred_len, self.step
        )
        y_mu = self.revin(y_mu_raw, "denorm")

        x_logvar = self.unpatch(self.past_vol_decoder(z_vol), self.input_len, self.freq)
        regime = self.residual_regime(x_norm, x_mu_raw)

        h = self.vol_init(torch.cat([z_vol.reshape(B, -1), regime], dim=-1))
        y_logvar_patches = []
        for t in range(self.step):
            step_input = torch.cat([z_trend_future[:, t, :], regime], dim=-1)
            h = self.vol_gru(step_input, h)
            y_logvar_patches.append(self.vol_out(h))
        y_logvar = self.unpatch(torch.stack(y_logvar_patches, dim=1), self.pred_len, self.step)

        return x_mu, y_mu, x_logvar, y_logvar, regime

    def forward(self, x):
        trend_mu, trend_logvar, z_trend, vol_mu, vol_logvar, z_vol = self.encode(x)
        x_mu, y_mu, x_logvar, y_logvar, regime = self.decode(z_trend, z_vol, x)
        return (
            x_mu,
            y_mu,
            trend_mu,
            trend_logvar,
            vol_mu,
            vol_logvar,
            x_logvar,
            y_logvar,
            z_trend,
            z_vol,
            regime,
        )

    def sample(self, x, num_samples=100):
        trend_mu, trend_logvar, _, vol_mu, vol_logvar, _ = self.encode(x)
        trend_std = torch.exp(0.5 * trend_logvar)
        vol_std = torch.exp(0.5 * vol_logvar)

        samples = []
        for _ in range(num_samples):
            z_trend = trend_mu + torch.randn_like(trend_std) * trend_std
            z_vol = vol_mu + torch.randn_like(vol_std) * vol_std
            _, y_mu, _, y_logvar, _ = self.decode(z_trend, z_vol, x)
            std_y = torch.exp(0.5 * torch.clamp(y_logvar, -10.0, 10.0))
            samples.append(y_mu + torch.randn_like(std_y) * std_y)
        return torch.stack(samples, dim=1)

    @staticmethod
    def gaussian_nll(x, mu, logvar, reduction="mean"):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        nll = 0.5 * (math.log(2 * math.pi) + logvar + (x - mu).pow(2) / logvar.exp())
        nll = nll.sum(dim=[1, 2])
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        return nll

    @staticmethod
    def kl_divergence(mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    @staticmethod
    def disentanglement_loss(z_trend, z_vol):
        zt = z_trend.reshape(z_trend.shape[0], -1)
        zv = z_vol.reshape(z_vol.shape[0], -1)
        cosine = F.cosine_similarity(zt, zv, dim=-1)
        return cosine.pow(2).mean()

    @staticmethod
    def mutual_information_proxy(z_trend, z_vol):
        zt = z_trend.reshape(z_trend.shape[0], -1)
        zv = z_vol.reshape(z_vol.shape[0], -1)
        zt = (zt - zt.mean(dim=0, keepdim=True)) / (zt.std(dim=0, keepdim=True) + 1e-6)
        zv = (zv - zv.mean(dim=0, keepdim=True)) / (zv.std(dim=0, keepdim=True) + 1e-6)
        cross_corr = zt.transpose(0, 1).matmul(zv) / max(zt.shape[0] - 1, 1)
        return cross_corr.pow(2).mean()

    def calibration_sharpness_loss(
        self, target, mu, logvar, coverage=0.9, sharpness_weight=0.01
    ):
        z_score = {
            0.5: 0.67448975,
            0.8: 1.28155157,
            0.9: 1.64485363,
            0.95: 1.95996398,
        }.get(float(coverage), 1.64485363)
        std = torch.exp(0.5 * torch.clamp(logvar, -10.0, 10.0))
        lower = mu - z_score * std
        upper = mu + z_score * std
        temp = self.calibration_temperature
        soft_inside = torch.sigmoid((target - lower) / temp) * torch.sigmoid(
            (upper - target) / temp
        )
        calibration = (soft_inside.mean() - coverage) ** 2
        sharpness = (upper - lower).mean()
        return calibration + sharpness_weight * sharpness, calibration, sharpness
