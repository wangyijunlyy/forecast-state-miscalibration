import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from probts.model.nn.prob.k2VAE.RevIN import RevIN


class PatchEmbedding(nn.Module):
    """Segment time series into non-overlapping patches and project them."""

    def __init__(self, patch_len: int, d_model: int, in_channels: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, N, D]
        bsz, seq_len, channels = x.shape
        n_patches = seq_len // self.patch_len
        x = x[:, : n_patches * self.patch_len, :]
        x = x.reshape(bsz, n_patches, self.patch_len * channels)
        return self.proj(x)


class SpectralBandFilter(nn.Module):
    """Soft band-pass filter applied on the embedding dimension."""

    def __init__(self, d_model: int, low_cut: float, high_cut: float, sharpness: float = 20.0):
        super().__init__()
        self.d_model = d_model
        n_freqs = d_model // 2 + 1
        freqs = torch.linspace(0.0, 0.5, n_freqs)
        low_mask = torch.sigmoid(sharpness * (freqs - low_cut))
        high_mask = torch.sigmoid(sharpness * (high_cut - freqs))
        self.register_buffer("mask", low_mask * high_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft = x_fft * self.mask
        return torch.fft.irfft(x_fft, n=self.d_model, dim=-1)


class BandEncoder(nn.Module):
    """Encoder branch for one spectral band."""

    def __init__(self, d_model: int, d_latent: int, n_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(n_layers):
            layers.extend([nn.Linear(d_model, d_model), nn.GELU()])
        self.backbone = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(d_model, d_latent)
        self.fc_logvar = nn.Linear(d_model, d_latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        return mu, logvar, z


class SpectralLatentEncoder(nn.Module):
    """Stage 1: split latent representation into frequency-aware bands."""

    def __init__(self, d_model: int, d_latent: int, n_bands: int = 3, n_layers: int = 2):
        super().__init__()
        self.n_bands = n_bands
        self.d_per_band = d_latent // n_bands
        self.d_latent = self.d_per_band * n_bands

        edges = torch.linspace(0.0, 0.5, n_bands + 1)
        self.filters = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.prior_logvar = nn.Parameter(torch.linspace(-2.0, 1.0, n_bands))

        for i in range(n_bands):
            self.filters.append(SpectralBandFilter(d_model, edges[i].item(), edges[i + 1].item()))
            self.encoders.append(BandEncoder(d_model, self.d_per_band, n_layers))

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        z_parts, kls = [], []
        mus, logvars = [], []

        for i in range(self.n_bands):
            h_band = self.filters[i](patches)
            mu_i, logvar_i, z_i = self.encoders[i](h_band)
            z_parts.append(z_i)
            mus.append(mu_i)
            logvars.append(logvar_i)

            prior_var = torch.exp(self.prior_logvar[i])
            kl_i = -0.5 * torch.mean(
                1.0
                + logvar_i
                - self.prior_logvar[i]
                - mu_i.pow(2) / prior_var
                - torch.exp(logvar_i) / prior_var
            )
            kls.append(kl_i)

        z = torch.cat(z_parts, dim=-1)
        band_info = {
            "mus": mus,
            "logvars": logvars,
            "prior_logvar": self.prior_logvar.detach(),
        }
        return z, kls, band_info


class LatentDynamics(nn.Module):
    """Flatten past latents and map them to future latents."""

    def __init__(self, n_past: int, n_future: int, d_latent: int, d_hidden: int = 512):
        super().__init__()
        self.n_future = n_future
        self.d_latent = d_latent
        self.mlp = nn.Sequential(
            nn.Linear(n_past * d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_future * d_latent),
        )

    def forward(self, z_past: torch.Tensor) -> torch.Tensor:
        bsz = z_past.size(0)
        return self.mlp(z_past.reshape(bsz, -1)).reshape(bsz, self.n_future, self.d_latent)


class StochasticFrequencyGate(nn.Module):
    """Stage 2: input-conditioned stochastic gate in frequency space."""

    def __init__(self, n_freqs: int, n_channels: int, d_condition: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_channels = n_channels
        gate_dim = n_freqs * n_channels

        hidden = max(d_condition // 2, 16)
        self.posterior_net = nn.Sequential(
            nn.Linear(d_condition, hidden),
            nn.GELU(),
            nn.Linear(hidden, gate_dim * 2),
        )

        prior_mu = torch.linspace(3.0, -3.0, n_freqs).unsqueeze(-1).expand(n_freqs, n_channels)
        self.register_buffer("prior_mu", prior_mu.clone())
        self.prior_logvar = nn.Parameter(torch.full((n_freqs, n_channels), -1.0))

    def forward(
        self, spectral_condition: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = spectral_condition.size(0)
        params = self.posterior_net(spectral_condition)
        mu_logit, logvar_logit = params.chunk(2, dim=-1)
        mu_logit = mu_logit.reshape(bsz, self.n_freqs, self.n_channels)
        logvar_logit = logvar_logit.reshape(bsz, self.n_freqs, self.n_channels)

        if deterministic or not self.training:
            gate_logit = mu_logit
        else:
            std = torch.exp(0.5 * logvar_logit)
            gate_logit = mu_logit + std * torch.randn_like(std)

        gate = torch.sigmoid(gate_logit)

        prior_var = torch.exp(self.prior_logvar)
        post_var = torch.exp(logvar_logit)
        kl = 0.5 * (
            self.prior_logvar
            - logvar_logit
            + post_var / prior_var
            + (mu_logit - self.prior_mu).pow(2) / prior_var
            - 1.0
        ).mean()
        return gate, kl, mu_logit, logvar_logit


class SpectralConditioner(nn.Module):
    """Extract spectral summary features from the normalized lookback."""

    def __init__(self, seq_len: int, n_channels: int, d_output: int):
        super().__init__()
        n_freqs = seq_len // 2 + 1
        self.encoder = nn.Sequential(
            nn.Linear(n_freqs * n_channels, d_output),
            nn.GELU(),
            nn.Linear(d_output, d_output),
        )

    def forward(self, x_past: torch.Tensor) -> torch.Tensor:
        bsz = x_past.size(0)
        spectrum = torch.fft.rfft(x_past, dim=1).abs()
        return self.encoder(spectrum.reshape(bsz, -1))


class FreqConditionedDualHeadDecoder(nn.Module):
    """Stage 3: dual-head location-scale decoder conditioned on spectral features."""

    def __init__(
        self,
        d_latent: int,
        patch_len: int,
        n_patches: int,
        out_channels: int,
        d_decoder: int = 256,
        d_spectral_cond: int = 128,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = n_patches
        self.out_channels = out_channels
        self.seq_len = patch_len * n_patches
        self.d_spectral_cond = d_spectral_cond
        self.eps = 1e-6

        out_dim = patch_len * out_channels
        self.shared = nn.Sequential(
            nn.Linear(d_latent, d_decoder),
            nn.GELU(),
            nn.Linear(d_decoder, d_decoder),
            nn.GELU(),
        )
        self.low_loc = nn.Linear(d_decoder, out_dim)
        self.high_loc = nn.Linear(d_decoder, out_dim)
        self.low_scale = nn.Sequential(
            nn.Linear(d_decoder + d_spectral_cond, d_decoder // 2),
            nn.GELU(),
            nn.Linear(d_decoder // 2, out_dim),
            nn.Softplus(),
        )
        self.high_scale = nn.Sequential(
            nn.Linear(d_decoder + d_spectral_cond, d_decoder // 2),
            nn.GELU(),
            nn.Linear(d_decoder // 2, out_dim),
            nn.Softplus(),
        )

    def _to_seq(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        return x.reshape(bsz, self.n_patches * self.patch_len, self.out_channels)

    def forward(self, z: torch.Tensor, spectral_cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_patches, _ = z.shape
        h = self.shared(z)
        mu_low = self._to_seq(self.low_loc(h))
        mu_high = self._to_seq(self.high_loc(h))

        sc = spectral_cond.unsqueeze(1).expand(bsz, n_patches, self.d_spectral_cond)
        h_cond = torch.cat([h, sc], dim=-1)
        sigma_low = self._to_seq(self.low_scale(h_cond)) + self.eps
        sigma_high = self._to_seq(self.high_scale(h_cond)) + self.eps
        return mu_low, sigma_low, mu_high, sigma_high


class SpectralFusion(nn.Module):
    """Fuse dual-head outputs in the frequency domain."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.eps = 1e-6

    def forward(
        self,
        mu_low: torch.Tensor,
        sigma_low: torch.Tensor,
        mu_high: torch.Tensor,
        sigma_high: torch.Tensor,
        gate: torch.Tensor,
        gate_logvar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_low_f = torch.fft.rfft(mu_low, dim=1)
        mu_high_f = torch.fft.rfft(mu_high, dim=1)

        gate_inv = 1.0 - gate
        mu_fused_f = gate * mu_low_f + gate_inv * mu_high_f
        mu = torch.fft.irfft(mu_fused_f, n=self.seq_len, dim=1)

        var_low_f = torch.fft.rfft(sigma_low.pow(2), dim=1).abs()
        var_high_f = torch.fft.rfft(sigma_high.pow(2), dim=1).abs()
        var_fused_f = gate.pow(2) * var_low_f + gate_inv.pow(2) * var_high_f

        if gate_logvar is not None:
            gate_var = torch.sigmoid(gate_logvar).clamp(min=self.eps)
            disagreement = (mu_low_f - mu_high_f).abs().pow(2)
            var_fused_f = var_fused_f + gate_var * disagreement

        var_time = torch.fft.irfft(var_fused_f, n=self.seq_len, dim=1)
        sigma = var_time.clamp(min=self.eps).sqrt()
        return mu, sigma


class FDHVAEv2(nn.Module):
    """Frequency-aware Deep Heteroscedastic VAE v2 for ProbTS."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.n_vars
        self.lookback_len = config.seq_len
        self.horizon = config.pred_len
        self.patch_len = config.patch_len
        self.d_model = config.d_model
        self.n_bands = getattr(config, "n_bands", 3)
        self.n_encoder_layers = getattr(config, "n_encoder_layers", 2)
        self.d_decoder = getattr(config, "d_decoder", config.hidden_dim)
        self.d_spectral_cond = getattr(config, "d_spectral_cond", 128)

        self.n_past = math.ceil(self.lookback_len / self.patch_len)
        self.n_future = math.ceil(self.horizon / self.patch_len)
        self.lookback_pad = self.n_past * self.patch_len - self.lookback_len
        d_per_band = config.dynamic_dim // self.n_bands
        self.d_latent = d_per_band * self.n_bands

        self.revin = RevIN(self.in_channels)
        self.patch_embed = PatchEmbedding(self.patch_len, self.d_model, self.in_channels)
        self.spectral_encoder = SpectralLatentEncoder(
            self.d_model, self.d_latent, self.n_bands, self.n_encoder_layers
        )
        self.latent_dynamics = LatentDynamics(
            self.n_past, self.n_future, self.d_latent, d_hidden=max(self.d_model, self.d_latent * 2)
        )
        self.spectral_cond = SpectralConditioner(
            self.lookback_len, self.in_channels, self.d_spectral_cond
        )
        self.gate_rec = StochasticFrequencyGate(
            self.lookback_len // 2 + 1, self.in_channels, self.d_spectral_cond
        )
        self.gate_pred = StochasticFrequencyGate(
            self.horizon // 2 + 1, self.in_channels, self.d_spectral_cond
        )
        self.dec_rec = FreqConditionedDualHeadDecoder(
            self.d_latent,
            self.patch_len,
            self.n_past,
            self.in_channels,
            self.d_decoder,
            self.d_spectral_cond,
        )
        self.dec_pred = FreqConditionedDualHeadDecoder(
            self.d_latent,
            self.patch_len,
            self.n_future,
            self.in_channels,
            self.d_decoder,
            self.d_spectral_cond,
        )
        self.fusion_rec = SpectralFusion(self.lookback_len)
        self.fusion_pred = SpectralFusion(self.horizon)

    def _pad_for_patching(self, x: torch.Tensor, total_pad: int) -> torch.Tensor:
        if total_pad <= 0:
            return x
        return torch.cat([x[:, -total_pad:, :], x], dim=1)

    def forward(self, x_past: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = x_past.size(0)

        x_norm = self.revin(x_past, "norm")
        x_padded = self._pad_for_patching(x_norm, self.lookback_pad)
        patches = self.patch_embed(x_padded)
        z_past, kl_bands, band_info = self.spectral_encoder(patches)
        z_future = self.latent_dynamics(z_past)
        spec_cond = self.spectral_cond(x_norm)

        gate_rec, kl_gate_rec, _, gate_logvar_rec = self.gate_rec(spec_cond)
        gate_pred, kl_gate_pred, _, gate_logvar_pred = self.gate_pred(spec_cond)

        mu_low_r, sig_low_r, mu_hi_r, sig_hi_r = self.dec_rec(z_past, spec_cond)
        mu_low_p, sig_low_p, mu_hi_p, sig_hi_p = self.dec_pred(z_future, spec_cond)

        mu_rec_full, sigma_rec_full = self.fusion_rec(
            mu_low_r[:, -self.lookback_len :, :],
            sig_low_r[:, -self.lookback_len :, :],
            mu_hi_r[:, -self.lookback_len :, :],
            sig_hi_r[:, -self.lookback_len :, :],
            gate_rec,
            gate_logvar_rec,
        )
        mu_pred_full, sigma_pred_full = self.fusion_pred(
            mu_low_p[:, : self.horizon, :],
            sig_low_p[:, : self.horizon, :],
            mu_hi_p[:, : self.horizon, :],
            sig_hi_p[:, : self.horizon, :],
            gate_pred,
            gate_logvar_pred,
        )

        mu_rec = self.revin(mu_rec_full, "denorm")
        mu_pred = self.revin(mu_pred_full, "denorm")
        sigma_rec = sigma_rec_full * self.revin.stdev
        sigma_pred = sigma_pred_full * self.revin.stdev

        return {
            "mu_rec": mu_rec,
            "sigma_rec": sigma_rec,
            "mu_pred": mu_pred,
            "sigma_pred": sigma_pred,
            "kl_bands": kl_bands,
            "kl_gate_rec": kl_gate_rec,
            "kl_gate_pred": kl_gate_pred,
            "aux_rec": {
                "mu_low": mu_low_r[:, -self.lookback_len :, :],
                "sigma_low": sig_low_r[:, -self.lookback_len :, :],
                "mu_high": mu_hi_r[:, -self.lookback_len :, :],
                "sigma_high": sig_hi_r[:, -self.lookback_len :, :],
                "gate": gate_rec,
                "gate_logvar": gate_logvar_rec,
            },
            "aux_pred": {
                "mu_low": mu_low_p[:, : self.horizon, :],
                "sigma_low": sig_low_p[:, : self.horizon, :],
                "mu_high": mu_hi_p[:, : self.horizon, :],
                "sigma_high": sig_hi_p[:, : self.horizon, :],
                "gate": gate_pred,
                "gate_logvar": gate_logvar_pred,
            },
            "band_info": band_info,
            "z_past": z_past,
            "z_future": z_future,
            "spec_cond": spec_cond.reshape(bsz, -1),
        }

    @torch.no_grad()
    def forecast(self, x_past: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.train()

        all_mu, all_sigma = [], []
        for _ in range(n_samples):
            result = self.forward(x_past)
            all_mu.append(result["mu_pred"])
            all_sigma.append(result["sigma_pred"])

        mus = torch.stack(all_mu, dim=0)
        sigmas = torch.stack(all_sigma, dim=0)
        samples = mus + sigmas * torch.randn_like(sigmas)

        if not was_training:
            self.eval()
        return samples, mus.mean(0), sigmas.mean(0)

    @staticmethod
    def heteroscedastic_nll(target: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp(min=1e-6)
        var = sigma.pow(2)
        return (0.5 * (var.log() + (target - mu).pow(2) / var)).mean()

    @staticmethod
    def frequency_consistency_loss(mu_low: torch.Tensor, mu_high: torch.Tensor) -> torch.Tensor:
        mu_low_f = torch.fft.rfft(mu_low, dim=1)
        mu_high_f = torch.fft.rfft(mu_high, dim=1)
        n_freqs = mu_low_f.size(1)
        w = torch.linspace(0.0, 1.0, n_freqs, device=mu_low.device).view(1, n_freqs, 1)
        loss_low = (mu_low_f.abs().pow(2) * w).mean()
        loss_high = (mu_high_f.abs().pow(2) * (1.0 - w)).mean()
        return loss_low + loss_high


class FDHVAEv2Loss(nn.Module):
    """Combined objective for FDH-VAE v2."""

    def __init__(self, beta_band: float = 0.1, beta_gate: float = 0.05, lambda_freq: float = 0.01):
        super().__init__()
        self.beta_band = beta_band
        self.beta_gate = beta_gate
        self.lambda_freq = lambda_freq

    def forward(
        self, result: Dict[str, torch.Tensor], x_past: torch.Tensor, x_future: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_rec = FDHVAEv2.heteroscedastic_nll(x_past, result["mu_rec"], result["sigma_rec"])
        l_pred = FDHVAEv2.heteroscedastic_nll(x_future, result["mu_pred"], result["sigma_pred"])

        kl_bands: List[torch.Tensor] = result["kl_bands"]
        l_kl_band = sum(kl_bands)
        l_kl_gate = result["kl_gate_rec"] + result["kl_gate_pred"]
        aux = result["aux_pred"]
        l_freq = FDHVAEv2.frequency_consistency_loss(aux["mu_low"], aux["mu_high"])

        total = (
            l_rec
            + l_pred
            + self.beta_band * l_kl_band
            + self.beta_gate * l_kl_gate
            + self.lambda_freq * l_freq
        )

        loss_dict = {
            "total": total.item(),
            "rec_nll": l_rec.item(),
            "pred_nll": l_pred.item(),
            "kl_band_total": l_kl_band.item(),
            "kl_bands": [k.item() for k in kl_bands],
            "kl_gate": l_kl_gate.item(),
            "freq_consistency": l_freq.item(),
        }
        return total, loss_dict
