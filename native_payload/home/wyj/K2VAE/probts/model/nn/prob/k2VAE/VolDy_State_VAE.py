"""State-aware VolDy-VAE: residual-initialized two-regime volatility paths."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN


class VolDy_State_VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.channels = config.n_vars
        self.latent_dim = config.dynamic_dim
        self.state_dim = getattr(config, "vol_hidden_dim", self.latent_dim)
        self.scale_backbone = getattr(config, "scale_backbone", "state")
        if self.scale_backbone not in ("state", "legacy"):
            raise ValueError(f"Unsupported scale_backbone: {self.scale_backbone}")
        self.n_history = math.ceil(self.input_len / self.patch_len)
        self.n_future = math.ceil(self.pred_len / self.patch_len)
        self.history_padding = self.n_history * self.patch_len - self.input_len
        self.patch_features = self.patch_len * self.channels
        self.eps = 1e-6

        self.revin = RevIN(self.channels)
        self.encoder = MLP(
            f_in=self.patch_features, f_out=2 * self.latent_dim,
            activation="relu", hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
        )
        self.future_proj = MLP(
            f_in=self.n_history * self.latent_dim,
            f_out=self.n_future * self.latent_dim,
            activation="relu",
            hidden_dim=max(1, self.n_history * self.latent_dim // 2),
            hidden_layers=1,
        )
        self.location = MLP(
            f_in=self.latent_dim, f_out=self.patch_features,
            activation="relu", hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
        )
        self.history_gru = nn.GRU(
            self.latent_dim + self.channels, self.state_dim, batch_first=True
        )
        self.history_scale = nn.Linear(self.state_dim, self.patch_features)
        self.initial_regime = nn.Linear(self.state_dim, 2)
        self.future_gru = nn.GRU(self.latent_dim, self.state_dim, batch_first=True)
        self.transition = nn.Linear(self.state_dim + self.latent_dim, 4)
        self.low_scale = nn.Linear(self.state_dim, self.patch_features)
        self.scale_gap = nn.Linear(self.state_dim, self.patch_features)
        if self.scale_backbone == "legacy":
            self.base_gru = nn.GRU(self.latent_dim, self.state_dim, batch_first=True)
            self.base_scale_head = nn.Linear(self.state_dim, self.patch_features)

    def _scale_factor(self):
        factor = self.revin.stdev
        if self.revin.affine:
            factor = factor / self.revin.affine_weight.view(1, 1, -1).abs().clamp_min(1e-8)
        return factor

    def _history_patches(self, normalized):
        if self.history_padding:
            # Match the original encoder's left-padding convention.
            normalized = torch.cat(
                [normalized[:, -self.history_padding:, :], normalized], dim=1
            )
        return normalized.reshape(
            normalized.shape[0], self.n_history, self.patch_len, self.channels
        )

    def encode(self, x):
        normalized = self.revin(x, "norm")
        patches = self._history_patches(normalized)
        encoded = self.encoder(patches.flatten(2))
        latent_mean, latent_logvar = encoded.chunk(2, dim=-1)
        if self.training:
            z = latent_mean + torch.randn_like(latent_mean) * torch.exp(0.5 * latent_logvar)
        else:
            z = latent_mean
        return normalized, patches, latent_mean, latent_logvar, z

    def decode_latents(self, normalized, patches, z):
        batch = z.shape[0]
        history_location = self.location(z).reshape(
            batch, self.n_history, self.patch_len, self.channels
        )
        # The residual cue is observed history only. Stop gradient through
        # the location prediction, as specified in the manuscript.
        innovations = torch.log(
            self.eps + (patches - history_location.detach()).square().mean(dim=2)
        )
        history_input = torch.cat([z, innovations], dim=-1)
        history_states, last_state = self.history_gru(history_input)
        if self.scale_backbone == "legacy":
            base_history_states, base_last_state = self.base_gru(z)
            history_scale = F.softplus(self.base_scale_head(base_history_states)) + self.eps
        else:
            history_scale = F.softplus(self.history_scale(history_states)) + self.eps
        history_scale = history_scale.reshape(
            batch, self.n_history, self.patch_len, self.channels
        )

        future_z = self.future_proj(z.flatten(1)).reshape(
            batch, self.n_future, self.latent_dim
        )
        future_location = self.location(future_z).reshape(
            batch, self.n_future, self.patch_len, self.channels
        )
        future_states, _ = self.future_gru(future_z, last_state)
        transition_input = torch.cat([future_states, future_z], dim=-1)
        transition_logprob = F.log_softmax(
            self.transition(transition_input).reshape(batch, self.n_future, 2, 2),
            dim=-1,
        )
        initial_logprob = F.log_softmax(self.initial_regime(last_state.squeeze(0)), dim=-1)
        if self.scale_backbone == "legacy":
            base_future_states, _ = self.base_gru(future_z, base_last_state)
            base_scale = F.softplus(self.base_scale_head(base_future_states)) + self.eps
            # A small, ordered log-scale perturbation starts close to the
            # original continuous-scale model while retaining distinct states.
            gap = F.softplus(self.scale_gap(future_states) - 3.0)
            low = base_scale * torch.exp(-gap)
            high = base_scale * torch.exp(gap)
        else:
            low = F.softplus(self.low_scale(future_states)) + self.eps
            high = low + F.softplus(self.scale_gap(future_states))
        scales = torch.stack([low, high], dim=2).reshape(
            batch, self.n_future, 2, self.patch_len, self.channels
        )

        history_mean = self.revin(
            history_location.flatten(1, 2)[:, -self.input_len:, :], "denorm"
        )
        history_sigma = history_scale.flatten(1, 2)[:, -self.input_len:, :] * self._scale_factor()
        future_mean = self.revin(
            future_location.flatten(1, 2)[:, :self.pred_len, :], "denorm"
        )
        future_sigma = scales * self._scale_factor().unsqueeze(2).unsqueeze(2)
        out = {
            "history_mean": history_mean,
            "history_sigma": history_sigma,
            "future_mean": future_mean,
            "future_sigma": future_sigma,
            "initial_logprob": initial_logprob,
            "transition_logprob": transition_logprob,
        }
        if self.scale_backbone == "legacy":
            out["base_future_sigma"] = (
                base_scale.reshape(batch, self.n_future, self.patch_len, self.channels)
                .flatten(1, 2)[:, :self.pred_len, :] * self._scale_factor()
            )
        return out

    def forward(self, x):
        normalized, patches, latent_mean, latent_logvar, z = self.encode(x)
        out = self.decode_latents(normalized, patches, z)
        out["latent_mean"] = latent_mean
        out["latent_logvar"] = latent_logvar
        return out

    @staticmethod
    def gaussian_nll(target, mean, sigma):
        sigma = sigma.clamp(1e-6, 1e3)
        return (torch.log(sigma) + 0.5 * ((target - mean) / sigma).square()).mean()

    def joint_nll(self, target, out):
        batch, horizon, channels = target.shape
        padded_horizon = self.n_future * self.patch_len
        pad = padded_horizon - horizon
        if pad:
            target = F.pad(target, (0, 0, 0, pad))
            mean = F.pad(out["future_mean"], (0, 0, 0, pad))
        else:
            mean = out["future_mean"]
        target = target.reshape(batch, self.n_future, self.patch_len, channels)
        mean = mean.reshape(batch, self.n_future, self.patch_len, channels)
        sigma = out["future_sigma"].clamp(1e-6, 1e3)
        log_emit = -torch.log(sigma) - 0.5 * ((target.unsqueeze(2) - mean.unsqueeze(2)) / sigma).square()
        if pad:
            log_emit[:, -1, :, -pad:, :] = 0.0
        log_emit = log_emit.sum(dim=(-1, -2))
        alpha = out["initial_logprob"]
        for step in range(self.n_future):
            alpha = log_emit[:, step, :] + torch.logsumexp(
                alpha.unsqueeze(-1) + out["transition_logprob"][:, step, :, :],
                dim=1,
            )
        return -torch.logsumexp(alpha, dim=-1).mean() / (horizon * channels)

    def marginal_nll(self, target, out):
        """Pointwise Gaussian mixture under the propagated regime prior.

        This ablation retains the Markov transition and path sampler, but
        prevents a whole high-dimensional patch from forcing one regime during
        likelihood training.
        """
        log_state = out["initial_logprob"]
        prior = []
        for step in range(self.n_future):
            log_state = torch.logsumexp(
                log_state.unsqueeze(-1)
                + out["transition_logprob"][:, step], dim=1
            )
            prior.append(log_state)
        log_prior = torch.stack(prior, dim=1).unsqueeze(-1).unsqueeze(-1)
        sigma = out["future_sigma"].clamp(1e-6, 1e3)
        batch, horizon, channels = target.shape
        padded = self.n_future * self.patch_len - horizon
        if padded:
            target = F.pad(target, (0, 0, 0, padded))
            mean = F.pad(out["future_mean"], (0, 0, 0, padded))
        else:
            mean = out["future_mean"]
        target = target.reshape(batch, self.n_future, self.patch_len, channels)
        mean = mean.reshape(batch, self.n_future, self.patch_len, channels)
        log_emit = -torch.log(sigma) - 0.5 * (
            (target.unsqueeze(2) - mean.unsqueeze(2)) / sigma
        ).square()
        log_mix = torch.logsumexp(log_prior + log_emit, dim=2)
        return -log_mix.flatten(1, 2)[:, :horizon].mean()

    @staticmethod
    def kl_divergence(mean, logvar):
        return (-0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1)).mean()

    @torch.no_grad()
    def sample(self, x, num_samples=100):
        # Each latent draw has its own historical residual cue and regime path.
        batch = x.shape[0]
        normalized, patches, latent_mean, latent_logvar, _ = self.encode(x)
        samples = []
        saved = {name: getattr(self.revin, name) for name in ("mean", "stdev")}
        try:
            # Bound the largest [B*S, M, 2, P, C] tensor. This matters for
            # high-channel datasets such as Traffic while preserving draws.
            elements_per_draw = batch * self.n_future * 2 * self.patch_len * self.channels
            chunk_size = min(num_samples, max(1, 120_000_000 // elements_per_draw))
            latent_std = torch.exp(0.5 * latent_logvar)
            for start in range(0, num_samples, chunk_size):
                count = min(chunk_size, num_samples - start)
                self.revin.mean = saved["mean"].repeat_interleave(count, dim=0)
                self.revin.stdev = saved["stdev"].repeat_interleave(count, dim=0)
                mean = latent_mean.repeat_interleave(count, dim=0)
                std = latent_std.repeat_interleave(count, dim=0)
                z = mean + torch.randn_like(mean) * std
                out = self.decode_latents(
                    normalized.repeat_interleave(count, dim=0),
                    patches.repeat_interleave(count, dim=0), z,
                )
                total = batch * count
                previous = torch.distributions.Categorical(logits=out["initial_logprob"]).sample()
                regimes = []
                row = torch.arange(total, device=x.device)
                for step in range(self.n_future):
                    logits = out["transition_logprob"][row, step, previous]
                    previous = torch.distributions.Categorical(logits=logits).sample()
                    regimes.append(previous)
                regimes = torch.stack(regimes, dim=1)
                selected = out["future_sigma"][row[:, None], torch.arange(self.n_future, device=x.device)[None, :], regimes]
                selected = selected.flatten(1, 2)[:, :self.pred_len, :]
                forecast = out["future_mean"] + torch.randn_like(out["future_mean"]) * selected
                samples.append(forecast.reshape(batch, count, self.pred_len, self.channels))
            return torch.cat(samples, dim=1)
        finally:
            self.revin.mean, self.revin.stdev = saved["mean"], saved["stdev"]
