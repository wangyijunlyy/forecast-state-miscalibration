"""Residual-state upgrade of the original K2VAE VolDy-VAE backbone."""

import torch
from torch import nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.VolDy_VAE import VolDy_VAE


class VolDy_VAE_Upgrade(VolDy_VAE):
    """Keep the original location/continuous-scale path and add a path state."""

    def __init__(self, config):
        super().__init__(config)
        state_dim = getattr(config, "state_dim", self.dynamic_dim)
        self.residual_encoder = nn.GRU(
            self.dynamic_dim + self.enc_in, state_dim, batch_first=True
        )
        self.state_dynamics = nn.GRU(self.dynamic_dim, state_dim, batch_first=True)
        self.initial_state = nn.Linear(state_dim, 2)
        self.state_transition = nn.Linear(state_dim, 4)
        self.scale_gap = nn.Linear(state_dim, self.patch_len * self.enc_in)
        nn.init.zeros_(self.scale_gap.weight)
        nn.init.constant_(self.scale_gap.bias, -3.0)

    def _residual_features(self, x, x_mu):
        # Residual energy is measured in the same normalized units as the encoder.
        normalizer = self.revin.stdev
        if self.revin.affine:
            normalizer = normalizer / self.revin.affine_weight.abs().view(1, 1, -1).clamp_min(1e-8)
        residual = (x - x_mu.detach()) / normalizer.clamp_min(1e-6)
        if self.padding_len:
            residual = torch.cat((residual[:, -self.padding_len:], residual), dim=1)
        energy = residual.reshape(x.shape[0], self.freq, self.patch_len, self.enc_in).square().mean(dim=2)
        return torch.log1p(energy)

    def decode_with_state(self, x, z):
        # This is exactly the original decoder, including its history-to-future GRU.
        x_mu, y_mu, x_sigma, base_sigma = super().decode(z)
        evidence = self._residual_features(x, x_mu)
        features = torch.cat((z, evidence), dim=-1)
        _, h0 = self.residual_encoder(features)
        future_z = self.future_proj(z.reshape(z.shape[0], -1)).reshape(
            z.shape[0], self.step, self.dynamic_dim
        )
        state, _ = self.state_dynamics(future_z, h0)
        initial_logits = self.initial_state(h0.squeeze(0))
        transition_logits = self.state_transition(state).reshape(-1, self.step, 2, 2)

        # Bounded symmetric modulation protects the original continuous scale.
        gap = 0.5 * torch.sigmoid(self.scale_gap(state))
        gap = gap.reshape(-1, self.step * self.patch_len, self.enc_in)[:, :self.pred_len]
        regime_sigma = torch.stack((base_sigma * torch.exp(-gap),
                                    base_sigma * torch.exp(gap)), dim=1)
        return {
            "x_mu": x_mu, "y_mu": y_mu, "x_sigma": x_sigma,
            "base_sigma": base_sigma, "regime_sigma": regime_sigma,
            "initial_logits": initial_logits,
            "transition_logits": transition_logits,
        }

    def forward(self, x):
        mu_z, logvar_z, z = self.encode(x)
        result = self.decode_with_state(x, z)
        result["mu_z"] = mu_z
        result["logvar_z"] = logvar_z
        return result

    def state_probs(self, result):
        probability = result["initial_logits"].softmax(-1)
        probabilities = []
        transitions = result["transition_logits"].softmax(-1)
        for patch in range(self.step):
            if patch:
                probability = torch.bmm(probability.unsqueeze(1), transitions[:, patch]).squeeze(1)
            probabilities.append(probability)
        return torch.stack(probabilities, dim=1)

    def marginal_nll(self, target, result):
        mu = result["y_mu"]
        sigma = result["regime_sigma"].clamp_min(1e-6)
        nll = torch.log(sigma) + (target.unsqueeze(1) - mu.unsqueeze(1)).square() / (2 * sigma.square())
        probs = self.state_probs(result).repeat_interleave(self.patch_len, dim=1)[:, :self.pred_len]
        log_prob = probs.clamp_min(1e-8).log().permute(0, 2, 1).unsqueeze(-1)
        return -torch.logsumexp(log_prob - nll, dim=1).mean()

    def sample(self, x, num_samples=100, chunk_size=25):
        batch = x.shape[0]
        mu_z, logvar_z, _ = self.encode(x)
        std_z = torch.exp(0.5 * logvar_z)
        saved = {name: getattr(self.revin, name) for name in ("mean", "stdev", "last")
                 if hasattr(self.revin, name)}
        outputs = []
        try:
            for start in range(0, num_samples, chunk_size):
                count = min(chunk_size, num_samples - start)
                z = mu_z[:, None] + torch.randn(batch, count, self.freq, self.dynamic_dim,
                                                 device=x.device, dtype=x.dtype) * std_z[:, None]
                z = z.reshape(batch * count, self.freq, self.dynamic_dim)
                x_repeat = x.repeat_interleave(count, dim=0)
                for name, value in saved.items():
                    setattr(self.revin, name, value.repeat_interleave(count, dim=0))
                result = self.decode_with_state(x_repeat, z)
                initial = result["initial_logits"].softmax(-1)
                transitions = result["transition_logits"].softmax(-1)
                path = []
                regime = torch.multinomial(initial, 1).squeeze(-1)
                for patch in range(self.step):
                    if patch:
                        row = transitions[torch.arange(batch * count, device=x.device), patch, regime]
                        regime = torch.multinomial(row, 1).squeeze(-1)
                    path.append(regime)
                path = torch.stack(path, dim=1).repeat_interleave(self.patch_len, dim=1)[:, :self.pred_len]
                selected = result["regime_sigma"].permute(0, 2, 1, 3)
                sigma = selected.gather(2, path[:, :, None, None].expand(-1, -1, 1, self.enc_in)).squeeze(2)
                draw = result["y_mu"] + torch.randn_like(sigma) * sigma
                outputs.append(draw.reshape(batch, count, self.pred_len, self.enc_in))
        finally:
            for name, value in saved.items():
                setattr(self.revin, name, value)
        return torch.cat(outputs, dim=1)
