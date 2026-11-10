"""Variance-preserving residual-state upgrade of original VolDy-VAE."""

import torch

from probts.model.nn.prob.k2VAE.VolDy_VAE_Upgrade import VolDy_VAE_Upgrade


class VolDy_VAE_Upgrade_V2(VolDy_VAE_Upgrade):
    def decode_with_state(self, x, z):
        result = super().decode_with_state(x, z)
        probabilities = self.state_probs(result).repeat_interleave(
            self.patch_len, dim=1
        )[:, :self.pred_len]
        probabilities = probabilities.permute(0, 2, 1).unsqueeze(-1)
        scale = result["regime_sigma"]
        mixture_variance = (probabilities * scale.square()).sum(dim=1)
        multiplier = result["base_sigma"] / torch.sqrt(mixture_variance.clamp_min(1e-12))
        result["regime_sigma"] = scale * multiplier.unsqueeze(1)
        return result
