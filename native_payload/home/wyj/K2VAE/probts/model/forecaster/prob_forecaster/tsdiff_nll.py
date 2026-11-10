# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------------
# Portions of this file are derived from TSDiff
# - Source: https://github.com/amazon-science/unconditional-time-series-diffusion
# - Paper: Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting
# - License: Apache-2.0
#
# Modified to use dual-head prediction (mu + logvar) with Gaussian NLL loss.
# ---------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from probts.utils import extract
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.S4.s4_backbones import BackboneModel
from probts.utils import repeat
import sys


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)


def gaussian_nll(target, mu, logvar, mask=None, reduction="mean"):
    """
    Gaussian Negative Log-Likelihood loss.
    
    Args:
        target: [B, L, C] ground truth
        mu:     [B, L, C] predicted mean
        logvar: [B, L, C] predicted log variance
        mask:   [B, L, C] loss mask (1 = compute loss, 0 = ignore)
        reduction: 'mean' or 'sum'
    
    Returns:
        scalar loss
    """
    # Clamp logvar for numerical stability
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    
    # NLL = 0.5 * (log(2π) + logvar + (target - mu)^2 / exp(logvar))
    nll = 0.5 * (math.log(2 * math.pi) + logvar + (target - mu) ** 2 / logvar.exp())
    
    if mask is not None:
        nll = nll * mask
        if reduction == "mean":
            return nll.sum() / mask.sum().clamp(min=1)
        else:
            return nll.sum()
    else:
        if reduction == "mean":
            return nll.mean()
        else:
            return nll.sum()


class TSDiffCondNLL(Forecaster):
    """
    TSDiff with Dual-Head Prediction and Gaussian NLL Loss.
    
    Key differences from original TSDiff:
    1. Backbone outputs 2 * target_dim (mu and logvar)
    2. Uses Gaussian NLL loss instead of MSE
    3. Supports heteroscedastic noise modeling
    """
    
    def __init__(
        self,
        hidden_dim: int,
        step_emb: int,
        timesteps: int,
        num_residual_blocks: int,
        dropout: float = 0,
        noise_observed=False,
        mode="diag",
        measure="diag",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Backbone outputs 2 * target_dim for dual-head (mu + logvar)
        backbone_parameters = {
            "input_dim": self.target_dim,
            "hidden_dim": hidden_dim,
            "output_dim": self.target_dim * 2,  # Dual-head output
            "step_emb": step_emb,
            "num_residual_blocks": num_residual_blocks,
            "residual_block": "s4",
            "mode": mode,
            'measure': measure,
        }
        
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.sqrt_one_minus_beta = torch.sqrt(1.0 - self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.backbone = BackboneModel(
            **backbone_parameters,
            num_features=self.target_dim,
            # init_skip must be False for dual-head output (output_dim != input_dim)
            init_skip=False,
            dropout=dropout,
        )
        self.noise_observed = noise_observed

    def _extract_features(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        x = inputs[:, :, :self.target_dim]
        features = inputs.clone()
        
        if self.use_time_feat:
            features[:, self.context_length:, :self.target_dim] = 0
        else:
            features = features[:, :, :self.target_dim]
            features[:, self.context_length:] = 0
        
        observation_mask = torch.zeros_like(x, device=x.device)
        observation_mask[:, :self.context_length] = 1
        
        return x, features, observation_mask

    def q_sample(self, x_start, t, noise=None):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses_nll(
        self,
        x_start,
        t,
        features=None,
        noise=None,
        reduction="none",
    ):
        """
        Compute Gaussian NLL loss with dual-head prediction.
        
        The backbone predicts both mu (noise mean) and logvar (noise variance).
        """
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Backbone outputs [B, L, 2*C] for dual-head
        backbone_out = self.backbone(x_noisy, t, features)
        predicted_mu, predicted_logvar = torch.chunk(backbone_out, 2, dim=-1)
        
        # Gaussian NLL loss
        nll = gaussian_nll(noise, predicted_mu, predicted_logvar, reduction=reduction)
        
        return nll, x_noisy, predicted_mu, predicted_logvar

    @torch.no_grad()
    def p_sample(self, x, t, t_index, features=None):
        """
        Reverse diffusion step with dual-head prediction.
        """
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Dual-head output
        backbone_out = self.backbone(x, t, features)
        predicted_mu, predicted_logvar = torch.chunk(backbone_out, 2, dim=-1)
        
        # Use predicted_mu as the noise estimate
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_mu / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            # Use learned variance for sampling
            # Option 1: Use posterior variance (original)
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            
            # Option 2: Use predicted logvar (heteroscedastic)
            predicted_logvar = torch.clamp(predicted_logvar, min=-10.0, max=10.0)
            learned_variance = torch.exp(predicted_logvar)
            
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(learned_variance) * noise

    def step(self, x, t, features, loss_mask):
        noise = torch.randn_like(x)
        if not self.noise_observed:
            noise = (1 - loss_mask) * x + noise * loss_mask

        num_eval = loss_mask.sum()
        nll_loss, _, _, _ = self.p_losses_nll(
            x,
            t,
            features,
            reduction="none",
            noise=noise,
        )

        if self.noise_observed:
            elbo_loss = nll_loss
        else:
            # Apply loss mask
            if isinstance(nll_loss, torch.Tensor) and nll_loss.dim() > 0:
                nll_loss = nll_loss * loss_mask
                elbo_loss = nll_loss.sum() / (num_eval if num_eval else 1)
            else:
                elbo_loss = nll_loss
                
        return elbo_loss

    def loss(self, batch_data):
        x, features, observation_mask = self._extract_features(batch_data)
        loss_mask = 1 - observation_mask

        t = torch.randint(
            0, self.timesteps, [x.shape[0]], device=x.device
        ).long()
        
        loss = self.step(x, t, features, loss_mask)

        if torch.isnan(loss):
            print("Loss is NaN, exiting.")
            sys.exit(1)
        return loss

    def forecast(self, batch_data, num_samples):
        observation, features, observation_mask = self._extract_features(batch_data)
        observation = observation.to(observation.device)

        pred = self.sample(
            observation=observation,
            observation_mask=observation_mask,
            n_samples=num_samples,
            features=features,
        )

        return pred[:, :, -self.prediction_length:, :]

    @torch.no_grad()
    def sample(self, observation, observation_mask, n_samples, features=None):
        repeated_observation = repeat(observation, n_samples)
        repeated_observation_mask = repeat(observation_mask, n_samples)
        repeated_features = repeat(features, n_samples)
        
        batch_size, length, ch = repeated_observation.shape
        seq = torch.randn_like(repeated_observation)

        for i in reversed(range(0, self.timesteps)):
            if not self.noise_observed:
                seq = repeated_observation_mask * repeated_observation + seq * (1 - repeated_observation_mask)

            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=repeated_observation.device, dtype=torch.long),
                i,
                repeated_features,
            )

        seq = seq.reshape(-1, n_samples, length, ch)
        return seq
