"""
* @description: K2VAE-NLL (改进版) - Forecaster 封装
*
* 改进点:
* 1. 使用显式的 Gaussian NLL 替代分布的 log_prob
* 2. 重建损失也使用 NLL 而非 MSE
* 3. 支持异方差建模和不确定性估计
*
* 对比原版 K2VAE:
* | 特性 | 原版 K2VAE | K2VAE-NLL |
* |------|------------|-----------|
* | 重建损失 | MSE | Gaussian NLL |
* | 预测损失 | -log_prob | Gaussian NLL |
* | 方差输出 | sigma (softplus) | logvar (clamp) |
* | 不确定性 | 隐式 | 显式可查询 |
"""

import math
import torch
from einops import rearrange

from probts import Forecaster
from probts.model.nn.prob.k2VAE.k2vae_nll import k2VAE_NLL
import torch.nn as nn
import torch.nn.functional as F


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class k2VAE_NLLModel(Forecaster):
    """
    改进版 K2VAE Forecaster with Gaussian NLL Loss
    
    训练流程:
    1. Encode: 历史序列 -> 潜在空间 (Koopman + Kalman)
    2. Decode: 潜在空间 -> (mu, logvar) 双头输出
    3. Loss: NLL(重建) + NLL(预测) + β * KLD(潜在)
    """
    
    def __init__(
        self,
        d_model,
        d_ff,
        e_layers,
        dropout,
        activation,
        n_heads,
        factor,
        patch_len,
        multistep,
        dynamic_dim,
        hidden_layers,
        hidden_dim,
        weight_beta=0.001,
        weight_rec=1.0,
        weight_pred=1.0,
        sample_schedule=5,
        init_kalman='identity',
        init_koopman='both',
        **kwargs
    ):
        """
        Args:
            d_model: Transformer d_model
            d_ff: Transformer feed-forward dim
            e_layers: Transformer encoder layers
            dropout: Dropout rate
            activation: Activation function
            n_heads: Number of attention heads
            factor: Attention factor
            patch_len: Patch length
            multistep: Whether to use multistep Koopman
            dynamic_dim: Dynamic (latent) dimension
            hidden_layers: MLP hidden layers
            hidden_dim: MLP hidden dim
            weight_beta: KLD loss weight
            weight_rec: Reconstruction NLL weight
            weight_pred: Prediction NLL weight
            sample_schedule: Number of sampling schedules
            init_kalman: Kalman init method
            init_koopman: Koopman init method
        """
        super().__init__(**kwargs)
        
        # Build config
        config = ConvertedParams(kwargs)
        config.d_model = d_model
        config.d_ff = d_ff
        config.hidden_layers = hidden_layers
        config.dropout = dropout
        config.activation = activation
        config.e_layers = e_layers
        config.n_heads = n_heads
        config.factor = factor
        config.patch_len = patch_len
        config.multistep = multistep
        config.dynamic_dim = dynamic_dim
        config.hidden_dim = hidden_dim
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.sample_schedule = sample_schedule
        config.init_kalman = init_kalman
        config.init_koopman = init_koopman
        
        self.weight_beta = weight_beta
        self.weight_rec = weight_rec
        self.weight_pred = weight_pred

        # Instantiate model
        self.model = k2VAE_NLL(config)

    def forward(self, input):
        """
        前向传播
        
        Args:
            input: [B, L, C]
        Returns:
            x_rec_mu, x_rec_logvar, x_pred_mu, x_pred_logvar, prior_dist
        """
        return self.model(input)

    def kld_loss(self, dist):
        """
        计算 KL 散度: KL(q(z|x) || p(z))
        
        Args:
            dist: MultivariateNormal 分布
        Returns:
            kld: scalar
        """
        mu_q = dist.loc  # Shape: (B, P, hidden)
        cov_q = dist.covariance_matrix  # Shape: (B, P, hidden, hidden)
        
        # Log determinant
        log_det_q = torch.linalg.slogdet(cov_q)[1]  # Shape: (B, P)
        
        # Trace
        trace_q = torch.einsum('btii->bt', cov_q)  # Shape: (B, P)
        
        # Squared norm of mean
        mu_term = torch.sum(mu_q ** 2, dim=-1)  # Shape: (B, P)
        
        # Latent dimension
        latent_dim = mu_q.size(-1)
        
        # KL divergence
        kld = 0.5 * (trace_q + mu_term - latent_dim - log_det_q)  # Shape: (B, P)
        kld = kld.mean(dim=-1)  # Shape: (B,)
        kld = kld.mean()  # Scalar
        
        return kld

    def loss(self, batch_data, **kwargs):
        """
        计算损失
        
        === 改进点: 使用 Gaussian NLL 替代 MSE 和 log_prob ===
        
        Loss = weight_rec * NLL(重建) + weight_pred * NLL(预测) + weight_beta * KLD
        """
        self.model.train()
        
        # Extract data
        input = batch_data.past_target_cdf[:, -self.context_length:, :]
        target = batch_data.future_target_cdf
        
        # Forward pass
        x_rec_mu, x_rec_logvar, x_pred_mu, x_pred_logvar, prior_dist = self.forward(input)
        
        # === Reconstruction NLL (使用 NLL 而非 MSE) ===
        rec_nll = self.model.gaussian_nll(input, x_rec_mu, x_rec_logvar)
        
        # === Prediction NLL ===
        pred_nll = self.model.gaussian_nll(target, x_pred_mu, x_pred_logvar)
        
        # === KLD Loss ===
        kld_loss = self.kld_loss(prior_dist)
        
        # Total loss
        loss = self.weight_rec * rec_nll + self.weight_pred * pred_nll + self.weight_beta * kld_loss
        
        # Logging
        mean_pred_std = torch.exp(0.5 * x_pred_logvar).mean().item()
        print(f"rec_nll: {rec_nll:.4f}, pred_nll: {pred_nll:.4f}, "
              f"kld: {kld_loss:.4f}, mean_pred_std: {mean_pred_std:.4f}")
        
        return loss

    def sample_from_distribution(self, input, num_samples):
        """
        从分布中采样多个预测
        """
        samples = self.model.sample(input, num_samples)
        return rearrange(samples, 'n b l c -> b n l c')

    def forecast(self, batch_data, num_samples=None):
        """
        生成预测
        
        Args:
            batch_data: 包含 past_target_cdf 的数据
            num_samples: 采样数量
        Returns:
            outputs: [B, num_samples, pred_len, C]
        """
        self.model.eval()
        input = batch_data.past_target_cdf[:, -self.context_length:, :]
        
        with torch.no_grad():
            if num_samples is not None:
                outputs = self.sample_from_distribution(input, num_samples)
            else:
                # 返回均值预测
                _, _, x_pred_mu, _, _ = self.forward(input)
                outputs = x_pred_mu.unsqueeze(1)  # [B, 1, pred_len, C]
        
        return outputs

    def get_uncertainty(self, batch_data):
        """
        获取预测的不确定性估计
        
        === K2VAE-NLL 独有功能 ===
        
        Args:
            batch_data: 包含 past_target_cdf 的数据
        Returns:
            mu: [B, pred_len, C] - 预测均值
            std: [B, pred_len, C] - 预测标准差
        """
        self.model.eval()
        input = batch_data.past_target_cdf[:, -self.context_length:, :]
        
        with torch.no_grad():
            mu, std = self.model.get_uncertainty(input)
        
        return mu, std
