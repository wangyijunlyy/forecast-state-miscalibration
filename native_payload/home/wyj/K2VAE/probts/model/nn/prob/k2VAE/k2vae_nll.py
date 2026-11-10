"""
* @author: EmpyreanMoon (Modified)
*
* @create: 2025-01-02 15:40
*
* @description: K2VAE-NLL - 改进版 K2VAE with Gaussian NLL Loss
*
* 改进点:
* 1. 双头输出更加明确: decoder_mu + decoder_logvar (而非 sigma)
* 2. 损失函数使用显式的 Gaussian NLL 而非分布的 log_prob
* 3. 重建损失也使用 NLL 而非 MSE
* 4. 支持异方差建模，在不确定区域预测大方差
"""

import torch.nn as nn
import torch
import math
from torch.distributions import Normal, MultivariateNormal

from probts.model.nn.prob.k2VAE.koopman import KPLayer, KPLayerApprox, MLP
from probts.model.nn.prob.k2VAE.kalman import KalmanFilter
from probts.model.nn.prob.k2VAE.auxiliarynet import Transformer as AuxiliaryNet
from probts.model.nn.prob.k2VAE.RevIN import RevIN
from einops import rearrange
import torch.nn.functional as F


class k2VAE_NLL(nn.Module):
    """
    改进版 K2VAE with Gaussian NLL Loss
    
    核心改进:
    - 使用显式的 Gaussian NLL 替代分布的 log_prob
    - decoder 输出 logvar 而非 sigma，更稳定
    - 支持在不确定区域预测大方差
    """
    
    def __init__(self, config):
        super(k2VAE_NLL, self).__init__()
        self.config = config
        self.input_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.multistep = config.multistep
        self.dynamic_dim = config.dynamic_dim
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars

        self.freq = math.ceil(self.input_len / self.patch_len)  # patch number of input
        self.step = math.ceil(self.pred_len / self.patch_len)   # patch number of output
        self.padding_len = self.patch_len * self.freq - self.input_len
        
        # Koopman operator for dynamics
        self.koopman = KPLayerApprox(config.dynamic_dim, config.init_koopman) if self.multistep else KPLayer(
            config.dynamic_dim, config.init_koopman)

        # Kalman filter for uncertainty propagation
        self.kalman = KalmanFilter(state_dim=self.dynamic_dim, init=config.init_kalman)

        # Auxiliary network for residual modeling
        self.auxiliary = AuxiliaryNet(self.config)

        # Encoder
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in, 
            f_out=self.dynamic_dim, 
            activation='relu',
            hidden_dim=self.hidden_dim, 
            hidden_layers=self.hidden_layers
        )
        
        # === 改进点 1: 双头 Decoder (mu + logvar) ===
        # Location Head: 预测均值
        self.decoder_mu = MLP(
            f_in=self.dynamic_dim, 
            f_out=self.patch_len * self.enc_in, 
            activation='relu',
            hidden_dim=self.hidden_dim, 
            hidden_layers=self.hidden_layers
        )
        
        # Scale Head: 预测对数方差 (logvar 比 sigma 更稳定)
        self.decoder_logvar = MLP(
            f_in=self.dynamic_dim, 
            f_out=self.patch_len * self.enc_in, 
            activation='relu',
            hidden_dim=self.hidden_dim, 
            hidden_layers=self.hidden_layers
        )
        
        # RevIN for normalization
        self.revin = RevIN(self.enc_in)

    def encode(self, x):
        """
        编码阶段: 输入序列 -> 潜在空间
        
        Args:
            x: [B, L, C] - 输入序列
        Returns:
            x_rec: [B, F, H] - Koopman 重建的潜在状态
            pred: [B, S, H] - 预测的潜在状态
            dist: MultivariateNormal - 潜在空间的分布
        """
        B, L, C = x.shape
        x = self.revin(x, 'norm')

        # Patch embedding
        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)  # F x B P C
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)  # B F PC

        # Encode to latent space
        res = self.encoder(res)  # B F H

        # Koopman dynamics
        x_rec, Y = self.koopman(res, self.step)  # B F H, B S H
        x_residual = res - x_rec
        
        # Auxiliary modeling
        residual = self.auxiliary(x_residual)
        
        # Kalman filtering
        pred, dist = self.kalman(res[:, -1, :], residual, Y)
        pred = pred + residual

        return x_rec, pred, dist

    def decode(self, x_rec, x_pred):
        """
        解码阶段: 潜在状态 -> 输出分布参数 (mu, logvar)
        
        === 改进点 2: 输出 logvar 而非直接 sigma ===
        
        Args:
            x_rec: [B, F, H] - 重建的潜在状态 (用于历史重建)
            x_pred: [B, S, H] - 预测的潜在状态 (用于未来预测)
        Returns:
            x_rec_mu: [B, L, C] - 历史重建均值
            x_rec_logvar: [B, L, C] - 历史重建对数方差
            x_pred_mu: [B, pred_len, C] - 未来预测均值
            x_pred_logvar: [B, pred_len, C] - 未来预测对数方差
        """
        B, F, H = x_rec.shape
        
        # ===== 历史重建 =====
        x_rec_mu = self.decoder_mu(x_rec)  # B F PC
        x_rec_mu = x_rec_mu.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_rec_mu = x_rec_mu.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C

        x_rec_logvar = self.decoder_logvar(x_rec)  # B F PC
        x_rec_logvar = x_rec_logvar.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_rec_logvar = x_rec_logvar.reshape(B, -1, self.enc_in)[:, :self.input_len, :]  # B L C
        x_rec_logvar = torch.clamp(x_rec_logvar, min=-10.0, max=10.0)  # 数值稳定性

        # ===== 未来预测 =====
        x_pred_mu = self.decoder_mu(x_pred)  # B S PC
        x_pred_mu = x_pred_mu.reshape(B, self.step, self.patch_len, self.enc_in)
        x_pred_mu = x_pred_mu.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B pred_len C

        x_pred_logvar = self.decoder_logvar(x_pred)  # B S PC
        x_pred_logvar = x_pred_logvar.reshape(B, self.step, self.patch_len, self.enc_in)
        x_pred_logvar = x_pred_logvar.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]  # B pred_len C
        x_pred_logvar = torch.clamp(x_pred_logvar, min=-10.0, max=10.0)

        # RevIN 反归一化 (只对均值)
        x_rec_mu = self.revin(x_rec_mu, 'denorm')
        x_pred_mu = self.revin(x_pred_mu, 'denorm')

        return x_rec_mu, x_rec_logvar, x_pred_mu, x_pred_logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化采样
        
        Args:
            mu: [B, L, C]
            logvar: [B, L, C]
        Returns:
            sampled: [B, L, C]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, L, C] - 输入历史序列
        Returns:
            x_rec_mu: [B, L, C] - 历史重建均值
            x_rec_logvar: [B, L, C] - 历史重建对数方差
            x_pred_mu: [B, pred_len, C] - 未来预测均值
            x_pred_logvar: [B, pred_len, C] - 未来预测对数方差
            prior_dist: MultivariateNormal - 潜在空间先验分布
        """
        # Encode
        x_rec_latent, pred_latent, prior_dist = self.encode(x)
        
        # Sample from prior
        pred_latent = pred_latent + prior_dist.rsample()
        
        # Decode
        x_rec_mu, x_rec_logvar, x_pred_mu, x_pred_logvar = self.decode(x_rec_latent, pred_latent)

        return x_rec_mu, x_rec_logvar, x_pred_mu, x_pred_logvar, prior_dist

    @staticmethod
    def gaussian_nll(target, mu, logvar, reduction='mean'):
        """
        === 改进点 3: 显式 Gaussian NLL 损失 ===
        
        NLL = 0.5 * (log(2π) + logvar + (target - mu)^2 / exp(logvar))
        
        优势:
        - 在不确定区域，模型会预测大 logvar 来"包裹"异常值
        - 保持 mu 的鲁棒性
        
        Args:
            target: [B, L, C] - 真实值
            mu: [B, L, C] - 预测均值
            logvar: [B, L, C] - 预测对数方差
            reduction: 'mean', 'sum', or 'none'
        Returns:
            loss: scalar or [B]
        """
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        nll = 0.5 * (math.log(2 * math.pi) + logvar + (target - mu) ** 2 / torch.exp(logvar))
        
        # 对时间和变量维度求和
        nll = nll.sum(dim=[1, 2])  # [B]
        
        if reduction == 'mean':
            return nll.mean()
        elif reduction == 'sum':
            return nll.sum()
        else:
            return nll

    @torch.no_grad()
    def sample(self, x, num_samples):
        """
        生成多个预测样本
        
        Args:
            x: [B, L, C] - 输入历史
            num_samples: 采样数量
        Returns:
            samples: [num_samples, B, pred_len, C]
        """
        x_rec_latent, pred_latent, _ = self.encode(x)
        
        sample_list = []
        k = num_samples // self.config.sample_schedule
        cnt = 0
        
        for i in range(self.config.sample_schedule):
            # 在潜在空间加入随机性
            t = pred_latent + torch.randn_like(pred_latent).to(x.device)
            
            # 解码得到分布参数
            _, _, pred_mu, pred_logvar = self.decode(x_rec_latent, t)
            
            # 确定这一轮采样数量
            sample_num = k if i < self.config.sample_schedule - 1 else num_samples - cnt
            
            # 从预测分布采样
            for _ in range(sample_num):
                sample = self.reparameterize(pred_mu, pred_logvar)
                sample_list.append(sample)
            
            cnt += sample_num

        # [num_samples, B, pred_len, C]
        return torch.stack(sample_list, dim=0)

    def get_uncertainty(self, x):
        """
        获取预测的不确定性估计
        
        Args:
            x: [B, L, C]
        Returns:
            mu: [B, pred_len, C] - 预测均值
            std: [B, pred_len, C] - 预测标准差
        """
        with torch.no_grad():
            x_rec_latent, pred_latent, prior_dist = self.encode(x)
            pred_latent = pred_latent + prior_dist.rsample()
            _, _, pred_mu, pred_logvar = self.decode(x_rec_latent, pred_latent)
            pred_std = torch.exp(0.5 * pred_logvar)
        
        return pred_mu, pred_std
