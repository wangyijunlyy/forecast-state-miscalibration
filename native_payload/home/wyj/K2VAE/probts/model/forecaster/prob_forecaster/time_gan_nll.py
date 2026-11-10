"""
* @description: 改进版 TimeGAN (Gaussian NLL Loss) - Forecaster 封装
*
* 特点:
* - Generator 双头输出: mu (均值) 和 logvar (对数方差)
* - 使用重参数化技巧采样
* - 辅助损失使用 Gaussian NLL，而非 MSE
* - 支持不确定性量化
*
* 优势对比:
* | 特性 | MSE版本 | NLL版本 |
* |------|---------|---------|
* | 输出 | 点估计 y_hat | 分布参数 (μ, σ²) |
* | 抗噪能力 | 差 | 强 (通过增大σ包裹异常值) |
* | 生成多样性 | 低 | 高 (采样机制引入随机性) |
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.time_gan_nll import TimeGANNLL


class ConvertedParams:
    """将字典转换为属性对象"""

    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class TimeGANNLLModel(Forecaster):
    """
    改进版 TimeGAN Forecaster (Gaussian NLL Loss)

    训练流程:
    1. Generator 输出分布参数 (mu, logvar)
    2. 使用重参数化技巧采样 gen_future ~ N(mu, sigma²)
    3. Discriminator 判别采样结果 vs 真实数据
    4. Generator Loss = Adversarial Loss + lambda * Gaussian NLL
    5. Discriminator Loss = BCE Loss
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dynamic_dim: int = 32,
        hidden_layers: int = 2,
        patch_len: int = 4,
        lambda_nll: float = 10.0,
        **kwargs,
    ):
        """
        Args:
            hidden_dim: MLP 隐层维度
            dynamic_dim: LSTM 隐层维度
            hidden_layers: MLP 层数
            patch_len: Patch 长度
            lambda_nll: NLL 损失权重 (相对于对抗损失)
        """
        super().__init__(**kwargs)

        # === 构建 Config ===
        config = ConvertedParams(kwargs)
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.hidden_dim = hidden_dim
        config.dynamic_dim = dynamic_dim
        config.hidden_layers = hidden_layers
        config.patch_len = patch_len

        self.lambda_nll = lambda_nll
        self.config = config

        # === 实例化模型 ===
        self.model = TimeGANNLL(config)

        # === 损失函数 ===
        self.adversarial_loss = nn.BCELoss()

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: [B, L, C]
        Returns:
            mu: [B, pred_len, C]
            logvar: [B, pred_len, C]
        """
        return self.model(x)

    def loss(self, batch_data, **kwargs) -> torch.Tensor:
        """
        计算损失 (交替训练 G 和 D)

        关键改进:
        - 使用 Gaussian NLL 替代 MSE
        - 判别器看到的是从分布采样的样本，而非确定性点
        """
        self.model.train()

        # === 提取数据 ===
        x = batch_data.past_target_cdf[:, -self.context_length :, :]  # [B, L, C]
        y_real = batch_data.future_target_cdf  # [B, pred_len, C]
        batch_size = x.size(0)
        device = x.device

        # === 标签 ===
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # ========================
        #  Generator 损失
        # ========================
        # 1. 生成分布参数
        mu, logvar = self.model(x)

        # 2. 重参数化采样: 判别器看到的是采样出来的具体路径
        gen_future = self.model.generator.reparameterize(mu, logvar)

        # 3. 对抗损失: 欺骗判别器
        g_adv_loss = self.adversarial_loss(self.model.discriminate(gen_future), valid)

        # 4. Gaussian NLL 损失: 替代 MSE
        # 这迫使 Generator 在"拿不准"的时候增大 logvar，而不是乱猜均值
        g_nll_loss = self.model.gaussian_nll(y_real, mu, logvar)

        # 5. Generator 总损失
        g_loss = g_adv_loss + self.lambda_nll * g_nll_loss

        # ========================
        #  Discriminator 损失
        # ========================
        # 1. 判别真实数据
        d_real_loss = self.adversarial_loss(self.model.discriminate(y_real), valid)

        # 2. 判别生成数据 (detach, 不更新 G)
        d_fake_loss = self.adversarial_loss(
            self.model.discriminate(gen_future.detach()), fake
        )

        # 3. Discriminator 总损失
        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        # ========================
        #  总损失 (联合训练)
        # ========================
        total_loss = g_loss + d_loss

        # 打印训练信息
        print(
            f"G_adv: {g_adv_loss.item():.4f}, G_nll: {g_nll_loss.item():.4f}, "
            f"D_real: {d_real_loss.item():.4f}, D_fake: {d_fake_loss.item():.4f}"
        )

        return total_loss

    def forecast(self, batch_data, num_samples=None) -> torch.Tensor:
        """
        生成预测

        改进点: 支持从分布采样多条轨迹，提供不确定性估计

        Args:
            batch_data: 包含 past_target_cdf 的数据
            num_samples: 采样数量
        Returns:
            outputs: [B, num_samples, pred_len, C] 或 [B, 1, pred_len, C]
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]

        with torch.no_grad():
            if num_samples is not None:
                # 从分布采样多条轨迹 (每条轨迹都不同)
                outputs = self.model.sample(x, num_samples)
            else:
                # 返回均值预测
                mu = self.model.get_mean_prediction(x)
                outputs = mu.unsqueeze(1)  # [B, 1, pred_len, C]

        return outputs

    def get_uncertainty(self, batch_data) -> tuple:
        """
        获取预测的不确定性

        Args:
            batch_data: 包含 past_target_cdf 的数据
        Returns:
            mu: [B, pred_len, C] - 预测均值
            std: [B, pred_len, C] - 预测标准差
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]

        with torch.no_grad():
            mu, logvar = self.model(x)
            std = torch.exp(0.5 * logvar)

        return mu, std
