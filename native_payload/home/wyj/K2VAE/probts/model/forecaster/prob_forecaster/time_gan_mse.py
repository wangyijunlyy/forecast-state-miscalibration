"""
* @description: 概率版 TimeGAN (NLL Loss) - Forecaster 封装
*
* 特点:
* - Generator 输出 mu 和 sigma (双头)
* - Forward 返回 Normal 分布对象 (类似 PatchTST)
* - 损失使用 -log_prob (NLL)
* - 与 ProbTS 框架集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.time_gan_mse import TimeGANMSE


class ConvertedParams:
    """将字典转换为属性对象"""

    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class TimeGANMSEModel(Forecaster):
    """
    概率版 TimeGAN Forecaster (NLL Loss)
    
    类似 PatchTST 的设计:
    - Forward 返回 Normal 分布对象
    - Loss 使用 -log_prob (NLL)
    - Forecast 使用 rsample 从分布采样
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dynamic_dim: int = 32,
        hidden_layers: int = 2,
        patch_len: int = 4,
        lambda_mse: float = 10.0,
        **kwargs,
    ):
        """
        Args:
            hidden_dim: MLP 隐层维度
            dynamic_dim: LSTM 隐层维度
            hidden_layers: MLP 层数
            patch_len: Patch 长度
            lambda_mse: MSE 损失权重 (相对于对抗损失)
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

        self.config = config

        # === 实例化模型 ===
        self.model = TimeGANMSE(config)

        # === 损失函数 (类似 PatchTST) ===
        self.loss_fn = lambda dist, targets: -dist.log_prob(targets)

    def forward(self, x: torch.Tensor):
        """
        前向传播 (返回分布对象，类似 PatchTST)

        Args:
            x: [B, L, C]
        Returns:
            dist: Normal 分布对象
        """
        return self.model(x)

    def loss(self, batch_data, **kwargs) -> torch.Tensor:
        """
        计算损失 (类似 PatchTST 使用 NLL)
        """
        inputs = self.get_inputs(batch_data, 'encode')
        dist = self(inputs)  # Normal 分布
        
        targets = batch_data.future_target_cdf
        loss = self.loss_fn(dist, targets)
        
        return loss.mean()

    def forecast(self, batch_data, num_samples=None) -> torch.Tensor:
        """
        生成概率预测 (类似 PatchTST)

        Args:
            batch_data: 包含 past_target_cdf 的数据
            num_samples: 采样数量
        Returns:
            outputs: [B, num_samples, pred_len, C]
        """
        inputs = self.get_inputs(batch_data, 'encode')
        dist = self(inputs)  # Normal 分布
        outputs = dist.rsample(sample_shape=(num_samples,))  # [num_samples, B, pred_len, C]
        outputs = outputs.permute(1, 0, 2, 3)  # [B, num_samples, pred_len, C]
        return outputs
