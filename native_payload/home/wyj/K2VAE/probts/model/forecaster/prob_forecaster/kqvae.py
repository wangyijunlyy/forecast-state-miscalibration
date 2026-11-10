"""
* @author: EmpyreanMoon
*
* @description: k2QVAE wrapper for ProbTS Forecaster
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from probts import Forecaster
from probts.model.nn.prob.k2VAE.kQVAE_1 import kQVAE  # 注意：这里路径按你放文件的位置修改


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class kQVAEModel(Forecaster):
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
        # QVAE 相关超参
        dist_beta=10.0,         # MixtureGeneric 平滑分布的 beta
        weight_beta=0.001,      # KL 权重（和你原先 k2VAEModel 一致）
        kl_beta=1.0,            # 传给底层模型的 KL 系数（如果在那边也要用）
        sample_schedule=5,
        init_kalman="identity",
        init_koopman="both",
        bm_visible_dim=None,
        bm_hidden_dim=None,
        **kwargs,
    ):
        """
        k2QVAE wrapper 构造函数
        """
        super().__init__(**kwargs)

        # ---- 组装 config 对象，传给 k2QVAE ----
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

        # ProbTS 侧的信息
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length

        # QVAE 相关
        config.dist_beta = dist_beta
        config.kl_beta = kl_beta
        config.sample_schedule = sample_schedule
        config.init_kalman = init_kalman
        config.init_koopman = init_koopman

        if bm_visible_dim is not None:
            config.bm_visible_dim = bm_visible_dim
        if bm_hidden_dim is not None:
            config.bm_hidden_dim = bm_hidden_dim

        # KL 权重（顶层）
        self.weight_beta = weight_beta

        # ---- 实例化底层 k2QVAE 模型 ----
        self.model = kQVAE(config)

    def forward(self, input: torch.Tensor):
        """
        Parameters:
            input: [B, L, C]
        Returns:
            x_rec:      [B, L, C]    (重构的历史部分)
            posterior:  MixtureGeneric 后验分布对象
            post_dist:  Normal 预测分布 (未来)
            zeta:       连续潜变量样本
        """
        x_rec,  post_dist, zeta ,_= self.model(input)
        return x_rec, post_dist, zeta,_

    def loss(self, batch_data, threshold: float = 1e2):
        """
        模仿 k2VAEModel 的 loss 结构：
            loss = rec_loss + weight_alpha * post_loss + weight_beta * KL

        其中：
          rec_loss = MSE(重构历史, 历史) + MSE(预测均值, 未来真值)
          post_loss = -log p(未来真值 | post_dist)
          KL = QVAE + BM 的 KL(q || p_BM)
        """
        self.model.train()

        # 和 k2VAEModel 保持一致的数据取法
        x = batch_data.past_target_cdf[:, -self.context_length :, :]   # [B, L, C]
        target = batch_data.future_target_cdf                          # [B, pred_len, C]

        # 前向传播
        rec, post_dist, zeta, post_dist_pred = self.forward(x)

        # ---- 重构项：历史重构 + 预测均值的 MSE ----
        rec_loss = F.mse_loss(rec, x) + F.mse_loss(post_dist_pred.loc, target)
        

        # ---- 似然项：- log p(y | z) ----
        post_loss = -post_dist_pred.log_prob(target).mean()

        # ---- KL 项：由 QVAE + BM 的 KL 计算 ----
        kld_loss = self.model.kl_divergence(post_dist, zeta)
        
        # ---- 自适应关闭 / 打开 post_loss， stabilize training ----
        weight_alpha = 1 if post_loss < threshold else 0.0

        loss = rec_loss + weight_alpha * post_loss + self.weight_beta * kld_loss
        print(f"rec_loss: {rec_loss:.4f}, post_loss: {post_loss:.4f}, kld_loss: {kld_loss:.4f}")
        return loss

    def sample_from_distribution(self, input: torch.Tensor, num_samples: int):
        """
        从 k2QVAE 的 sample 函数中采样:
            self.model.sample(x, num_samples) -> [num_samples, B, pred_len, C]
        然后 rearrange 成 [B, num_samples, pred_len, C]，和 ProbTS 兼容。
        """
        samples = self.model.sample(input, num_samples)
        return rearrange(samples, "n b l c -> b n l c")

    def forecast(self, batch_data, num_samples=None):
        """
        生成预测：
            - 如果 num_samples is None，就不返回预测（可以只用于评估 loss）
            - 否则返回 [B, num_samples, pred_len, C] 的样本
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]

        with torch.no_grad():
            if num_samples is not None:
                outputs = self.sample_from_distribution(x, num_samples)
            else:
                outputs = None

        return outputs  # [B, num_samples, pred_len, C]