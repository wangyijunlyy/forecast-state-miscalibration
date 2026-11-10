"""
* @description: FrequencyMixGaussianVAE wrapper for ProbTS Forecaster
"""

import torch

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.frequency_mix_gaussian_vae import FrequencyMixGaussianVAE


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class FrequencyMixGaussianVAEModel(Forecaster):
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
        likelihood="gaussian",
        student_t_df_mode="global",
        student_t_min_df=2.0,
        student_t_init_df=30.0,
        mix_period_len=2,
        mix_lpf=2,
        mix_alpha=0.5,
        mix_use_local_conv=True,
        sample_schedule=5,
        init_kalman="identity",
        init_koopman="both",
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        config.likelihood = likelihood
        config.student_t_df_mode = student_t_df_mode
        config.student_t_min_df = student_t_min_df
        config.student_t_init_df = student_t_init_df
        config.mix_period_len = mix_period_len
        config.mix_lpf = mix_lpf
        config.mix_alpha = mix_alpha
        config.mix_use_local_conv = mix_use_local_conv

        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.sample_schedule = sample_schedule
        config.init_kalman = init_kalman
        config.init_koopman = init_koopman

        self.weight_beta = weight_beta
        self.likelihood = likelihood.lower()
        if self.likelihood in {"student", "student-t", "student_t", "t"}:
            self.likelihood = "student_t"

        self.model = FrequencyMixGaussianVAE(config)

    def forward(self, input: torch.Tensor):
        return self.model(input)

    def loss(self, batch_data, **kwargs):
        self.model.train()

        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        target = batch_data.future_target_cdf

        outputs = self.forward(x)
        if self.likelihood == "student_t":
            x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar, x_df, y_df = outputs
            rec_loss = self.model.student_t_nll(x, x_mu, x_logvar, x_df, reduction="mean")
            pred_loss = self.model.student_t_nll(target, y_mu, y_logvar, y_df, reduction="mean")
            df_report = y_df.detach().mean()
        else:
            x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar = outputs
            rec_loss = self.model.gaussian_nll(x, x_mu, x_logvar, reduction="mean")
            pred_loss = self.model.gaussian_nll(target, y_mu, y_logvar, reduction="mean")
            df_report = None

        kld_loss = self.model.kl_divergence(mu_z, logvar_z)

        if df_report is None:
            print(
                f"rec_nll: {rec_loss:.4f}, pred_nll: {pred_loss:.4f}, "
                f"kld_loss: {kld_loss:.4f}"
            )
        else:
            print(
                f"rec_nll: {rec_loss:.4f}, pred_nll: {pred_loss:.4f}, "
                f"kld_loss: {kld_loss:.4f}, student_t_df: {df_report:.4f}"
            )

        return rec_loss + pred_loss + self.weight_beta * kld_loss

    def forecast(self, batch_data, num_samples=None):
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]

        with torch.no_grad():
            if num_samples is not None:
                outputs = self.model.sample(x, num_samples)
            else:
                outputs = self.forward(x)
                y_mu = outputs[1]
                outputs = y_mu.unsqueeze(1)

        return outputs
