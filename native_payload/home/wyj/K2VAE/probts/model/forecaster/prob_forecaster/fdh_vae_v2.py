import torch

from probts import Forecaster
from probts.model.nn.prob.k2VAE.fdh_vae_v2 import FDHVAEv2, FDHVAEv2Loss


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class FDHVAEv2Model(Forecaster):
    """ProbTS wrapper for FDH-VAE v2."""

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
        d_decoder=256,
        n_bands=3,
        n_encoder_layers=2,
        d_spectral_cond=128,
        beta_band=0.1,
        beta_gate=0.05,
        lambda_freq=0.01,
        sample_schedule=20,
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
        config.d_decoder = d_decoder
        config.n_bands = n_bands
        config.n_encoder_layers = n_encoder_layers
        config.d_spectral_cond = d_spectral_cond
        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.sample_schedule = sample_schedule

        self.model = FDHVAEv2(config)
        self.criterion = FDHVAEv2Loss(
            beta_band=beta_band,
            beta_gate=beta_gate,
            lambda_freq=lambda_freq,
        )

    def forward(self, input: torch.Tensor):
        return self.model(input)

    def loss(self, batch_data, **kwargs):
        self.model.train()
        x_past = batch_data.past_target_cdf[:, -self.context_length :, :]
        x_future = batch_data.future_target_cdf
        result = self.forward(x_past)
        loss, loss_dict = self.criterion(result, x_past, x_future)
        print(
            "rec_nll: {:.4f}, pred_nll: {:.4f}, kl_band: {:.4f}, kl_gate: {:.4f}, freq: {:.4f}".format(
                loss_dict["rec_nll"],
                loss_dict["pred_nll"],
                loss_dict["kl_band_total"],
                loss_dict["kl_gate"],
                loss_dict["freq_consistency"],
            )
        )
        return loss

    def forecast(self, batch_data, num_samples=None):
        self.model.eval()
        x_past = batch_data.past_target_cdf[:, -self.context_length :, :]

        with torch.no_grad():
            if num_samples is not None:
                samples, _, _ = self.model.forecast(x_past, n_samples=num_samples)
                outputs = samples.permute(1, 0, 2, 3)
            else:
                result = self.forward(x_past)
                outputs = result["mu_pred"].unsqueeze(1)
        return outputs

    def get_uncertainty(self, batch_data):
        self.model.eval()
        x_past = batch_data.past_target_cdf[:, -self.context_length :, :]
        with torch.no_grad():
            result = self.forward(x_past)
        return result["mu_pred"], result["sigma_pred"]
