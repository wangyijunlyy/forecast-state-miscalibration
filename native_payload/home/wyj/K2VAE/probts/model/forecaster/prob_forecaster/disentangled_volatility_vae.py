import torch

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.disentangled_volatility_vae import (
    DisentangledVolatilityVAE,
)


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class DisentangledVolatilityVAEModel(Forecaster):
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
        disentangle_weight=0.01,
        mi_weight=0.01,
        calibration_weight=0.05,
        sharpness_weight=0.01,
        calibration_coverage=0.9,
        calibration_temperature=0.1,
        trend_kernel=25,
        regime_dim=32,
        vol_hidden_dim=None,
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
        config.trend_kernel = trend_kernel
        config.regime_dim = regime_dim
        config.vol_hidden_dim = dynamic_dim if vol_hidden_dim is None else vol_hidden_dim
        config.calibration_temperature = calibration_temperature

        config.n_vars = self.input_size
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.sample_schedule = sample_schedule
        config.init_kalman = init_kalman
        config.init_koopman = init_koopman

        self.weight_beta = weight_beta
        self.disentangle_weight = disentangle_weight
        self.mi_weight = mi_weight
        self.calibration_weight = calibration_weight
        self.sharpness_weight = sharpness_weight
        self.calibration_coverage = calibration_coverage
        self.model = DisentangledVolatilityVAE(config)

    def forward(self, input: torch.Tensor):
        return self.model(input)

    def loss(self, batch_data, **kwargs):
        self.model.train()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        target = batch_data.future_target_cdf

        (
            x_mu,
            y_mu,
            trend_mu,
            trend_logvar,
            vol_mu,
            vol_logvar,
            x_logvar,
            y_logvar,
            z_trend,
            z_vol,
            _,
        ) = self.forward(x)

        rec_nll = self.model.gaussian_nll(x, x_mu, x_logvar)
        pred_nll = self.model.gaussian_nll(target, y_mu, y_logvar)
        trend_kl = self.model.kl_divergence(trend_mu, trend_logvar)
        vol_kl = self.model.kl_divergence(vol_mu, vol_logvar)
        disentangle = self.model.disentanglement_loss(z_trend, z_vol)
        mi_proxy = self.model.mutual_information_proxy(z_trend, z_vol)
        cal_loss, cal_gap, sharpness = self.model.calibration_sharpness_loss(
            target,
            y_mu,
            y_logvar,
            coverage=self.calibration_coverage,
            sharpness_weight=self.sharpness_weight,
        )

        loss = (
            rec_nll
            + pred_nll
            + self.weight_beta * (trend_kl + vol_kl)
            + self.disentangle_weight * disentangle
            + self.mi_weight * mi_proxy
            + self.calibration_weight * cal_loss
        )

        print(
            f"rec_nll: {rec_nll:.4f}, pred_nll: {pred_nll:.4f}, "
            f"trend_kl: {trend_kl:.4f}, vol_kl: {vol_kl:.4f}, "
            f"orth: {disentangle:.4f}, mi: {mi_proxy:.4f}, "
            f"cal: {cal_gap:.4f}, sharp: {sharpness:.4f}"
        )
        return loss

    def forecast(self, batch_data, num_samples=None):
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        with torch.no_grad():
            if num_samples is not None:
                outputs = self.model.sample(x, num_samples)
            else:
                outputs = self.forward(x)[1].unsqueeze(1)
        return outputs
