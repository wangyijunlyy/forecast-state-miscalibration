"""ProbTS wrapper for the variance-preserving original-backbone upgrade."""

from probts.model.forecaster.prob_forecaster.VolDy_VAE_Upgrade_Forecaster import VolDy_VAE_UpgradeModel
from probts.model.nn.prob.k2VAE.VolDy_VAE_Upgrade_V2 import VolDy_VAE_Upgrade_V2


class VolDy_VAE_Upgrade_V2Model(VolDy_VAE_UpgradeModel):
    def __init__(self, anchor_weight=0.8, state_dim=None, **kwargs):
        super().__init__(anchor_weight=anchor_weight, state_dim=state_dim, **kwargs)
        self.model = VolDy_VAE_Upgrade_V2(self.model.config)
