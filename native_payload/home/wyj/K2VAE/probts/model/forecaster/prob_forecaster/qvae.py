"""
Quantum VAE Forecaster wrapper for ProbTS framework.
"""

import math
import torch
import torch.nn.functional as F

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.k2VAE.quantum_vae import QuantumVAE


class ConvertedParams:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class QVAEModel(Forecaster):
    """
    Quantum VAE Forecaster for probabilistic time series prediction.
    
    Uses variational quantum circuits for encoding/decoding with
    vMF latent space and Gaussian NLL loss.
    """
    
    def __init__(
        self,
        patch_len: int = 16,
        n_qbits: int = 6,
        n_blocks: int = 2,
        bottleneck_qbits: int = 2,
        kappa: float = 0.1,
        weight_beta: float = 0.001,
        **kwargs,
    ):
        """
        Args:
            patch_len: Length of each patch
            n_qbits: Number of qubits in quantum circuit
            n_blocks: Number of VQC blocks
            bottleneck_qbits: Qubits to trace out for latent dim
            kappa: vMF concentration parameter
            weight_beta: KL weight
        """
        super().__init__(**kwargs)
        
        # Config assembly
        config = ConvertedParams(kwargs)
        config.seq_len = self.context_length
        config.pred_len = self.prediction_length
        config.n_vars = self.input_size
        config.patch_len = patch_len
        config.n_qbits = n_qbits
        config.n_blocks = n_blocks
        config.bottleneck_qbits = bottleneck_qbits
        config.kappa = kappa
        
        self.weight_beta = weight_beta
        
        # Instantiate model
        self.model = QuantumVAE(config)
    
    def forward(self, input: torch.Tensor):
        """
        Forward pass.
        
        Args:
            input: [B, L, C]
            
        Returns:
            x_mu, y_mu, mu_z, x_logvar, y_logvar
        """
        return self.model(input)
    
    def loss(self, batch_data, **kwargs):
        """
        Compute loss = rec_nll + pred_nll + weight_beta * KL
        """
        self.model.train()
        
        # Get history
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        target = batch_data.future_target_cdf
        
        # Forward
        x_mu, y_mu, mu_z, x_logvar, y_logvar = self.forward(x)
        
        # Reconstruction NLL
        rec_loss = self.model.gaussian_nll(x, x_mu, x_logvar, reduction="mean")
        
        # Prediction NLL
        pred_loss = self.model.gaussian_nll(target, y_mu, y_logvar, reduction="mean")
        
        # KL loss
        kl_loss = self.model.kl_vmf_uniform(mu_z)
        
        loss = rec_loss + pred_loss + self.weight_beta * kl_loss
        
        return loss
    
    def forecast(self, batch_data, num_samples=None):
        """
        Generate probabilistic forecasts.
        
        Args:
            batch_data: ProbTSBatchData
            num_samples: Number of samples
            
        Returns:
            outputs: [B, num_samples, pred_len, C]
        """
        self.model.eval()
        x = batch_data.past_target_cdf[:, -self.context_length:, :]
        
        with torch.no_grad():
            if num_samples is not None:
                outputs = self.model.sample(x, num_samples)
            else:
                x_mu, y_mu, _, _, _ = self.forward(x)
                outputs = y_mu.unsqueeze(1)
        
        return outputs
