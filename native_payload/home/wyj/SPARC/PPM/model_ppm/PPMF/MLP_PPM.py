import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        
        self.K = configs.K

        # encoder
        self.fc1 = nn.Linear(self.seq_len, self.d_model)
        self.fc_mu = nn.Linear(self.d_model, self.d_model)     # mean
        self.fc_logvar = nn.Linear(self.d_model, self.d_model) # log variance

        # decoder
        self.x_emb = nn.Linear(self.seq_len, self.d_model)


        self.decode_net = nn.Sequential(nn.Linear(2*self.d_model, self.d_ff),
                                        nn.GELU(),
                                        nn.Linear(self.d_ff, self.pred_len),)
                                        # nn.GELU(),
                                        # nn.Linear(self.d_model, self.pred_len))

    def encode(self, x):
        h = F.gelu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, K):
        std = torch.exp(0.5 * logvar)   # standard deviation

        mu = mu.unsqueeze(-2).expand(-1, -1, K, -1)
        std = std.unsqueeze(-2).expand(-1, -1, K, -1)

        eps = torch.randn_like(std)     # sample epsilon ~ N(0, I)
        return mu + eps * std           # z = mu + std * eps

    def decode(self, z, x, K):
        cond = self.x_emb(x).unsqueeze(-2).expand(-1, -1, K, -1)
        z = torch.cat((z, cond), dim=-1)
        # h = F.gelu(self.fc2(z))
        # return self.fc3(h)  # output in [0,1]
        outputs = self.decode_net(z)

        return outputs

    def forward(self, x, K):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-6)

        x = x.permute(0,2,1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, K)
        recon_x = self.decode(z, x, K)

        recon_x = recon_x.permute(0,2,3,1)

        recon_x = recon_x * (std.unsqueeze(1) + 1e-6) + mean.unsqueeze(1)

        return recon_x
    