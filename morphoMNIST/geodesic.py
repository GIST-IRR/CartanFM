import torch
import torch.nn as nn


"""Geodesic-curve parameterization utilities for MorphoMNIST experiments."""

'''class GeodesicMaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.w1 = nn.Linear(32, 32)
        self.mu = nn.Linear(32, 32 * 2)
        self.logvar = nn.Linear(32, 32 * 2)
        self.b1 = nn.Linear(32, 32)
        self.bmu = nn.Linear(32, 2)
        self.blogvar = nn.Linear(32, 2)

        self.anchor = nn.Parameter(torch.zeros(2)).requires_grad_()

    def forward(self, z):
        a = self.anchor.clone().unsqueeze(0).repeat(z.size(0), 1)
        conc = torch.concat((z, a), dim=1)
        conc = self.encoder(conc)
        w1 = self.w1(conc)
        wmu = self.mu(conc)
        wlogvar = self.logvar(conc)        
        b1 = self.b1(conc)
        bmu = self.bmu(conc)
        blogvar = self.blogvar(conc)

        return w1, wmu, wlogvar, b1, bmu, blogvar

def geodesic(t, w1, wmu, wlogvar, b1, bmu, blogvar):
    batch_size = t.size(0)
    t = t.view(batch_size, 1, 1)
    w1 = w1.view(batch_size, 1, 32)
    wmu = wmu.view(batch_size, 32, 2)
    wlogvar = wlogvar.view(batch_size, 32, 2)
    b1 = b1.view(batch_size, 1, 32)
    bmu = bmu.view(batch_size, 1, 2)
    blogvar = blogvar.view(batch_size, 1, 2)
    output = torch.bmm(t, w1) + b1
    output = torch.relu(output)
    mu_output = torch.bmm(output, wmu) + bmu
    logvar_output = torch.bmm(output, wlogvar) + blogvar
    output = mu_output + torch.randn_like(logvar_output).cuda() * (logvar_output * 0.5).exp()
    return output.squeeze(), mu_output.squeeze(), logvar_output.squeeze()
'''

class GeodesicMaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.w1 = nn.Linear(32, 32)
        self.w2 = nn.Linear(32, 32*32)
        self.b1 = nn.Linear(32, 32)
        self.b2 = nn.Linear(32, 32)

        self.anchor = nn.Parameter(torch.zeros(2)).requires_grad_()

    def forward(self, z):
        a = self.anchor.clone().unsqueeze(0).repeat(z.size(0), 1)
        conc = torch.concat((z, a), dim=1)
        conc = self.encoder(conc)
        w1 = self.w1(conc)
        w2 = self.w2(conc)
        b1 = self.b1(conc)
        b2 = self.b2(conc)

        return w1, w2, b1, b2

def geodesic(t, w1, w2, b1, b2, vae_encoder):
    """Evaluate geodesic point at parameter `t` and decode latent moments."""
    batch_size = t.size(0)
    t = t.view(batch_size, 1, 1)
    w1 = w1.view(batch_size, 1, 32)
    b1 = b1.view(batch_size, 1, 32)
    w2 = w2.view(batch_size, 32, 32)
    b2 = b2.view(batch_size, 1, 32)
    output = torch.bmm(t, w1) + b1
    output = torch.relu(output)
    output = torch.bmm(output, w2) + b2
    output = torch.relu(output.squeeze())
    mu_output, logvar_output = vae_encoder.w_to_z(output)
    output = mu_output + torch.randn_like(logvar_output).cuda() * (logvar_output * 0.5).exp()
    
    return output.squeeze(), mu_output.squeeze(), logvar_output.squeeze()