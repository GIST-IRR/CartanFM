import torch
import torch.nn as nn

from vae import VAE
from geodesic import GeodesicMaker, geodesic


"""Geodesic-symmetry VAE baseline for MorphoMNIST."""


class GSVAE(nn.Module):
    """VAE augmented with geodesic and symmetry regularization terms."""

    def __init__(self):
        """Initialize vanilla VAE and geodesic parameter network."""
        super().__init__()
        self.vae = VAE()
        self.gamma = GeodesicMaker()

    def forward(self, x):
        """Return loss terms and VAE outputs for one batch."""
        output, z, mean, logvar = self.vae(x)
        loss = self.loss(x, output, z, mean, logvar)
        return loss, output, z, mean, logvar

    def loss(self, x, output, z, mean, logvar):
        """Compute reconstruction, KL, geodesic, and symmetry objectives."""
        bce = nn.functional.binary_cross_entropy(output, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        batch_size = z.size(0)
        w1, w2, b1, b2 = self.gamma(z)
        
        zero = torch.zeros((z.size(0), 1, 1)).cuda()
        one = torch.ones((z.size(0), 1, 1)).cuda()
        
        energy = 0
        
        gamma_zero, mu_zero, logvar_zero = geodesic(zero, w1, w2, b1, b2, self.vae.encoder)
        gamma_one, mu_one, logvar_one = geodesic(one, w1, w2, b1, b2, self.vae.encoder)
        
        gamma_old = gamma_zero
        mu_old = mu_zero
        logvar_old = logvar_zero
        
        for i in range(1, 33):
            p = one * (i / 32)
            gamma_p, mu_p, logvar_p = geodesic(p, w1, w2, b1, b2, self.vae.encoder)
            energy += torch.pow(mu_old - mu_p, 2) + (logvar_old.exp() + logvar_p.exp())
            gamma_old, mu_old, logvar_old = gamma_p, mu_p, logvar_p
                    
        geodesic_loss = (
            nn.functional.l1_loss(gamma_zero, self.gamma.anchor.unsqueeze(0).repeat(batch_size, 1), reduction="sum")
            + nn.functional.l1_loss(gamma_one, z, reduction="sum")
            + energy.sum()
        )

        inverse, _, _ = geodesic(-one, w1, w2, b1, b2, self.vae.encoder)
        gsloss = nn.functional.l1_loss(2 * self.gamma.anchor - z, inverse).sum()

        return (bce , kld , (geodesic_loss + gsloss))