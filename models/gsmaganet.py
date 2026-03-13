import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.maganetmodule import Encoder, Decoder

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


"""Geodesic-symmetry enhanced MAGANet (GSMAGANet)."""

class GeodesicMaker(nn.Module):
    def __init__(self, config):
        """Predict geodesic curve parameters conditioned on latent endpoints."""
        super(GeodesicMaker, self).__init__()
        self.latent_dim = config.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.weight1_encoder = nn.Linear(128, 64)
        self.weight2_encoder = nn.Linear(128, 64 * 256)
        self.bias1_encoder = nn.Linear(128, 64)
        self.bias2_encoder = nn.Linear(128, 256)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, anchor, z):
        """Return batched curve parameters `(w1, w2, b1, b2)`."""
        batch_size = z.size(0)
        output = self.encoder(z) + self.encoder(anchor)
        w1 = self.weight1_encoder(output).view(batch_size, 1, 64)
        w2 = self.weight2_encoder(output).view(batch_size, 64, 256)
        b1 = self.bias1_encoder(output).view(batch_size, 1, -1)
        b2 = self.bias2_encoder(output).view(batch_size, 1, -1)
        return w1, w2, b1, b2
    

class GSMAGANet(nn.Module):
    def __init__(self, config):
        """Initialize GSMAGANet modules and trainable latent anchor."""
        super(GSMAGANet, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.geodesic_maker = GeodesicMaker(config)
        self.latent_dim = config.latent_dim
        self.step_size = config.step_size

        self.anchor_latent = nn.Parameter(torch.zeros(config.latent_dim), requires_grad=True)

        self.pivot = None

        self.flag_to_anchoring = False

    def init_weights(self):
        """Apply Xavier initialization to convolutional and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def set_pivot(self, pivot):
        """Set fixed pivot sample used during evaluation-time anchoring."""
        self.pivot = pivot.to(device)
        self.pivot = self.pivot.to(device)
        self.decoder.set_pivot(pivot)

    def forward(self, x, loss_fn):
        """Run paired encoding/decoding and return loss dictionary + reconstructions."""
        batch_size = x.size(0)
        assert batch_size % 2 == 0

        assert x.min() >= 0.0 and x.max() <= 1.0

        if self.pivot is None or self.training:
            perm_idx = torch.randperm(x.size(0))
            x1 = x[perm_idx]
            x2 = x
        elif self.pivot is not None and not self.training:
            x1 = self.pivot.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
            x2 = x
        else:
            raise NotImplementedError

        z1, mu1, logvar1 = self.encoder(x1)
        z2, mu2, logvar2 = self.encoder(x2)

        z = z2 - z1
        x2_recon = self.decoder(z, x1)

        loss = self.loss(x2, x2_recon, mu2, logvar2, z2, loss_fn)
        
        return (loss, x2_recon)
    

    def loss(
        self,
        x2,
        x2_recon,
        mu,
        logvar,
        z,
        loss_fn,
    ):
        """Compute reconstruction, KL, latent, geodesic, and symmetry losses."""
        
        result = {"elbo": {}, "obj": {}, "id": {}}

        # Reconstruction loss

        recon_loss = loss_fn(
            input=x2_recon, target=x2,
        ) / x2.size(0)

        # KL loss
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)
        )

        # Latent reconstruction loss
        random_idx = torch.randperm(z.size(0))

        z_perm = z[random_idx]
        x_perm = x2[random_idx]

        x_hat = self.decoder(z_perm, x_perm)
        z_tilde = self.encoder(x_perm)[0]
        z_hat = self.encoder(x_hat)[0]

        z_recon = z_hat - z_tilde

        latent_recon_loss = F.l1_loss(z_recon, z_perm, reduction="sum") / z_perm.size(0)

        # Geodesic loss and GS loss
        batch_size = z.size(0)
        w1, w2, b1, b2 = self.geodesic_maker(self.anchor_latent, z)
        symm_point, mu_symm, logvar_symm = self.geodesic(w1, w2, b1, b2, -torch.ones((batch_size, 1, 1)).to(device))
        extra_point = self.anchor_latent.unsqueeze(0).repeat(batch_size, 1) * 2 - z
        gs_loss = nn.functional.l1_loss(symm_point, extra_point, reduction="sum")

        geodesic_loss = self.geodesic_loss(z, w1, w2, b1, b2)

        result["obj"]["reconst"] = recon_loss.unsqueeze(0)
        result["obj"]["kld"] = kld_loss.unsqueeze(0)
        result["obj"]["latent_recon_loss"] = latent_recon_loss.unsqueeze(0)
        result["obj"]["gs"] = gs_loss.unsqueeze(0)
        result["obj"]["geodesic"] = geodesic_loss.unsqueeze(0)

        return result


    def geodesic(self, w1, w2, b1, b2, t):
        """Evaluate latent geodesic parameterized by `t` and return sampled point."""
        batch_size = t.size(0)
        t = t.view(batch_size, 1, 1)
        w1 = w1.view(batch_size, 1, 64)
        w2 = w2.view(batch_size, 64, 256)
        b1 = b1.view(batch_size, 1, 64)
        b2 = b2.view(batch_size, 1, 256)
        output = torch.bmm(t, w1) + b1
        output = torch.relu(output)
        output = torch.bmm(output, w2) + b2
        mu = self.encoder.w_to_mu(output.squeeze())
        logvar = self.encoder.w_to_logvar(output.squeeze())
        output = self.encoder.reparameterize(mu, logvar)
        return output, mu, logvar
    
    def geodesic_loss(self, z, w1, w2, b1, b2):
        """Approximate geodesic energy and endpoint constraints."""
        self.encoder = self.encoder.requires_grad_(False)

        zero = torch.zeros((z.size(0), 1, 1)).to(device)
        one = torch.ones((z.size(0), 1, 1)).to(device)

        gamma_zero, mu_zero, logvar_zero = self.geodesic(w1, w2, b1, b2, zero)
        gamma_one, mu_one, logvar_one = self.geodesic(w1, w2, b1, b2, one)

        energy = 0

        gamma_old, mu_old, logvar_old = gamma_zero, mu_zero, logvar_zero

        for i in range(1, 11, 1):
            p = one * (i / 10)
            gamma_p, mu_p, logvar_p = self.geodesic(w1, w2, b1, b2, p)
            energy += torch.pow(mu_old - mu_p, 2) + (logvar_old.exp() + logvar_p.exp())
            gamma_old, mu_old, logvar_old = gamma_p, mu_p, logvar_p
                    
        geodesic_loss = (
            nn.functional.l1_loss(gamma_zero, self.anchor_latent.unsqueeze(0).repeat(z.size(0), 1), reduction="sum")
            + nn.functional.l1_loss(gamma_one, z, reduction="sum")
            + energy.sum()
        )

        geodesic_loss /= z.size(0)

        self.encoder = self.encoder.requires_grad_(True)

        return geodesic_loss