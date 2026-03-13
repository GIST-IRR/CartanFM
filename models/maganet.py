import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.maganetmodule import Encoder, Decoder

device = "cuda" if torch.cuda.is_available() else "cpu"


"""MAGANet model for manifold-aware latent translation and reconstruction."""


class MAGANet(nn.Module):
    def __init__(self, config):
        """Initialize MAGANet encoder/decoder and latent settings."""
        super(MAGANet, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.latent_dim = config.latent_dim

        self.pivot = None

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
        """Compute latent translation between paired inputs and reconstruction."""
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
        """Compute reconstruction, KL, and latent-reconstruction losses."""
        result = {"elbo": {}, "obj": {}, "id": {}}

        # reconstruction loss

        recon_loss = loss_fn(
            input=x2_recon, target=x2,
        ) / x2.size(0)
        result["obj"]["reconst"] = recon_loss
        # KL divergence
        kl_div = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)
        )
        result["obj"]["kld"] = kl_div

        # latent reconstruction loss

        random_idx = torch.randperm(z.size(0))

        z_perm = z[random_idx]
        x_perm = x2[random_idx]

        x_hat = self.decoder(z_perm, x_perm)
        z_tilde = self.encoder(x_perm)[0]
        z_hat = self.encoder(x_hat)[0]

        z_recon = z_hat - z_tilde

        latent_recon_loss = F.l1_loss(z_recon, z_perm, reduction="sum") / z_perm.size(0)
        result["obj"]["latent_recon_loss"] = latent_recon_loss

        return result
