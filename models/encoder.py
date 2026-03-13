import torch
import torch.nn as nn

import numpy as np

from models.cnn_layer import CNNLayer


"""CNN-based encoders for dSprites and 3D Shapes datasets."""


class CNN2DShapesEncoder(nn.Module):
    def __init__(self, config):
        """Construct encoder backbone for grayscale (dSprites) images."""
        super(CNN2DShapesEncoder, self).__init__()
        modules = []
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states

        # Design Encoder Factor-VAE ref
        modules.append(CNNLayer(in_channels=1, out_channels=32))
        modules.append(CNNLayer(in_channels=32, out_channels=32))
        modules.append(CNNLayer(in_channels=32, out_channels=64))
        modules.append(CNNLayer(in_channels=64, out_channels=64))
        self.hidden_layers = nn.ModuleList(modules)

        self.dense = nn.Linear(config.dense_dim[0], config.dense_dim[1])
        self.mu = nn.Linear(config.dense_dim[1], self.latent_dim)
        self.logvar = nn.Linear(config.dense_dim[1], self.latent_dim)

    def forward(self, input):
        """Encode image input into latent sample, mean, and log-variance."""
        all_hidden_states = ()

        output = input
        if self.hidden_states:
            all_hidden_states = all_hidden_states + (output,)
        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)
            if self.hidden_states:
                all_hidden_states = all_hidden_states + (output,)
        # output = torch.flatten(output, start_dim=1)
        output = self.dense(
            output.contiguous().view(output.size(0), -1)
        )  # 4-D tensor: [Batch, *] --> 2-D tensor: [Batch, latent dim]
        mu = self.mu(output)  # [Batch, latent dim]
        logvar = self.logvar(output)  # [Batch, latent dim]

        z = self.reparameterization(mu, logvar)
        outputs = (z, mu, logvar,) + (
            all_hidden_states,
        )  # (z, mu, logvar, (outputs))
        return outputs

    def reparameterization(self, mu, logvar):
        """Sample latent vector via the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z


class CNN3DShapesEncoder(CNN2DShapesEncoder):
    def __init__(self, config):
        """Construct encoder backbone for RGB (3D Shapes) images."""
        super(CNN3DShapesEncoder, self).__init__(config)
        modules = []
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states

        # Design Encoder Factor-VAE ref
        modules.append(CNNLayer(in_channels=3, out_channels=32))
        modules.append(CNNLayer(in_channels=32, out_channels=32))
        modules.append(CNNLayer(in_channels=32, out_channels=64))
        modules.append(CNNLayer(in_channels=64, out_channels=64))
        self.hidden_layers = nn.ModuleList(modules)

        self.dense = nn.Linear(config.dense_dim[0], config.dense_dim[1])
        self.mu = nn.Linear(config.dense_dim[1], self.latent_dim)
        self.logvar = nn.Linear(config.dense_dim[1], self.latent_dim)

