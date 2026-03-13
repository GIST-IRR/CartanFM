import numpy as np
import torch
import torch.nn as nn

from models.cnn_layer import CNNLayer
from models.glow import ReallyInvertibleFlowNet


"""Core encoder/decoder modules used by MAGANet-family models."""


class Encoder(nn.Module):
    def __init__(self, config):
        """Construct convolutional encoder for latent Gaussian parameters."""
        super(Encoder, self).__init__()
        self.latent_dim = config.latent_dim

        input_channels = config.in_channel

        self.cnn1 = CNNLayer(in_channels=input_channels, out_channels=32)
        self.cnn2 = CNNLayer(32, 32)
        self.cnn3 = CNNLayer(32, 32)
        self.cnn4 = CNNLayer(32, 32)
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, self.latent_dim)
        self.logvar = nn.Linear(256, self.latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Encode image tensor to latent sample, mean, and log-variance."""
        output = x

        output = self.cnn1(output)
        output = self.cnn2(output)
        output = self.cnn3(output)
        output = self.cnn4(output)
        output = output.contiguous().view(output.size(0), -1)
        output = self.relu(self.linear1(output))
        output = self.relu(self.linear2(output))
        mu = self.mu(output)
        logvar = self.logvar(output)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample latent vector using reparameterization."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def w_to_mu(self, w):
        """Project intermediate vector to latent mean."""
        return self.mu(w)

    def w_to_logvar(self, w):
        """Project intermediate vector to latent log-variance."""
        return self.logvar(w)


class Decoder(nn.Module):
    def __init__(self, config):
        """Construct flow-based decoder with latent action operator."""
        super(Decoder, self).__init__()
        self.latent_dim = config.latent_dim
        self.channels = config.in_channel
        self.input_size = config.image_shape[-1]
        self.flow = ReallyInvertibleFlowNet(config)
        self.action = nn.Linear(
            self.latent_dim, self.input_size**2 * self.channels, bias=False
        )
        nn.init.normal_(self.action.weight)
        self.pivot = None

    def set_pivot(self, pivot):
        """Register pivot image for inference-time decoding."""
        self.pivot = pivot
        self.pivot = self.pivot.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, z, x1=None):
        """Decode latent displacement `z` conditioned on source image `x1`."""
        if x1 is None:
            x1 = self.pivot.unsqueeze(0).repeat(z.shape[0], 1, 1, 1)
        x_prime, logdet = self.flow(x1)
        x_prime_shape = x_prime.shape
        action = self.action(z)
        rho = x_prime.view(action.shape) + action
        output = self.flow(rho.view(x_prime_shape), reverse=True)
        return output
