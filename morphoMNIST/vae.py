import torch
import torch.nn as nn


"""Lightweight VAE model used in MorphoMNIST experiments."""


class CNNLayer(nn.Module):
    """Downsampling convolution block: Conv2D + ReLU."""

    def __init__(self, in_channels, out_channels):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        """Apply convolution and activation."""
        output = self.conv(input)
        output = self.relu(output)
        return output


class CNNTransposedLayer(nn.Module):
    """Upsampling transposed-convolution block: ConvTranspose2D + ReLU."""

    def __init__(self, in_channels, out_channels):
        super(CNNTransposedLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.ReLU(True)

    def forward(self, input):
        """Apply transposed convolution and activation."""
        output = self.conv(input)
        output = self.relu(output)
        return output

class Encoder(nn.Module):
    """Encode image to 2D latent Gaussian parameters."""

    def __init__(self):
        """Build convolutional feature extractor and latent heads."""
        super().__init__()
        self.encoder = nn.Sequential(
            CNNLayer(1, 16),
            CNNLayer(16, 32),
            CNNLayer(32, 32),
        )
        self.mean = nn.Linear(32, 2)
        self.logvar = nn.Linear(32, 2)
        
    def forward(self, x):
        """Return latent mean and log-variance for input image batch."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(1)
        w = self.encoder(x)
        w = w.view(-1, 32)
        mean = self.mean(w)
        logvar = self.logvar(w)
        return mean, logvar
    
    def w_to_z(self, w):
        """Project hidden vector to latent mean/log-variance."""
        return self.mean(w), self.logvar(w)
    
class Decoder(nn.Module):
    """Decode 2D latent vectors into reconstructed images."""

    def __init__(self):
        """Build linear projection + transposed-convolution decoder."""
        super().__init__()
        self.dec_linear = nn.Linear(2, 7*7*32)
        self.decoder = nn.Sequential(
            CNNTransposedLayer(32, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        """Decode latent vectors into pixel-space probabilities."""
        z = self.dec_linear(z)
        z = z.view(-1, 32, 7, 7)
        output = self.decoder(z).squeeze()
        return output

class VAE(nn.Module):
    """Minimal convolutional VAE for MorphoMNIST experiments."""

    def __init__(self):
        """Initialize encoder and decoder submodules."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterization(self, mean, logvar):
        """Sample latent code via the reparameterization trick."""
        eps = torch.rand(logvar.shape).cuda()
        z = mean + eps * (logvar * 0.5).exp()
        return z

    def forward(self, x):
        """Return reconstruction, latent sample, mean, and log-variance."""
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        output = self.decoder(z)
        return output, z, mean, logvar