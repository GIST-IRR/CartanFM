import torch.nn as nn


"""Reusable convolutional building blocks for encoder/decoder stacks."""


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """A simple Conv2D + ReLU block with stride-2 downsampling."""
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        """Apply convolution then nonlinearity."""
        output = self.conv(input)
        output = self.relu(output)
        return output


class CNNTransposedLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """A transposed-convolution + ReLU block for upsampling."""
        super(CNNTransposedLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.relu = nn.ReLU(True)

    def forward(self, input):
        """Apply transposed convolution then nonlinearity."""
        output = self.conv(input)
        output = self.relu(output)
        return output
