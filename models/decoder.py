import torch
import torch.nn as nn
import math
import numpy as np
from models.cnn_layer import CNNTransposedLayer


"""CNN-based decoders for dSprites and 3D Shapes datasets."""


class CNN2DShapesDecoder(nn.Module):
    def __init__(self, config):
        """Construct decoder backbone for grayscale (dSprites) outputs."""
        super(CNN2DShapesDecoder, self).__init__()
        modules = []
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states

        # Design Decoder Factor-VAE ref
        self.dense1 = nn.Linear(self.latent_dim, config.dense_dim[1])
        self.dense2 = nn.Linear(config.dense_dim[1], 4 * config.dense_dim[0])
        self.relu = nn.ReLU(True)
        self.active = nn.Sigmoid()

        modules.append(CNNTransposedLayer(in_channels=64, out_channels=64))
        modules.append(CNNTransposedLayer(in_channels=64, out_channels=32))
        modules.append(CNNTransposedLayer(in_channels=32, out_channels=32))
        modules.append(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1
            )
        )
        self.hidden_layers = nn.ModuleList(modules)

    def forward(self, input):
        """Decode latent vector into reconstructed image logits."""
        # all_hidden_states = ()

        output = self.dense1(input)
        output = self.relu(output)
        output = self.dense2(output)
        output = self.relu(output)  # (B, ...)

        if len(output.shape) == 1:
            output = output.unsqueeze(0)

        output = output.view(output.size(0), 64, 4, 4)

        for i, hidden_layer in enumerate(self.hidden_layers):
            output = hidden_layer(output)

        # To use BCEWithLogitsLoss, do not use Sigmoid
        #output = self.active(output)

        outputs = (output,)  # + (all_hidden_states,)
        return outputs


class CNN3DShapesDecoder(CNN2DShapesDecoder):
    def __init__(self, config):
        """Construct decoder backbone for RGB (3D Shapes) outputs."""
        super(CNN3DShapesDecoder, self).__init__(config)

        modules = []
        self.latent_dim = config.latent_dim
        self.hidden_states = config.hidden_states

        # Design Decoder Factor-VAE ref
        self.dense1 = nn.Linear(config.latent_dim, config.dense_dim[1])
        self.dense2 = nn.Linear(config.dense_dim[1], 4 * config.dense_dim[0])
        self.relu = nn.ReLU(True)
        self.active = nn.Sigmoid()

        modules.append(CNNTransposedLayer(in_channels=64, out_channels=64))
        modules.append(CNNTransposedLayer(in_channels=64, out_channels=32))
        modules.append(CNNTransposedLayer(in_channels=32, out_channels=32))
        modules.append(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1
            )
        )
        self.hidden_layers = nn.ModuleList(modules)

