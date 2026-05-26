

import torch
import torch.nn as nn


class BaseCNNDecoder(nn.Module):
    
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim_total = sum(config.subspace_sizes_ls)
        
        
        self.decoder_fc = nn.Linear(self.latent_dim_total, config.cnn_feature_dim)
        self.decoder_reshape = nn.Unflatten(1, (64, 4, 4))
        
        
        self.decoder_cnn = nn.Sequential(
            nn.ReLU(False),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(False),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(False),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(False),
            nn.ConvTranspose2d(32, config.input_channels, 4, 2, 1),
            nn.Sigmoid() if config.reconstruction_loss_type == 'bce' else nn.Identity()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = self.decoder_reshape(h)
        return self.decoder_cnn(h)
