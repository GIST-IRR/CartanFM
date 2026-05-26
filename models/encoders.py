import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class BaseCNNEncoder(nn.Module):
    
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, 4, 2, 1), nn.ReLU(False),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(False),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(False),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(False),
            nn.Flatten(),
        )
        
        self.dense1 = nn.Linear(64 * 4 * 4, config.encoder_fc_dim1)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(config.encoder_fc_dim1, config.encoder_fc_dim2)
        self.mu = nn.Linear(config.encoder_fc_dim2, config.latent_dim)
        self.logvar = nn.Linear(config.encoder_fc_dim2, config.latent_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.cnn_backbone(input_tensor)
        output = self.relu(self.dense1(output))
        output = self.relu(self.dense2(output))
        mean = self.mu(output)
        logvar = self.logvar(output)
        z = self.reparameterization(mean, logvar)
        return z, mean, logvar

    def reparameterization(self, mean, logvar):
        eps = torch.randn_like(logvar)
        z = mean + eps * (logvar * 0.5).exp()
        return z


class CNNEncoder_for_CartanVAE(BaseCNNEncoder):
    
    
    def __init__(self, config):
        super().__init__(config)
        
        self.subgroup_sizes_ls = config.subgroup_sizes_ls
        self.subspace_sizes_ls = config.subspace_sizes_ls
        self.num_k_factors = config.num_k_factors
        self.num_p_factors = config.num_p_factors

        
        self.dense1_shared = nn.Linear(config.cnn_feature_dim, config.encoder_fc_dim1)
        self.relu_shared = nn.ReLU()

        self.dense2_for_z_coeffs = nn.Linear(config.encoder_fc_dim1, sum(self.subgroup_sizes_ls))
        
        self.active_for_z_coeffs = nn.ReLU()  

        
        self.to_means = nn.ModuleList([])
        self.to_logvar = nn.ModuleList([])
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            self.to_means.append(
                nn.Linear(subgroup_size_i, self.subspace_sizes_ls[i])
            )
            self.to_logvar.append(
                nn.Linear(subgroup_size_i, self.subspace_sizes_ls[i])
            )

        
        if self.num_k_factors > 0:
            self.to_coeffs_k = nn.Linear(config.encoder_fc_dim1, self.num_k_factors)
        else:
            self.to_coeffs_k = None

        if self.num_p_factors > 0:
            self.to_coeffs_p = nn.Linear(config.encoder_fc_dim1, self.num_p_factors)
        else:
            self.to_coeffs_p = None

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        cnn_features = self.cnn_backbone(input_tensor)  
        shared_mlp_features = self.relu_shared(self.dense1_shared(cnn_features))  

        
        group_feats = self.active_for_z_coeffs(self.dense2_for_z_coeffs(shared_mlp_features))
        
        
        b_idx = 0
        means_z_ls, logvars_z_ls = [], []
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            group_feats_i = group_feats[:, b_idx:b_idx + subgroup_size_i]
            
            means_z_i = self.to_means[i](group_feats_i)
            logvars_z_i = self.to_logvar[i](group_feats_i)
            
            
            logvars_z_i = torch.clamp(logvars_z_i, min=-10, max=10)
            
            means_z_ls.append(means_z_i)
            logvars_z_ls.append(logvars_z_i)
            b_idx += subgroup_size_i
        
        mu_z = torch.cat(means_z_ls, dim=-1)
        logvar_z = torch.cat(logvars_z_ls, dim=-1)
        z = self.reparameterization(mu_z, logvar_z)
        
        
        coeffs_k = self.to_coeffs_k(shared_mlp_features) if self.to_coeffs_k else None
        coeffs_p = self.to_coeffs_p(shared_mlp_features) if self.to_coeffs_p else None
        
        return z, mu_z, logvar_z, coeffs_k, coeffs_p
