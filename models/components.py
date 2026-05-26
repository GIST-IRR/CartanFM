

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, cast
from .encoders import CNNEncoder_for_CartanVAE
from .lie_algebra import LieBasisModule_GeneralCartan


class AdaGN(nn.Module):
    
    
    def __init__(self, in_dim: int, cond_dim: int, num_groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, in_dim)
        self.scale_shift = nn.Linear(cond_dim, in_dim * 2)  

    def forward(self, x, cond):
        
        params = self.scale_shift(cond)  
        scale, shift = params.chunk(2, dim=1)  
        scale = scale.unsqueeze(-1).unsqueeze(-1)  
        shift = shift.unsqueeze(-1).unsqueeze(-1)  
        x = self.norm(x)
        x = x * (1 + scale) + shift
        return x  


class ConvBlock(nn.Module):
    
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = AdaGN(out_channels, cond_dim, num_groups)
        self.norm2 = AdaGN(out_channels, cond_dim, num_groups)
        self.activation = nn.ReLU(inplace=True)  
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x, cond)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x, cond)
        x = self.activation(x)
        return x


class DownBlock(nn.Module):
    
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, cond_dim, num_groups)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)  
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block(x, cond)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)  
        self.conv_block = ConvBlock(in_channels + skip_channels, out_channels, cond_dim, num_groups)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x, cond)
        return x


class MiddleBlock(nn.Module):
    
    
    def __init__(self, channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv_block1 = ConvBlock(channels, channels, cond_dim, num_groups)
        self.conv_block2 = ConvBlock(channels, channels, cond_dim, num_groups)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x, cond)
        x = self.conv_block2(x, cond)
        return x


class CycleVectorFieldNet(nn.Module):
    
    
    def __init__(self, encoder: CNNEncoder_for_CartanVAE, lie_basis_module: LieBasisModule_GeneralCartan, config, image_size: int = 64):
        super().__init__()
        self.encoder = encoder
        self.lie_basis_module = lie_basis_module
        self.config = config
        self.image_size = image_size
        
        self.n_freqs = config.n_freqs
        
        
        base_channels = 64
        cond_dim = lie_basis_module.matrix_dim ** 2
        
        
        input_channels = config.input_channels + 2 * self.n_freqs
        
        
        self.down1 = DownBlock(input_channels, base_channels, cond_dim, config.num_groups)  
        self.down2 = DownBlock(base_channels, base_channels * 2, cond_dim, config.num_groups)  
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, cond_dim, config.num_groups)  
        
        
        self.middle = MiddleBlock(base_channels * 4, cond_dim, config.num_groups)  
        
        
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, cond_dim, config.num_groups)  
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, cond_dim, config.num_groups)      
        self.up3 = UpBlock(base_channels, base_channels, base_channels // 2, cond_dim, config.num_groups)         
        
        
        self.final_conv = nn.Conv2d(base_channels // 2, config.input_channels, 3, padding=1)

    def time_encoder(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        
        B, C, H, W = x_shape
        
        
        if t.dim() == 0:  
            t = t.expand(B)  
        elif t.dim() == 1 and t.size(0) == 1:  
            t = t.expand(B)  
        elif t.dim() == 1 and t.size(0) != B:  
            t = t.expand(B)  
            
        t = t.view(B, 1, 1, 1)  

        freqs = torch.arange(0, self.n_freqs, device=t.device).float()  
        freqs = freqs.view(1, -1, 1, 1)  
        time_enc = 2 * torch.pi * freqs * t  
        time_enc = torch.cat([torch.sin(time_enc), torch.cos(time_enc)], dim=1)  
        time_enc = time_enc.expand(-1, -1, H, W)  
        return time_enc

    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        
        cond = kwargs.get("cond", None)
        if cond is None:
            raise ValueError("Condition must be provided")
        
        
        time_enc = self.time_encoder(t, x.shape)  
        x = torch.cat([x, time_enc], dim=1)  

        
        x, skip1 = self.down1(x, cond)  
        x, skip2 = self.down2(x, cond)  
        x, skip3 = self.down3(x, cond)  
        
        x = self.middle(x, cond)  
        
        
        x = self.up1(x, skip3, cond)  
        x = self.up2(x, skip2, cond)  
        x = self.up3(x, skip1, cond)  
        
        x = self.final_conv(x)
        return x

    def calc_basis_norm_loss(self) -> torch.Tensor:
        
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.lie_basis_module.raw_k_basis_params is not None:
            k_basis = self.lie_basis_module.raw_k_basis_params
            loss += torch.sum(torch.norm(k_basis, dim=(1, 2)) ** 2)
            
        if self.lie_basis_module.raw_p_basis_params is not None:
            p_basis = self.lie_basis_module.raw_p_basis_params
            loss += torch.sum(torch.norm(p_basis, dim=(1, 2)) ** 2)
            
        return loss

    def calc_non_triviality_loss(self, epsilon=1e-8) -> torch.Tensor:
        
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.lie_basis_module.raw_k_basis_params is not None:
            k_basis = self.lie_basis_module.raw_k_basis_params
            k_norms = torch.norm(k_basis, dim=(1, 2))
            loss += torch.sum(1.0 / (k_norms + epsilon))
            
        if self.lie_basis_module.raw_p_basis_params is not None:
            p_basis = self.lie_basis_module.raw_p_basis_params
            p_norms = torch.norm(p_basis, dim=(1, 2))
            loss += torch.sum(1.0 / (p_norms + epsilon))
            
        return loss


class SimpleVectorFieldNet(nn.Module):
    
    
    def __init__(self, encoder: CNNEncoder_for_CartanVAE, config, image_size: int = 64):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.image_size = image_size
        
        
        self.n_freqs = config.n_freqs  
        
        
        self.cond_dim = config.num_k_factors + config.num_p_factors  
        
        
        base_channels = 64
        
        self.input_conv = nn.Conv2d(config.input_channels + 2 * self.n_freqs, base_channels, 3, padding=1)
        
        
        self.down1 = DownBlock(base_channels, base_channels * 2, self.cond_dim, config.num_groups)      
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, self.cond_dim, config.num_groups)  
        self.down3 = DownBlock(base_channels * 4, base_channels * 4, self.cond_dim, config.num_groups)  
        
        
        self.middle = MiddleBlock(base_channels * 4, self.cond_dim, config.num_groups)  
        
        
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, self.cond_dim, config.num_groups)  
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, self.cond_dim, config.num_groups)      
        self.up3 = UpBlock(base_channels, base_channels, base_channels // 2, self.cond_dim, config.num_groups)         
        
        
        self.output_conv = nn.Conv2d(base_channels // 2, config.input_channels, 3, padding=1)  

    def time_encoder(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        
        B, C, H, W = x_shape
        
        
        if t.dim() == 0:  
            t = t.expand(B)  
        elif t.dim() == 1 and t.size(0) == 1:  
            t = t.expand(B)  
        elif t.dim() == 1 and t.size(0) != B:  
            t = t.expand(B)  
            
        t = t.view(B, 1, 1, 1)  

        freqs = torch.arange(0, self.n_freqs, device=t.device).float()  
        freqs = freqs.view(1, -1, 1, 1)  
        time_enc = 2 * torch.pi * freqs * t  
        time_enc = torch.cat([torch.sin(time_enc), torch.cos(time_enc)], dim=1)  
        time_enc = time_enc.expand(-1, -1, H, W)  
        return time_enc

    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        
        cond = kwargs.get("cond", None)
        if cond is None:
            raise ValueError("Condition must be provided")
        
        
        time_embed = self.time_encoder(t, x.shape)  
        x = torch.cat([x, time_embed], dim=1)  
        
        
        
        
        h = self.input_conv(x)
        
        
        h1, skip1 = self.down1(h, cond)
        h2, skip2 = self.down2(h1, cond)
        h3, skip3 = self.down3(h2, cond)
        
        
        h = self.middle(h3, cond)
        
        
        h = self.up1(h, skip3, cond)
        h = self.up2(h, skip2, cond)
        h = self.up3(h, skip1, cond)
        
        
        out = self.output_conv(h)
        
        return out

    def calc_basis_norm_loss(self) -> torch.Tensor:
        
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def calc_non_triviality_loss(self) -> torch.Tensor:
        
        return torch.tensor(0.0, device=next(self.parameters()).device)


class BaselineVectorFieldNet(nn.Module):
    
    
    def __init__(self, encoder: CNNEncoder_for_CartanVAE, config, image_size: int = 64):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.image_size = image_size
        
        
        self.n_freqs = config.n_freqs
        
        
        self.cond_dim = config.latent_dim  
        
        
        base_channels = 64
        
        input_channels = config.input_channels + 2 * self.n_freqs
        
        
        self.down1 = DownBlock(input_channels, base_channels, self.cond_dim, config.num_groups)  
        self.down2 = DownBlock(base_channels, base_channels * 2, self.cond_dim, config.num_groups)  
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, self.cond_dim, config.num_groups)  
        
        
        self.middle = MiddleBlock(base_channels * 4, self.cond_dim, config.num_groups)  
        
        
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, self.cond_dim, config.num_groups)  
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, self.cond_dim, config.num_groups)      
        self.up3 = UpBlock(base_channels, base_channels, base_channels // 2, self.cond_dim, config.num_groups)         
        
        
        self.final_conv = nn.Conv2d(base_channels // 2, config.input_channels, 3, padding=1)

    def time_encoder(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        
        B, C, H, W = x_shape
        
        
        if t.dim() == 0:
            t = t.expand(B)
        elif t.dim() == 1 and t.size(0) == 1:
            t = t.expand(B)
        elif t.dim() == 1 and t.size(0) != B:
            t = t.expand(B)
            
        t = t.view(B, 1, 1, 1)  

        freqs = torch.arange(0, self.n_freqs, device=t.device).float()  
        freqs = freqs.view(1, -1, 1, 1)  
        time_enc = 2 * torch.pi * freqs * t  
        time_enc = torch.cat([torch.sin(time_enc), torch.cos(time_enc)], dim=1)  
        time_enc = time_enc.expand(-1, -1, H, W)  
        return time_enc

    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        
        cond = kwargs.get("cond", None)
        if cond is None:
            raise ValueError("Condition must be provided")
        
        
        time_enc = self.time_encoder(t, x.shape)  
        x = torch.cat([x, time_enc], dim=1)  

        
        x, skip1 = self.down1(x, cond)  
        x, skip2 = self.down2(x, cond)  
        x, skip3 = self.down3(x, cond)  
        
        x = self.middle(x, cond)  
        
        
        x = self.up1(x, skip3, cond)  
        x = self.up2(x, skip2, cond)  
        x = self.up3(x, skip1, cond)  
        
        x = self.final_conv(x)
        return x

    def calc_basis_norm_loss(self) -> torch.Tensor:
        
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def calc_non_triviality_loss(self) -> torch.Tensor:
        
        return torch.tensor(0.0, device=next(self.parameters()).device)


class SimpleVAEEncoder(nn.Module):
    
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        
        self.conv1 = nn.Conv2d(config.in_channel, 32, 4, 2, 1)  
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)               
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)               
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)               
        
        
        self.flatten_dim = 64 * 4 * 4
        
        
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        
        self.fc_mu = nn.Linear(128, config.latent_dim)
        self.fc_logvar = nn.Linear(128, config.latent_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        
        
        h = h.view(h.size(0), -1)
        
        
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        
        
        mu_z = self.fc_mu(h)
        logvar_z = self.fc_logvar(h)
        
        
        z = self.reparameterize(mu_z, logvar_z)
        
        return z, mu_z, logvar_z
    
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class NoLieVectorFieldNet(nn.Module):
    
    
    def __init__(self, encoder: SimpleVAEEncoder, config, image_size: int = 64):
        super().__init__()
        self.config = config
        
        self.image_size = image_size
        
        
        self.n_freqs = config.n_freqs
        
        
        self.cond_dim = config.latent_dim  
        
        
        base_channels = 64
        
        input_channels = config.input_channels + 2 * self.n_freqs
        
        
        self.down1 = DownBlock(input_channels, base_channels, self.cond_dim, config.num_groups)  
        self.down2 = DownBlock(base_channels, base_channels * 2, self.cond_dim, config.num_groups)  
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, self.cond_dim, config.num_groups)  
        
        
        self.middle = MiddleBlock(base_channels * 4, self.cond_dim, config.num_groups)  
        
        
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, self.cond_dim, config.num_groups)  
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, self.cond_dim, config.num_groups)      
        self.up3 = UpBlock(base_channels, base_channels, base_channels // 2, self.cond_dim, config.num_groups)         
        
        
        self.final_conv = nn.Conv2d(base_channels // 2, config.input_channels, 3, padding=1)

    def time_encoder(self, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        
        B, C, H, W = x_shape
        
        
        if t.dim() == 0:
            t = t.expand(B)
        elif t.dim() == 1 and t.size(0) == 1:
            t = t.expand(B)
        elif t.dim() == 1 and t.size(0) != B:
            t = t.expand(B)
            
        t = t.view(B, 1, 1, 1)  

        freqs = torch.arange(0, self.n_freqs, device=t.device).float()  
        freqs = freqs.view(1, -1, 1, 1)  
        time_enc = 2 * torch.pi * freqs * t  
        time_enc = torch.cat([torch.sin(time_enc), torch.cos(time_enc)], dim=1)  
        time_enc = time_enc.expand(-1, -1, H, W)  
        return time_enc

    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs):

        cond = kwargs.get('cond')
        if cond is None:
            raise ValueError("cond must be provided for NoLieVectorFieldNet")
        
        
        time_enc = self.time_encoder(t, x.shape)  
        x = torch.cat([x, time_enc], dim=1)  

        
        x, skip1 = self.down1(x, cond)  
        x, skip2 = self.down2(x, cond)  
        x, skip3 = self.down3(x, cond)  
        
        x = self.middle(x, cond)  
        
        
        x = self.up1(x, skip3, cond)  
        x = self.up2(x, skip2, cond)  
        x = self.up3(x, skip1, cond)  
        
        x = self.final_conv(x)
        return x


class CondVF(nn.Module):
    
    
    def __init__(self, net: nn.Module, n_steps: int = 100, ode_solver: str = 'dopri5') -> None:
        super().__init__()
        self.net = net
        self.n_steps = n_steps
        self.ode_solver = ode_solver
        
    def forward(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(t, x, **kwargs)
    
    def wrapper(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x, **kwargs)

    def encode(self, x_1: torch.Tensor, return_all: bool = False, **kwargs) -> torch.Tensor:
        
        from torchdiffeq import odeint_adjoint as odeint
        from functools import partial
        
        wrapped_func = partial(self.wrapper, **kwargs)
        if return_all:
            return odeint(wrapped_func, x_1, torch.linspace(1., 0., self.n_steps).to(x_1.device), 
                         method=self.ode_solver, adjoint_params=self.parameters())
        else:
            return odeint(wrapped_func, x_1, torch.linspace(1., 0., self.n_steps).to(x_1.device), 
                         method=self.ode_solver, adjoint_params=self.parameters())[-1]

    def decode(self, x_0: torch.Tensor, return_all: bool = False, **kwargs) -> torch.Tensor:
        
        from torchdiffeq import odeint_adjoint as odeint
        from functools import partial
        
        wrapped_func = partial(self.wrapper, **kwargs)
        if return_all:
            return odeint(wrapped_func, x_0, torch.linspace(0., 1., self.n_steps).to(x_0.device), 
                         method=self.ode_solver, adjoint_params=self.parameters())
        else:
            return odeint(wrapped_func, x_0, torch.linspace(0., 1., self.n_steps).to(x_0.device), 
                         method=self.ode_solver, adjoint_params=self.parameters())[-1]
