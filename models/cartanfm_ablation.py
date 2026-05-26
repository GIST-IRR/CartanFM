"""
CartanFM Ablation Study Models

Various CartanFM model versions for ablation study:
1. Original: Lie algebra + Cycle consistency
2. NoCycle: Lie algebra only (no cycle consistency)
3. NoLie: Cycle consistency only (no Lie algebra)  
4. Baseline: Neither used (basic flow matching only)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .components import (
    CondVF, CycleVectorFieldNet, SimpleVectorFieldNet, BaselineVectorFieldNet,
    SimpleVAEEncoder, NoLieVectorFieldNet  
)
from .lie_algebra import LieBasisModule_GeneralCartan
from .encoders import CNNEncoder_for_CartanVAE
from .flow_matching_ablation import (
    OriginalFlowMatching, 
    NoCycleFlowMatching,
    NoLieFlowMatching, 
    BaselineFlowMatching
)


class BaseCartanFM(nn.Module):
    
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = getattr(config, 'model_type', 'original')
        
        
        self.encoder = CNNEncoder_for_CartanVAE(config)
        
        
        self._setup_networks()
        
    def _setup_networks(self):
        
        pass
        
    def forward(self, x: torch.Tensor, return_all: bool = False):
        
        
        z, mu_z, logvar_z, coeffs_k, coeffs_p = self.encoder(x)
        
        
        cond = self._generate_condition(coeffs_k, coeffs_p)
        
        
        
        x_recon = None  
        
        if return_all:
            result = {
                'x_recon': x_recon,
                'z': z,
                'mu_z': mu_z,
                'logvar_z': logvar_z,
                'coeffs_k': coeffs_k,
                'coeffs_p': coeffs_p,
                'cond': cond
            }
            
            if hasattr(self, 'lie_basis_module'):
                g_total, g_k_component = self.lie_basis_module(coeffs_k, coeffs_p)
                result.update({
                    'g_total': g_total,
                    'g_k_component': g_k_component
                })
            return result
        
        return x_recon
    
    def _generate_condition(self, coeffs_k, coeffs_p):
        
        raise NotImplementedError
    
    def training_step(self, x: torch.Tensor):
        
        
        results = self.forward(x, return_all=True)
        
        
        kwargs = {
            'cond': results['cond'],
            'mu_z': results['mu_z'],
            'logvar_z': results['logvar_z'],
            'coeffs_k': results['coeffs_k'],
            'coeffs_p': results['coeffs_p']
        }
        
        
        if results is not None and 'g_k_component' in results:
            kwargs['g_k_component'] = results['g_k_component']
        
        
        loss_result = self.fm_loss.loss(self.condvf, x, **kwargs)
        
        
        if isinstance(loss_result, tuple):
            
            total_loss, loss_dict = loss_result
        else:
            
            loss_dict = loss_result
            total_loss = loss_dict['total_loss']
        
        return total_loss, loss_dict


class OriginalCartanFM(BaseCartanFM):
    
    
    def _setup_networks(self):
        
        self.lie_basis_module = LieBasisModule_GeneralCartan(
            matrix_dim=self.config.lie_algebra_matrix_dim,
            num_k_factors=self.config.num_k_factors,
            num_p_factors=self.config.num_p_factors
        )
        
        
        self.net = CycleVectorFieldNet(
            encoder=self.encoder,
            lie_basis_module=self.lie_basis_module,
            config=self.config
        )
        
        
        self.condvf = CondVF(
            net=self.net,
            n_steps=100,
            ode_solver=self.config.ode_solver
        )
        
        
        self.fm_loss = OriginalFlowMatching(device=self.config.device)
    
    def _generate_condition(self, coeffs_k, coeffs_p):
        
        g_total, _ = self.lie_basis_module(coeffs_k, coeffs_p)
        return g_total.view(coeffs_k.shape[0], -1)


class NoCycleCartanFM(BaseCartanFM):
    
    
    def _setup_networks(self):
        
        self.lie_basis_module = LieBasisModule_GeneralCartan(
            matrix_dim=self.config.lie_algebra_matrix_dim,
            num_k_factors=self.config.num_k_factors,
            num_p_factors=self.config.num_p_factors
        )
        
        
        self.net = CycleVectorFieldNet(
            encoder=self.encoder,
            lie_basis_module=self.lie_basis_module,
            config=self.config
        )
        
        
        self.condvf = CondVF(
            net=self.net,
            n_steps=100,
            ode_solver=self.config.ode_solver
        )
        
        
        self.fm_loss = NoCycleFlowMatching(device=self.config.device)
    
    def _generate_condition(self, coeffs_k, coeffs_p):
        
        g_total, _ = self.lie_basis_module(coeffs_k, coeffs_p)
        return g_total.view(coeffs_k.shape[0], -1)


class NoLieCartanFM(nn.Module):
    
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = getattr(config, 'model_type', 'nolie')
        
        
        self.encoder = SimpleVAEEncoder(self.config)
        
        
        self._setup_networks()
    
    def _setup_networks(self):
        
        self.net = NoLieVectorFieldNet(
            encoder=self.encoder,
            config=self.config
        )
        
        
        self.condvf = CondVF(
            net=self.net,
            n_steps=100,
            ode_solver=self.config.ode_solver
        )
        
        
        self.condvf._encoder_ref = self.encoder
        
        
        self.fm_loss = NoLieFlowMatching(device=self.config.device)
    
    def forward(self, x: torch.Tensor, return_all: bool = False):
        
        
        z, mu_z, logvar_z = self.encoder(x)
        
        
        cond = z  
        
        
        x_recon = None
        
        if return_all:
            result = {
                'x_recon': x_recon,
                'z': z,
                'mu_z': mu_z,
                'logvar_z': logvar_z,
                'coeffs_k': None,  
                'coeffs_p': None,  
                'cond': cond
            }
            return result
        
        return x_recon
    
    def training_step(self, x: torch.Tensor):
        
        
        z, mu_z, logvar_z = self.encoder(x)
        
        
        kwargs = {
            'cond': z,  
            'mu_z': mu_z,
            'logvar_z': logvar_z,
        }
        
        
        loss_result = self.fm_loss.loss(self.condvf, x, **kwargs)
        
        
        if isinstance(loss_result, tuple):
            
            total_loss, loss_dict = loss_result
        else:
            
            loss_dict = loss_result
            total_loss = loss_dict['total_loss']
        
        return total_loss, loss_dict
    
    def forward(self, x: torch.Tensor, return_all: bool = False):
        
        
        z, mu_z, logvar_z = self.encoder(x)
        
        
        cond = z  
        
        
        x_recon = None
        
        if return_all:
            result = {
                'x_recon': x_recon,
                'z': z,
                'mu_z': mu_z,
                'logvar_z': logvar_z,
                'coeffs_k': None,  
                'coeffs_p': None,  
                'cond': cond
            }
            return result
        
        return x_recon
    
    def training_step(self, x: torch.Tensor):
        
        
        z, mu_z, logvar_z = self.encoder(x)
        
        
        kwargs = {
            'cond': z,  
            'mu_z': mu_z,
            'logvar_z': logvar_z,
        }
        
        
        loss_result = self.fm_loss.loss(self.condvf, x, **kwargs)
        
        
        if isinstance(loss_result, tuple):
            
            total_loss, loss_dict = loss_result
        else:
            
            loss_dict = loss_result
            total_loss = loss_dict['total_loss']
        
        return total_loss, loss_dict
    
    def _generate_condition(self, coeffs_k, coeffs_p):
        
        raise NotImplementedError("NoLie model does not use coeffs_k and coeffs_p")


class BaselineCartanFM(BaseCartanFM):
    
    
    def _setup_networks(self):
        
        from .components import BaselineVectorFieldNet
        
        self.net = BaselineVectorFieldNet(
            encoder=self.encoder,
            config=self.config
        )
        
        
        self.condvf = CondVF(
            net=self.net,
            n_steps=100,
            ode_solver=self.config.ode_solver
        )
        
        
        self.fm_loss = BaselineFlowMatching(device=self.config.device)
    
    def forward(self, x: torch.Tensor, return_all: bool = False):
        
        
        z, mu_z, logvar_z, coeffs_k, coeffs_p = self.encoder(x)
        
        
        cond = mu_z  
        
        
        x_recon = None  
        
        if return_all:
            result = {
                'x_recon': x_recon,
                'z': z,
                'mu_z': mu_z,
                'logvar_z': logvar_z,
                'coeffs_k': coeffs_k,
                'coeffs_p': coeffs_p,
                'cond': cond
            }
            return result
        
        return x_recon
    
    def _generate_condition(self, coeffs_k, coeffs_p):
        
        
        raise NotImplementedError("BaselineCartanFM uses z directly in forward method")


def create_cartanfm_model(config):
    
    model_type = getattr(config, 'model_type', 'original')
    
    if model_type == 'original':
        return OriginalCartanFM(config)
    elif model_type == 'nocycle':
        return NoCycleCartanFM(config)
    elif model_type == 'nolie':
        return NoLieCartanFM(config)
    elif model_type == 'baseline':
        return BaselineCartanFM(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of ['original', 'nocycle', 'nolie', 'baseline']")
