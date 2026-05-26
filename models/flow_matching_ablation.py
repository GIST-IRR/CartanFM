

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .lie_algebra import calculate_cartan_commutation_losses


class BaselineFlowMatching:
    
    
    def __init__(self, sig_min: float = 1e-6, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.sig_min = sig_min * torch.ones(1, device=self.device)
        self.eps = 1e-6

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        sig_min = self.sig_min.to(x.device)
        return (1 - (1 - sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        
        cond = kwargs["cond"]
        mu_z = kwargs.get("mu_z", None)
        logvar_z = kwargs.get("logvar_z", None)
        
        if mu_z is None or logvar_z is None:
            raise ValueError("mu_z and logvar_z must be provided")
        
        B = x_1.shape[0]
        t = torch.rand(B, device=x_1.device) * (1 - self.eps)
        t_expand = t.view(B, 1, 1, 1)
        x_0 = torch.randn_like(x_1).to(x_1.device)
        
        
        v_psi = v_t(t, self.psi_t(x_0, x_1, t_expand), **{"cond": cond})
        sig_min = self.sig_min.to(x_1.device)
        d_psi = x_1 - (1 - sig_min) * x_0

        
        kld_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp()) / B

        
        cfm_loss = torch.sum(torch.pow(v_psi - d_psi, 2)) / B

        
        total_loss = cfm_loss + v_t.net.config.beta_kld * kld_loss

        return {
            'total_loss': total_loss,
            'cfm_loss': cfm_loss,
            'kld_loss': kld_loss,
            'geodesic_loss': torch.tensor(0.0, device=x_1.device),
            'loss_kk_ortho_p': torch.tensor(0.0, device=x_1.device),
            'loss_kp_ortho_k': torch.tensor(0.0, device=x_1.device),
            'loss_pp_ortho_p': torch.tensor(0.0, device=x_1.device),
            'basis_norm_loss': torch.tensor(0.0, device=x_1.device),
            'loss_non_triviality': torch.tensor(0.0, device=x_1.device),
            'cycle_consistency_loss': torch.tensor(0.0, device=x_1.device)
        }


class NoLieFlowMatching:
    
    
    def __init__(self, sig_min: float = 1e-6, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.sig_min = sig_min * torch.ones(1, device=self.device)
        self.eps = 1e-6

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        sig_min = self.sig_min.to(x.device)
        return (1 - (1 - sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        
        cond = kwargs["cond"]
        mu_z = kwargs.get("mu_z", None)
        logvar_z = kwargs.get("logvar_z", None)
        coeffs_k = kwargs.get("coeffs_k", None)
        coeffs_p = kwargs.get("coeffs_p", None)
        
        if mu_z is None or logvar_z is None:
            raise ValueError("mu_z and logvar_z must be provided")
        
        
        B = x_1.shape[0]
        t = torch.rand(B, device=x_1.device) * (1 - self.eps)
        t_expand = t.view(B, 1, 1, 1)
        x_0 = torch.randn_like(x_1).to(x_1.device)
        
        
        v_psi = v_t(t, self.psi_t(x_0, x_1, t_expand), **{"cond": cond})
        sig_min = self.sig_min.to(x_1.device)
        d_psi = x_1 - (1 - sig_min) * x_0

        
        kld_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp()) / B

        
        cfm_loss = torch.sum(torch.pow(v_psi - d_psi, 2)) / B

        
        if coeffs_k is not None:
            coeffs_k_rand = torch.randn_like(coeffs_k) * 5.0
        else:
            coeffs_k_rand = None
            
        if coeffs_p is not None:
            coeffs_p_rand = torch.randn_like(coeffs_p) * 5.0
        else:
            coeffs_p_rand = None
            
        
        
        z_rand = torch.randn_like(cond.view(B, -1))
        
        
        t_zero = torch.zeros(B, device=x_1.device)
        x_rand_out = v_t.net(t_zero, x_0, **{"cond": z_rand})
        sig_min = self.sig_min.to(x_1.device)
        x_rand = x_rand_out + (1 - sig_min) * x_0
        
        
        
        model_encoder = getattr(v_t, '_encoder_ref', None)
        if model_encoder is None:
            
            model_encoder = getattr(v_t.net, 'encoder', None)
        if model_encoder is None:
            raise ValueError("Encoder not found for NoLie model")
        
        try:
            encoder_result = model_encoder(x_rand)
            if len(encoder_result) >= 3:
                z_rand_cycle = encoder_result[0]  
            else:
                z_rand_cycle = encoder_result
        except:
            
            z_rand_cycle, _, _ = model_encoder(x_rand)

        
        cycle_consistency_loss = torch.sum(torch.pow(z_rand - z_rand_cycle, 2)) / B

        
        total_loss = (
            cfm_loss +
            v_t.net.config.beta_kld * kld_loss +
            v_t.net.config.cycle_consistency_weight * cycle_consistency_loss
        )

        return {
            'total_loss': total_loss,
            'cfm_loss': cfm_loss,
            'kld_loss': kld_loss,
            'geodesic_loss': torch.tensor(0.0, device=x_1.device),
            'loss_kk_ortho_p': torch.tensor(0.0, device=x_1.device),
            'loss_kp_ortho_k': torch.tensor(0.0, device=x_1.device),
            'loss_pp_ortho_p': torch.tensor(0.0, device=x_1.device),
            'basis_norm_loss': torch.tensor(0.0, device=x_1.device),
            'loss_non_triviality': torch.tensor(0.0, device=x_1.device),
            'cycle_consistency_loss': cycle_consistency_loss
        }


class NoCycleFlowMatching:
    
    
    def __init__(self, sig_min: float = 1e-6, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.sig_min = sig_min * torch.ones(1, device=self.device)
        self.eps = 1e-6

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        sig_min = self.sig_min.to(x.device)
        return (1 - (1 - sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        
        cond = kwargs["cond"]
        mu_z = kwargs.get("mu_z", None)
        logvar_z = kwargs.get("logvar_z", None)
        coeffs_k = kwargs.get("coeffs_k", None)
        coeffs_p = kwargs.get("coeffs_p", None)
        g_k_component = kwargs.get("g_k_component", None)
        
        if mu_z is None or logvar_z is None:
            raise ValueError("mu_z and logvar_z must be provided")
        if coeffs_k is None or coeffs_p is None:
            raise ValueError("coeffs_k and coeffs_p must be provided")
        if g_k_component is None:
            raise ValueError("g_k_component must be provided")
        
        B = x_1.shape[0]
        t = torch.rand(B, device=x_1.device) * (1 - self.eps)
        t_expand = t.view(B, 1, 1, 1)
        x_0 = torch.randn_like(x_1).to(x_1.device)
        
        
        v_psi = v_t(t, self.psi_t(x_0, x_1, t_expand), **{"cond": cond})
        sig_min = self.sig_min.to(x_1.device)
        d_psi = x_1 - (1 - sig_min) * x_0

        
        kld_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp()) / B

        
        k_basis = v_t.net.lie_basis_module.get_k_basis()
        p_basis = v_t.net.lie_basis_module.get_p_basis()
        loss_kk_ortho_p, loss_kp_ortho_k, loss_pp_ortho_p = calculate_cartan_commutation_losses(k_basis, p_basis)
        
        
        basis_norm_loss = v_t.net.calc_basis_norm_loss()
        loss_non_triviality = v_t.net.calc_non_triviality_loss()

        
        geodesic_loss = torch.sum(g_k_component**2) / B
        
        
        cfm_loss = torch.sum(torch.pow(v_psi - d_psi, 2)) / B

        
        total_loss = (
            cfm_loss +
            v_t.net.config.beta_kld * kld_loss +
            v_t.net.config.geodesic_loss_weight * geodesic_loss +
            v_t.net.config.comm_kkp_weight * loss_kk_ortho_p +
            v_t.net.config.comm_kpk_weight * loss_kp_ortho_k +
            v_t.net.config.comm_ppp_weight * loss_pp_ortho_p +
            v_t.net.config.basis_norm_loss_weight * basis_norm_loss +
            v_t.net.config.non_triviality_weight * loss_non_triviality
        )

        return {
            'total_loss': total_loss,
            'cfm_loss': cfm_loss,
            'kld_loss': kld_loss,
            'geodesic_loss': geodesic_loss,
            'loss_kk_ortho_p': loss_kk_ortho_p,
            'loss_kp_ortho_k': loss_kp_ortho_k,
            'loss_pp_ortho_p': loss_pp_ortho_p,
            'basis_norm_loss': basis_norm_loss,
            'loss_non_triviality': loss_non_triviality,
            'cycle_consistency_loss': torch.tensor(0.0, device=x_1.device)
        }



from .flow_matching import OTFlowMatchingCycleConsistency as OriginalFlowMatching
