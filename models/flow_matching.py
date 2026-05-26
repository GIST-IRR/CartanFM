


import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .lie_algebra import calculate_cartan_commutation_losses


class OTFlowMatchingCycleConsistency:
    
    
    def __init__(self, sig_min: float = 1e-6, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.sig_min = sig_min * torch.ones(1, device=self.device)
        self.eps = 1e-6

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        sig_min = self.sig_min.to(x.device)
        return (1 - (1 - sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        cond = kwargs["cond"]
        mu_z = kwargs.get("mu_z", None)
        logvar_z = kwargs.get("logvar_z", None)
        coeffs_k = kwargs.get("coeffs_k", None)
        coeffs_p = kwargs.get("coeffs_p", None)
        g_k_component = kwargs.get("g_k_component", None)
        
        if mu_z is None or logvar_z is None:
            raise ValueError("mu_z and logvar_z must be provided in kwargs for loss calculation.")
        if coeffs_k is None or coeffs_p is None:
            raise ValueError("coeffs_k and coeffs_p must be provided in kwargs for loss calculation.")
        if g_k_component is None:
            raise ValueError("g_k_component must be provided in kwargs for loss calculation.")
        
        B = x_1.shape[0]
        t = torch.rand(B, device=x_1.device) * (1 - self.eps)  
        t_expand = t.view(B, 1, 1, 1)  
        x_0 = torch.randn_like(x_1).to(x_1.device)  
        
        
        v_psi = v_t(t, self.psi_t(x_0, x_1, t_expand), **{"cond": cond})
        sig_min = self.sig_min.to(x_1.device)
        d_psi = x_1 - (1 - sig_min) * x_0

        
        kld_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1).mean()

        
        k_basis = v_t.net.lie_basis_module.get_k_basis()
        p_basis = v_t.net.lie_basis_module.get_p_basis()
        loss_kk_ortho_p, loss_kp_ortho_k, loss_pp_ortho_p = calculate_cartan_commutation_losses(k_basis, p_basis)
        
        
        basis_norm_loss = v_t.net.calc_basis_norm_loss()
        loss_non_triviality = v_t.net.calc_non_triviality_loss()

        
        geodesic_loss = torch.sum(torch.sum(g_k_component**2, dim=[1, 2])) / B
        
        
        cfm_loss = torch.sum(torch.pow(v_psi - d_psi, 2)) / B

        
        coeffs_k_gs, coeffs_p_gs = -coeffs_k, -coeffs_p
        g_total_gs, g_k_component_gs = v_t.net.lie_basis_module(coeffs_k_gs, coeffs_p_gs)
        t_zero = torch.zeros(B, device=x_1.device)  
        x_gs_out = v_t.net(t_zero, x_0, **{"cond": g_total_gs.view(B, -1)})  
        sig_min = self.sig_min.to(x_1.device)
        x_gs = x_gs_out + (1 - sig_min) * x_0

        
        z_gs_cycle, mu_z_gs_cycle, logvar_z_gs_cycle, coeffs_k_gs_cycle, coeffs_p_gs_cycle = v_t.net.encoder(x_gs)

        
        cycle_consistency_loss = (
            torch.sum(torch.pow(coeffs_k - coeffs_k_gs_cycle, 2)) / B +
            torch.sum(torch.pow(coeffs_p - coeffs_p_gs_cycle, 2)) / B
        )

        
        total_loss = (
            cfm_loss +
            v_t.net.config.beta_kld * kld_loss +
            v_t.net.config.geodesic_loss_weight * geodesic_loss +
            v_t.net.config.comm_kkp_weight * loss_kk_ortho_p +
            v_t.net.config.comm_kpk_weight * loss_kp_ortho_k +
            v_t.net.config.comm_ppp_weight * loss_pp_ortho_p +
            v_t.net.config.basis_norm_loss_weight * basis_norm_loss +
            v_t.net.config.non_triviality_weight * loss_non_triviality +
            v_t.net.config.cycle_consistency_weight * cycle_consistency_loss
        )

        
        loss_components = {
            "total_loss": total_loss, "cfm_loss": cfm_loss, "kld_loss": kld_loss,
            "basis_norm_loss": basis_norm_loss, "comm_kk_ortho_p_loss": loss_kk_ortho_p, 
            "comm_kp_ortho_k_loss": loss_kp_ortho_k, "comm_pp_ortho_p_loss": loss_pp_ortho_p,
            "non_triviality_loss": loss_non_triviality,
            "geodesic_loss": geodesic_loss,
            "cycle_consistency_loss": cycle_consistency_loss,
        }
        return total_loss, loss_components


class BaselineFlowMatching:
    
    
    def __init__(self, sig_min: float = 1e-6, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.sig_min = sig_min * torch.ones(1, device=self.device)
        self.eps = 1e-6

    def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        sig_min = self.sig_min.to(x.device)
        return (1 - (1 - sig_min) * t) * x + t * x_1

    def loss(self, v_t: nn.Module, x_1: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        cond = kwargs["cond"]
        mu_z = kwargs.get("mu_z", None)
        logvar_z = kwargs.get("logvar_z", None)
        
        if mu_z is None or logvar_z is None:
            raise ValueError("mu_z and logvar_z must be provided in kwargs for loss calculation.")
        
        B = x_1.shape[0]
        t = torch.rand(B, device=x_1.device) * (1 - self.eps)
        t_expand = t.view(B, 1, 1, 1)
        x_0 = torch.randn_like(x_1).to(x_1.device)
        
        
        v_psi = v_t(t, self.psi_t(x_0, x_1, t_expand), **{"cond": cond})
        sig_min = self.sig_min.to(x_1.device)
        d_psi = x_1 - (1 - sig_min) * x_0

        
        kld_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1).mean()
        
        
        cfm_loss = torch.sum(torch.pow(v_psi - d_psi, 2)) / B

        
        total_loss = cfm_loss + v_t.net.config.beta_kld * kld_loss

        
        loss_components = {
            "total_loss": total_loss, 
            "cfm_loss": cfm_loss, 
            "kld_loss": kld_loss,
            "basis_norm_loss": torch.tensor(0.0, device=x_1.device),
            "comm_kk_ortho_p_loss": torch.tensor(0.0, device=x_1.device),
            "comm_kp_ortho_k_loss": torch.tensor(0.0, device=x_1.device),
            "comm_pp_ortho_p_loss": torch.tensor(0.0, device=x_1.device),
            "non_triviality_loss": torch.tensor(0.0, device=x_1.device),
            "geodesic_loss": torch.tensor(0.0, device=x_1.device),
            "cycle_consistency_loss": torch.tensor(0.0, device=x_1.device),
        }
        return total_loss, loss_components
