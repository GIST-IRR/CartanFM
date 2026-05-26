

import torch
import torch.nn as nn
from typing import Optional, Tuple


def frobenius_inner_product(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    
    return torch.sum(A * B)


def commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    
    return A @ B - B @ A


def calculate_cartan_commutation_losses(
    k_basis: Optional[torch.Tensor], 
    p_basis: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    device = 'cpu'
    if k_basis is not None and k_basis.numel() > 0:
        device = k_basis.device
    elif p_basis is not None and p_basis.numel() > 0:
        device = p_basis.device

    loss_kk_ortho_p = torch.tensor(0.0, device=device)
    loss_kp_ortho_k = torch.tensor(0.0, device=device)
    loss_pp_ortho_p = torch.tensor(0.0, device=device)
    
    num_terms_kkp, num_terms_kpk, num_terms_ppp = 0, 0, 0

    
    if k_basis is not None and p_basis is not None and k_basis.shape[0] >= 2 and p_basis.shape[0] >= 1:
        for i in range(k_basis.shape[0]):
            for j in range(i + 1, k_basis.shape[0]):
                comm_kk = commutator(k_basis[i], k_basis[j])
                for l in range(p_basis.shape[0]):
                    loss_kk_ortho_p += frobenius_inner_product(comm_kk, p_basis[l]) ** 2
                    num_terms_kkp += 1
    
    
    if k_basis is not None and p_basis is not None and k_basis.shape[0] >= 1 and p_basis.shape[0] >= 1:
        for i in range(k_basis.shape[0]):
            for j in range(p_basis.shape[0]):
                comm_kp = commutator(k_basis[i], p_basis[j])
                for l in range(k_basis.shape[0]):
                    loss_kp_ortho_k += frobenius_inner_product(comm_kp, k_basis[l]) ** 2
                    num_terms_kpk += 1

    
    if p_basis is not None and p_basis.shape[0] >= 2:
        for i in range(p_basis.shape[0]):
            for j in range(i + 1, p_basis.shape[0]):
                comm_pp = commutator(p_basis[i], p_basis[j])
                for l in range(p_basis.shape[0]):
                    loss_pp_ortho_p += frobenius_inner_product(comm_pp, p_basis[l]) ** 2
                    num_terms_ppp += 1
    
    
    if num_terms_kkp > 0:
        loss_kk_ortho_p /= num_terms_kkp
    if num_terms_kpk > 0:
        loss_kp_ortho_k /= num_terms_kpk
    if num_terms_ppp > 0:
        loss_pp_ortho_p /= num_terms_ppp
        
    return loss_kk_ortho_p, loss_kp_ortho_k, loss_pp_ortho_p


class LieBasisModule_GeneralCartan(nn.Module):
    
    
    def __init__(self, matrix_dim: int, num_k_factors: int, num_p_factors: int):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.num_k_factors = num_k_factors
        self.num_p_factors = num_p_factors

        
        if self.num_k_factors > 0:
            self.raw_k_basis_params = nn.Parameter(
                torch.randn(num_k_factors, matrix_dim, matrix_dim) * 0.1
            )
        else:
            self.raw_k_basis_params = None

        
        if self.num_p_factors > 0:
            self.raw_p_basis_params = nn.Parameter(
                torch.randn(num_p_factors, matrix_dim, matrix_dim) * 0.1
            )
        else:
            self.raw_p_basis_params = None

    def get_k_basis(self) -> Optional[torch.Tensor]:
        return self.raw_k_basis_params

    def get_p_basis(self) -> Optional[torch.Tensor]:
        return self.raw_p_basis_params

    def forward(self, coeffs_k: Optional[torch.Tensor], coeffs_p: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, device = 1, 'cpu'
        if coeffs_k is not None and coeffs_k.numel() > 0:
            B, device = coeffs_k.shape[0], coeffs_k.device
        elif coeffs_p is not None and coeffs_p.numel() > 0:
            B, device = coeffs_p.shape[0], coeffs_p.device
        elif self.raw_k_basis_params is not None:
            device = self.raw_k_basis_params.device
        elif self.raw_p_basis_params is not None:
            device = self.raw_p_basis_params.device

        
        g_k_component = torch.zeros(B, self.matrix_dim, self.matrix_dim, device=device)
        if self.num_k_factors > 0 and coeffs_k is not None:
            for i in range(self.num_k_factors):
                g_k_component += coeffs_k[:, i:i+1, None] * self.raw_k_basis_params[i].unsqueeze(0)

        
        g_p_component = torch.zeros(B, self.matrix_dim, self.matrix_dim, device=device)
        if self.num_p_factors > 0 and coeffs_p is not None:
            for i in range(self.num_p_factors):
                g_p_component += coeffs_p[:, i:i+1, None] * self.raw_p_basis_params[i].unsqueeze(0)
            
        g_total = g_k_component + g_p_component
        return g_total, g_k_component
