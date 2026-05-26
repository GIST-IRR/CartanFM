
import os
import time
import torch
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, Union

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def setup_wandb(config) -> None:
    
    if not config.wandb or not WANDB_AVAILABLE or wandb is None:
        return
    
    split_type = "r2e" if config.r2e else "r2r"
    
    wandb.init(
        project=config.project_name,
        config={
            "dataset": config.dataset,
            "split_type": split_type,
            "case": config.case,
            "seed": config.seed,
            "latent_dim": config.latent_dim,
            "num_k_factors": config.num_k_factors,
            "num_p_factors": config.num_p_factors,
            "subgroup_sizes": config.subgroup_sizes_ls,
            "subspace_sizes": config.subspace_sizes_ls,
            "num_groups": config.num_groups,
            "n_freqs": config.n_freqs,
            "beta_kld": config.beta_kld,
            "geodesic_loss_weight": config.geodesic_loss_weight,
            "basis_norm_loss_weight": config.basis_norm_loss_weight,
            "comm_kkp_weight": config.comm_kkp_weight,
            "comm_kpk_weight": config.comm_kpk_weight,
            "comm_ppp_weight": config.comm_ppp_weight,
            "non_triviality_weight": config.non_triviality_weight,
            "cycle_consistency_weight": config.cycle_consistency_weight,
            "ode_solver": config.ode_solver,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.train_batch_size
        }
    )


def train_epoch(model: torch.nn.Module, train_dataloader: DataLoader, 
               optimizer: Any, device: torch.device) -> Dict[str, float]:
    
    model.train()
    epoch_losses = {
        "total_loss": 0.0, "cfm_loss": 0.0, "kld_loss": 0.0,
        "basis_norm_loss": 0.0, "comm_kk_ortho_p_loss": 0.0, 
        "comm_kp_ortho_k_loss": 0.0, "comm_pp_ortho_p_loss": 0.0,
        "non_triviality_loss": 0.0, "geodesic_loss": 0.0,
        "cycle_consistency_loss": 0.0
    }
    
    for batch in tqdm(train_dataloader, desc="Training Batches", leave=False):
        input_images = batch[0].to(device)
        
        
        optimizer.zero_grad()
        
        
        total_loss, batch_loss_components = model.training_step(input_images)
        
        
        total_loss.backward()
        optimizer.step()
        
        
        for key in epoch_losses:
            if key in batch_loss_components:
                epoch_losses[key] += batch_loss_components[key].item()
    
    
    num_batches = len(train_dataloader)
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def train_model(model: torch.nn.Module, train_dataloader: DataLoader, 
               config) -> None:
    
    print("Starting training mode...")
    
    start_epoch = 0
    
    
    if config.resume:
        
        checkpoint_files = [f for f in os.listdir(config.output_dir) 
                          if f.startswith("cartanfm_checkpoint_") and f.endswith(".pt")]
        
        if checkpoint_files:
            
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(config.output_dir, latest_checkpoint)
            
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            model, optimizer, start_epoch = load_checkpoint(checkpoint_path, config, config.device)
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found. Starting training from scratch.")
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in tqdm(range(start_epoch, config.epochs), desc="Training Epochs"):
        epoch_losses = train_epoch(model, train_dataloader, optimizer, config.device)
        
        
        if config.wandb and WANDB_AVAILABLE and wandb is not None:
            wandb_log = {f"train/{key}": value for key, value in epoch_losses.items()}
            wandb_log["train/epoch"] = epoch
            wandb.log(wandb_log)
        
        
        print(f"Epoch {epoch + 1}/{config.epochs} - "
              f"Total Loss: {epoch_losses['total_loss']:.4f}, "
              f"CFM Loss: {epoch_losses['cfm_loss']:.4f}, "
              f"KLD Loss: {epoch_losses['kld_loss']:.4f}, "
              f"Geodesic Loss: {epoch_losses['geodesic_loss']:.4f}")
        
        
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, config)
def save_checkpoint(model: torch.nn.Module, optimizer: Any, 
                   epoch: int, config) -> None:
    
    checkpoint_path = os.path.join(config.output_dir, f"cartanfm_checkpoint_{epoch}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'config': config.to_dict()
    }, checkpoint_path)


def save_model(model: torch.nn.Module, config, output_dir: str, 
              filename: str = "cartanfm_final.pt") -> str:
    
    model_path = os.path.join(output_dir, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict()
    }, model_path)
    return model_path


def load_model(model_path: str, config, device: torch.device):
    
    from models.cartanfm_ablation import create_cartanfm_model
    
    checkpoint = torch.load(model_path, map_location=device)
    
    
    model = create_cartanfm_model(config).to(device)
    
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def load_checkpoint(model_path: str, config, device: torch.device):
    
    from models.cartanfm_ablation import create_cartanfm_model
    
    checkpoint = torch.load(model_path, map_location=device)
    
    
    model = create_cartanfm_model(config).to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  
    
    return model, optimizer, start_epoch
