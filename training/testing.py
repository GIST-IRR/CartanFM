
import os
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from typing import Dict, List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
else:
    DataLoader = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def evaluate_batch(model: torch.nn.Module, test_images: torch.Tensor, 
                  config) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    
    batch_size = test_images.shape[0]
    
    
    
    results = model(test_images, return_all=True)
    
    
    noise = torch.randn_like(test_images)
    output = model.condvf.decode(noise, cond=results['cond'])
    
    
    total_loss, loss_components = model.training_step(test_images)
    
    
    if config.reconstruction_loss_type == 'mse':
        recon_loss = F.mse_loss(output, test_images, reduction='sum') / batch_size
    elif config.reconstruction_loss_type == 'bce':
        clamped_output = torch.clamp(output, min=1e-3, max=1-1e-3)
        recon_loss = F.binary_cross_entropy(clamped_output, test_images, reduction='sum') / batch_size
    else:
        raise ValueError(f"Unknown reconstruction_loss_type: {config.reconstruction_loss_type}")
    
    
    loss_components["recon_loss"] = recon_loss
    
    return output, loss_components, total_loss


def test_model(model: torch.nn.Module, test_dataloader, 
              config, output_dir: str) -> Dict[str, float]:
    
    print("Starting testing mode...")
    
    model.eval()
    device = next(model.parameters()).device
    
    
    all_loss_components = {
        "cfm_loss": 0.0, "kld_loss": 0.0, "geodesic_loss": 0.0,
        "basis_norm_loss": 0.0, "comm_kk_ortho_p_loss": 0.0,
        "comm_kp_ortho_k_loss": 0.0, "comm_pp_ortho_p_loss": 0.0,
        "non_triviality_loss": 0.0, "cycle_consistency_loss": 0.0,
        "total_loss": 0.0, "recon_loss": 0.0
    }
    
    
    all_original_images = []
    all_reconstructed_images = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Final Evaluation")):
            test_images = batch[0].to(device)
            
            
            output, loss_components, total_loss = evaluate_batch(model, test_images, config)
            
            
            for key, value in loss_components.items():
                if key in all_loss_components:
                    all_loss_components[key] += value.item()
                else:
                    
                    all_loss_components[key] = value.item()
                    print(f"Warning: New loss component found: {key}")
            
            
            all_original_images.append(test_images.cpu())
            all_reconstructed_images.append(output.cpu())
            
            
            if batch_idx < 5:
                save_batch_images(test_images.cpu(), output.cpu(), batch_idx, config, output_dir)
    
    
    num_batches = len(test_dataloader)
    for key in all_loss_components:
        all_loss_components[key] /= num_batches
    
    
    save_all_reconstruction_grids(all_original_images, all_reconstructed_images, config, output_dir)
    
    
    print_test_results(all_loss_components)
    save_test_results(all_loss_components, config, output_dir)
    
    return all_loss_components


def save_batch_images(original_batch: torch.Tensor, reconstructed_batch: torch.Tensor, 
                     batch_idx: int, config, output_dir: str) -> None:
    
    batch_size = original_batch.shape[0]
    nrow = min(8, batch_size)
    
    
    comparison_images = torch.zeros(batch_size * 2, *original_batch.shape[1:])
    for i in range(batch_size):
        comparison_images[i * 2] = original_batch[i]  
        comparison_images[i * 2 + 1] = reconstructed_batch[i]  
    
    
    grid = make_grid(comparison_images, nrow=nrow, padding=2, normalize=True, scale_each=True)
    
    batch_filename = f"reconstruction_batch_{batch_idx:03d}.png"
    batch_path = os.path.join(output_dir, batch_filename)
    
    save_image(grid, batch_path)
    print(f"Batch {batch_idx} reconstruction saved to {batch_path}")


def save_all_reconstruction_grids(all_original_images: List[torch.Tensor], 
                                 all_reconstructed_images: List[torch.Tensor], 
                                 config, output_dir: str) -> None:
    
    if not all_original_images or not all_reconstructed_images:
        return
    
    print(f"Saving all reconstruction images from {len(all_original_images)} batches...")
    
    
    all_original_tensor = torch.cat(all_original_images, dim=0)
    all_reconstructed_tensor = torch.cat(all_reconstructed_images, dim=0)
    
    total_images = all_original_tensor.shape[0]
    print(f"Total images: {total_images}")
    
    
    save_full_grid(all_original_tensor, all_reconstructed_tensor, config, total_images, output_dir)
    
    
    save_sampled_grid(all_original_tensor, all_reconstructed_tensor, config, total_images, output_dir)


def save_full_grid(all_original_tensor: torch.Tensor, all_reconstructed_tensor: torch.Tensor,
                  config, total_images: int, output_dir: str) -> None:
    
    
    max_display = min(256, total_images)
    original_display = all_original_tensor[:max_display]
    reconstructed_display = all_reconstructed_tensor[:max_display]
    
    
    comparison_images = torch.zeros(max_display * 2, *original_display.shape[1:])
    for i in range(max_display):
        comparison_images[i * 2] = original_display[i]  
        comparison_images[i * 2 + 1] = reconstructed_display[i]  
    
    
    nrow = 16  
    grid = make_grid(comparison_images, nrow=nrow, padding=2, normalize=True, scale_each=True)
    
    all_reconstruction_path = os.path.join(output_dir, "all_reconstructions.png")
    save_image(grid, all_reconstruction_path)
    print(f"All reconstructions grid saved to {all_reconstruction_path}")
    
    
    if config.wandb and WANDB_AVAILABLE:
        try:
            import wandb
            if wandb.run is not None:  
                wandb.log({"all_reconstructions": wandb.Image(all_reconstruction_path)})
            else:
                print("Warning: wandb.run is None, skipping wandb logging for all_reconstructions")
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")


def save_sampled_grid(all_original_tensor: torch.Tensor, all_reconstructed_tensor: torch.Tensor,
                     config, total_images: int, output_dir: str) -> None:
    
    sample_indices = torch.linspace(0, total_images-1, min(128, total_images)).long()
    sampled_original = all_original_tensor[sample_indices]
    sampled_reconstructed = all_reconstructed_tensor[sample_indices]
    
    sampled_comparison = torch.zeros(len(sample_indices) * 2, *sampled_original.shape[1:])
    for i in range(len(sample_indices)):
        sampled_comparison[i * 2] = sampled_original[i]
        sampled_comparison[i * 2 + 1] = sampled_reconstructed[i]
    
    sampled_grid = make_grid(sampled_comparison, nrow=8, padding=2, normalize=True, scale_each=True)
    
    split_type = "r2e" if config.r2e else "r2r"
    sampled_filename = f"reconstruction_sampled_{config.dataset}_{split_type}_{config.case}_{config.seed}.png"
    sampled_path = os.path.join(output_dir, sampled_filename)
    
    save_image(sampled_grid, sampled_path)
    print(f"Sampled reconstructions grid saved to {sampled_path}")
    
    
    if config.wandb and WANDB_AVAILABLE:
        try:
            import wandb
            if wandb.run is not None:  
                wandb.log({"sampled_reconstructions": wandb.Image(sampled_path)})
            else:
                print("Warning: wandb.run is None, skipping wandb logging for sampled_reconstructions")
        except Exception as e:
            print(f"Warning: Failed to log to wandb: {e}")


def print_test_results(all_loss_components: Dict[str, float]) -> None:
    
    print("=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    for key, value in all_loss_components.items():
        print(f"{key}: {value:.6f}")
    print("=" * 60)
    print(f"Main Reconstruction Loss: {all_loss_components['recon_loss']:.6f}")
    print("=" * 60)


def save_test_results(all_loss_components: Dict[str, float], config, output_dir: str) -> None:
    
    split_type = "r2e" if config.r2e else "r2r"
    result_filename = f"{config.dataset}_{split_type}_{config.case}_{config.seed}.yaml"
    
    result_data = {
        'metadata': {
            'dataset': config.dataset,
            'split_type': split_type,
            'case': config.case,
            'seed': config.seed,
            'artifact_dir': output_dir
        },
        'loss_components': {key: float(value) for key, value in all_loss_components.items()},
        'model_configuration': {
            'latent_dimension': int(config.latent_dim),
            'num_k_factors': int(config.num_k_factors),
            'num_p_factors': int(config.num_p_factors),
            'beta_kld': float(config.beta_kld),
            'geodesic_loss_weight': float(config.geodesic_loss_weight),
            'basis_norm_loss_weight': float(config.basis_norm_loss_weight),
            'comm_kkp_weight': float(config.comm_kkp_weight),
            'comm_kpk_weight': float(config.comm_kpk_weight),
            'comm_ppp_weight': float(config.comm_ppp_weight),
            'non_triviality_weight': float(config.non_triviality_weight),
            'cycle_consistency_weight': float(config.cycle_consistency_weight),
            'subgroup_sizes_ls': config.subgroup_sizes_ls,
            'subspace_sizes_ls': config.subspace_sizes_ls,
            'num_groups': int(config.num_groups)
        }
    }
    
    result_path = os.path.join(output_dir, result_filename)
    with open(result_path, 'w') as f:
        yaml.dump(result_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Results saved to {result_path}")
