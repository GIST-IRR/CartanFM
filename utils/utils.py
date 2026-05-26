

import os
import random
import logging
import numpy as np
import torch
from typing import Tuple


def set_seed(seed: int, log_level: int = logging.INFO) -> None:

    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    os.environ['PYTHONHASHSEED'] = str(seed)  
    logging.log(log_level, f"Random seed set to {seed}")


def create_output_directory(config) -> str:

    
    model_type_dirs = {
        'original': 'original_lie_cycle',
        'nocycle': 'nocycle_lie_only', 
        'nolie': 'nolie_cycle_only',
        'baseline': 'baseline_neither'
    }
    
    model_subdir = model_type_dirs.get(config.model_type, config.model_type)
    
    
    split_type = "r2e" if config.r2e else "r2r"
    
    
    experiment_name = f"{config.dataset}_{split_type}_case{config.case}_seed{config.seed}"
    
    
    output_dir = os.path.join(config.output_dir, model_subdir, experiment_name)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, loss: float, filepath: str) -> None:

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                         filepath: str, device: torch.device) -> Tuple[int, float]:

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def count_parameters(model: torch.nn.Module) -> int:

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:

    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def calculate_model_size(model: torch.nn.Module) -> str:

    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return f"{size_all_mb:.2f}MB"


def print_model_info(model: torch.nn.Module) -> None:

    num_params = count_parameters(model)
    model_size = calculate_model_size(model)
    
    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: {model_size}")
    print("=" * 50)
