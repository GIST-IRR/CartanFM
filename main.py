

import os
import sys
import time
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")


from utils import CartanFMConfig, get_argument_parser, create_config_from_args
from utils import set_seed, create_output_directory, print_model_info
from data import setup_dataset, get_dataset_info
from training import setup_wandb, train_model, save_model, load_model
from training import test_model
from models.cartanfm_ablation import create_cartanfm_model


def main():
    
    parser = get_argument_parser()
    args = parser.parse_args()
    
    
    config = create_config_from_args(args)
    
    
    set_seed(config.seed)
    
    
    output_dir = create_output_directory(config)
    config.output_dir = output_dir
    
    
    train_dataloader, test_dataloader, num_factors, factor_sizes = setup_dataset(config)
    config.num_factors = num_factors
    config.factor_sizes = factor_sizes
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    
    
    model = create_cartanfm_model(config).to(device)
    
    
    print_model_info(model)
    
    
    if config.train and WANDB_AVAILABLE and config.wandb:
        setup_wandb(config)
    
    
    if config.train:
        print(f"Starting training for {config.epochs} epochs...")
        start_time = time.time()
        
        train_model(model, train_dataloader, config)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        
        model_path = save_model(model, config, output_dir)
        print(f"Model saved to: {model_path}")
    
    
    if config.test:
        
        if not config.train:
            if config.model_path:
                print(f"Loading model from: {config.model_path}")
                model = load_model(config.model_path, config, device)
            else:
                
                auto_model_path = os.path.join(output_dir, "cartanfm_final.pt")
                if os.path.exists(auto_model_path):
                    print(f"Loading model from auto-detected path: {auto_model_path}")
                    model = load_model(auto_model_path, config, device)
                else:
                    
                    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith("cartanfm_checkpoint_") and f.endswith(".pt")]
                    if checkpoint_files:
                        
                        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                        latest_checkpoint = checkpoint_files[-1]
                        auto_model_path = os.path.join(output_dir, latest_checkpoint)
                        print(f"Loading model from latest checkpoint: {auto_model_path}")
                        model = load_model(auto_model_path, config, device)
                    else:
                        print("Warning: No trained model found. Using randomly initialized model.")
                        print("Please specify --model_path or run training first.")
        
        print("Starting testing...")
        start_time = time.time()
        
        test_model(model, test_dataloader, config, output_dir)
        
        testing_time = time.time() - start_time
        print(f"Testing completed in {testing_time:.2f} seconds")
    
    
    if config.train and WANDB_AVAILABLE and config.wandb:
        wandb.finish()
    
    print("All tasks completed successfully!")


if __name__ == '__main__':
    main()
