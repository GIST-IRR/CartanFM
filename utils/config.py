
import argparse
from typing import Optional, List
import torch


class CartanFMConfig:
    
    
    def __init__(self, args: Optional[argparse.Namespace] = None):

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.dataset = "dsprites"  
        self.data_dir = None
        self.r2e = False
        self.r2r = False
        self.case = 0
        self.split = 0.2
        self.shuffle = False
        self.train_batch_size = 128
        self.test_batch_size = 64
        self.num_workers = 4
        
        
        self.in_channel = 1  
        self.base_channels = 64
        self.n_freqs = 4  
        self.num_groups = 8  
        
        
        self.subgroup_sizes_ls = [100]  
        self.subspace_sizes_ls = [10]   
        self.latent_dim = sum(self.subspace_sizes_ls)
        
        
        self.cnn_feature_dim = 1024  
        self.encoder_fc_dim1 = 256   
        self.encoder_fc_dim2 = 128   
        
        
        self.num_k_factors = 5       
        self.num_p_factors = 10      
        self.lie_algebra_matrix_dim = self.latent_dim  
        
        
        self.beta_kld = 0.01                     
        self.geodesic_loss_weight = 0.1          
        self.basis_norm_loss_weight = 0.0        
        self.comm_kkp_weight = 1.0               
        self.comm_kpk_weight = 1.0               
        self.comm_ppp_weight = 1.0               
        self.non_triviality_weight = 0.001       
        self.cycle_consistency_weight = 1.0      
        
        
        self.learning_rate = 5e-4    
        self.epochs = 100            
        self.reconstruction_loss_type = 'bce'  
        
        
        self.ode_solver = 'dopri5'  
        self.sig_min = 1e-6         
        
        
        self.output_dir = './cartan_pilot_output/fm/'
        
        
        self.model_type = 'original'  
        
        
        self.seed = 42
        
        
        self.project_name = 'cartan_pilot'
        self.logging_steps = 10
        self.wandb = False
        
        
        self.train = False
        self.test = False
        self.model_path = None

        self.resume = False

        
        if args:
            self._update_from_args(args)
        
        
        if self.dataset == "dsprites":
            self.input_channels = 1
        elif self.dataset in ["3dshapes", "mpi3d"]:
            self.input_channels = 3
        else:
            self.input_channels = 3  
        self.in_channel = self.input_channels  
        
        
        if args and hasattr(args, 'subgroup_sizes') and args.subgroup_sizes:
            self.subgroup_sizes_ls = args.subgroup_sizes
        if args and hasattr(args, 'subspace_sizes') and args.subspace_sizes:
            self.subspace_sizes_ls = args.subspace_sizes
            self.latent_dim = sum(self.subspace_sizes_ls)
    
    def _update_from_args(self, args: argparse.Namespace) -> None:
        
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> None:
        
        if not self.train and not self.test:
            raise ValueError("At least one of --train or --test must be specified")
        
        if self.data_dir is None:
            raise ValueError("--data_dir must be specified")
        
        if len(self.subgroup_sizes_ls) != len(self.subspace_sizes_ls):
            raise ValueError("subgroup_sizes_ls and subspace_sizes_ls must have the same length")
        
        if self.dataset not in ['dsprites', '3dshapes', 'mpi3d']:
            raise ValueError("dataset must be 'dsprites', '3dshapes', or 'mpi3d'")
        
        if self.reconstruction_loss_type not in ['bce', 'mse']:
            raise ValueError("reconstruction_loss_type must be 'bce' or 'mse'")
    
    def to_dict(self) -> dict:
        
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def __repr__(self) -> str:
        
        config_dict = self.to_dict()
        return f"CartanFMConfig({config_dict})"


def create_config_from_args(args: argparse.Namespace) -> CartanFMConfig:
    
    config = CartanFMConfig(args)
    config.validate()
    return config


def get_argument_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description='CartanFM configuration')
    
    
    parser.add_argument('--dataset', type=str, default='dsprites', 
                        choices=['dsprites', '3dshapes', 'mpi3d'], 
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing the dataset')
    parser.add_argument('--r2e', action='store_true', help='Use r2e dataset split')
    parser.add_argument('--r2r', action='store_true', help='Use r2r dataset split')
    parser.add_argument('--case', type=int, default=0, choices=[0, 1, 2],
                        help='Dataset case number')
    
    
    parser.add_argument('--subgroup_sizes', type=int, nargs='+', default=[100], 
                        help='Input dimensions for the to_means/logvar heads')
    parser.add_argument('--subspace_sizes', type=int, nargs='+', default=[10], 
                        help='Output dimensions of each to_means/logvar head')
    parser.add_argument('--num_k_factors', type=int, default=5, 
                        help='Number of k factors in Lie algebra')
    parser.add_argument('--num_p_factors', type=int, default=10, 
                        help='Number of p factors in Lie algebra')
    parser.add_argument('--num_groups', type=int, default=8, 
                        help='Number of groups for AdaGN normalization')
    parser.add_argument('--n_freqs', type=int, default=4, 
                        help='Number of frequency bands for positional encoding')
    
    
    parser.add_argument('--cnn_feature_dim', type=int, default=1024, 
                        help='Output dimension of CNN backbone after flattening')
    parser.add_argument('--encoder_fc_dim1', type=int, default=256, 
                        help='Output dimension of first dense layer in encoder')
    parser.add_argument('--encoder_fc_dim2', type=int, default=128, 
                        help='Output dimension of second dense layer in encoder')
    
    
    parser.add_argument('--beta_kld', type=float, default=0.01, 
                        help='Weight for KL divergence loss (increased for VAE stability)')
    parser.add_argument('--geodesic_loss_weight', type=float, default=0.1, 
                        help='Weight for geodesic component loss')
    parser.add_argument('--basis_norm_loss_weight', type=float, default=0.0, 
                        help='Weight for basis norm loss')
    parser.add_argument('--comm_kkp_weight', type=float, default=1, 
                        help='Weight for [k,k] vs p commutation loss')
    parser.add_argument('--comm_kpk_weight', type=float, default=1, 
                        help='Weight for [k,p] vs k commutation loss')
    parser.add_argument('--comm_ppp_weight', type=float, default=1, 
                        help='Weight for [p,p] vs p commutation loss')
    parser.add_argument('--non_triviality_weight', type=float, default=0.001, 
                        help='Weight for non-triviality loss')
    parser.add_argument('--cycle_consistency_weight', type=float, default=1.0, 
                        help='Weight for cycle consistency loss')
    
    
    parser.add_argument('--learning_rate', type=float, default=5e-4, 
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--reconstruction_loss_type', type=str, default='bce', 
                        choices=['bce', 'mse'], help='Type of reconstruction loss')
    
    
    parser.add_argument('--split', type=float, default=0.2, 
                        help='Split ratio for train and test set')
    parser.add_argument('--shuffle', action='store_true', 
                        help='Whether to shuffle dataset')
    parser.add_argument('--train_batch_size', type=int, default=128, 
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=128, 
                        help='Test batch size')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='./cartan_pilot_output/fm/', 
                        help='Directory to save outputs')
    
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    
    parser.add_argument('--ode_solver', type=str, default='dopri5', 
                        choices=['euler', 'midpoint', 'rk4', 'explicit_adams', 
                                'fixed_adams', 'dopri8', 'dopri5', 'bosh3', 
                                'adaptive_heun', 'tsit5'],
                        help='ODE solver method')
    
    
    parser.add_argument('--project_name', type=str, default='cartan_pilot', 
                        help='Project name for wandb')
    parser.add_argument('--logging_steps', type=int, default=10, 
                        help='Logging frequency for wandb')
    parser.add_argument('--wandb', action='store_true', 
                        help='Whether to use wandb')
    
    
    parser.add_argument('--train', action='store_true', 
                        help='Enable training mode')
    parser.add_argument('--test', action='store_true', 
                        help='Enable testing mode')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load pre-trained model for testing')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    
    
    parser.add_argument('--model_type', type=str, default='original',
                        choices=['original', 'nocycle', 'nolie', 'baseline'],
                        help='Model type for ablation study: '
                             'original (Lie + Cycle), nocycle (Lie only), '
                             'nolie (Cycle only), baseline (neither)')
    
    return parser
