

__version__ = "0.1.0"
__author__ = "CartanFM Team"
__email__ = "your.email@example.com"


from . import data
from . import training
from . import utils
from . import models


from .utils import CartanFMConfig, get_argument_parser, create_config_from_args
from .utils import set_seed, create_output_directory, print_model_info
from .data import setup_dataset, get_dataset_info
from .training import setup_wandb, train_model, save_model, load_model, test_model

__all__ = [
    
    'CartanFMConfig', 'get_argument_parser', 'create_config_from_args',
    
    
    'set_seed', 'create_output_directory', 'print_model_info',
    
    
    'setup_dataset', 'get_dataset_info',
    
    
    'setup_wandb', 'train_model', 'save_model', 'load_model', 'test_model',
    
    
    'data', 'training', 'utils', 'models'
]

from .config import CartanFMConfig
from .model import CartanFM
from .components import (
    CNNEncoder_for_CartanVAE,
    CycleVectorFieldNet,
    AdaGN,
    ConvBlock,
    DownBlock,
    UpBlock,
    MiddleBlock
)
from .lie_algebra import (
    LieBasisModule_GeneralCartan,
    OTFlowMatchingCycleConsistency,
    calculate_cartan_commutation_losses
)
from .utils import set_seed, create_output_directory, print_model_info
from .datasets import setup_dataset, get_dataset_info
from .training import setup_wandb, train_model, save_model, load_model
from .testing import test_model

__version__ = "1.0.0"

__all__ = [
    'CartanFMConfig',
    'CartanFM',
    'CNNEncoder_for_CartanVAE',
    'CycleVectorFieldNet',
    'AdaGN',
    'ConvBlock',
    'DownBlock',
    'UpBlock',
    'MiddleBlock',
    'LieBasisModule_GeneralCartan',
    'OTFlowMatchingCycleConsistency',
    'calculate_cartan_commutation_losses',
    'set_seed',
    'create_output_directory',
    'print_model_info',
    'setup_dataset',
    'get_dataset_info',
    'setup_wandb',
    'train_model',
    'save_model',
    'load_model',
    'test_model'
]
