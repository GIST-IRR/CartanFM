
import os
import torch
from typing import Tuple


from data.r2r import r2r_dsprites, r2r_shape3d, r2r_mpi3d
from data.r2e import r2e_dsprites, r2e_shape3d, r2e_mpi3d
from data.dsprites import DSprites
from data.shapes3d import Shapes3d
from data.mpi3d import MPI3d


def setup_dataset(config) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, list]:

    if config.dataset == "dsprites":
        train_dataset, test_dataset = setup_dsprites_dataset(config)
        num_factors = 5  
        factor_sizes = [3, 6, 40, 32, 32]  
    elif config.dataset == "3dshapes":
        train_dataset, test_dataset = setup_3dshapes_dataset(config)
        num_factors = 6  
        factor_sizes = [10, 10, 10, 8, 4, 15]  
    elif config.dataset == "mpi3d":
        train_dataset, test_dataset = setup_mpi3d_dataset(config)
        num_factors = 7  
        factor_sizes = [6, 6, 2, 3, 3, 40, 40]  
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.test_batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    return train_dataloader, test_dataloader, num_factors, factor_sizes


def setup_dsprites_dataset(config) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    
    data_path = os.path.join(config.data_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"DSprites dataset not found at {data_path}")
    
    dataset = DSprites(data_path, shuffle=False, split_ratio=0.0)
    
    if config.r2e:
        train_dataset, test_dataset = r2e_dsprites(config.case, dataset)
    else:
        train_dataset, test_dataset = r2r_dsprites(config.case, dataset)
    
    return train_dataset, test_dataset


def setup_3dshapes_dataset(config) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    
    data_path = os.path.join(config.data_dir, "3D_shapes_fast")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"3D Shapes dataset not found at {data_path}")
    
    dataset = Shapes3d(data_path, shuffle=False, split_ratio=0.0)
    
    if config.r2e:
        train_dataset, test_dataset = r2e_shape3d(config.case, dataset)
    else:
        train_dataset, test_dataset = r2r_shape3d(config.case, dataset)
    
    return train_dataset, test_dataset


def setup_mpi3d_dataset(config) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    
    data_path = os.path.join(config.data_dir, "mpi3d_toy.npz")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"MPI3D dataset not found at {data_path}")
    
    dataset = MPI3d(data_path, shuffle=False, split_ratio=0.0)
    
    if config.r2e:
        train_dataset, test_dataset = r2e_mpi3d(config.case, dataset)
    else:
        train_dataset, test_dataset = r2r_mpi3d(config.case, dataset)
    
    return train_dataset, test_dataset


def get_dataset_info(config) -> dict:

    info = {
        "dataset": config.dataset,
        "split_type": "r2e" if config.r2e else "r2r",
        "case": config.case
    }
    
    if config.dataset == "dsprites":
        info.update({
            "image_size": (64, 64),
            "channels": 1,
            "num_factors": 5,  
            "total_samples": 737280
        })
    elif config.dataset == "3dshapes":
        info.update({
            "image_size": (64, 64),
            "channels": 3,
            "num_factors": 6,  
            "total_samples": 480000
        })
    elif config.dataset == "mpi3d":
        info.update({
            "image_size": (64, 64),
            "channels": 3,
            "num_factors": 7,  
            "total_samples": 1728000  
        })
    
    return info
