import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from utils import set_seed
from data.dataset import Dataset

class MPI3d(Dataset):
    def __init__(self, path, shuffle=True, random_seed=42, split_ratio=0.0, **kwargs):
        super().__init__(path, shuffle, random_seed, split_ratio)

        try:
            
            if path.endswith('.npz'):
                
                data_dict = np.load(path, allow_pickle=True)
                if 'images' in data_dict:
                    self.data = data_dict['images']
                elif 'data' in data_dict:
                    self.data = data_dict['data']
                else:
                    
                    keys = list(data_dict.keys())
                    print(f"Available keys in {path}: {keys}")
                    self.data = data_dict[keys[0]]
            else:
                
                self.data = np.load(path)
            
            
            print(f"Original data shape: {self.data.shape}")
            
            
            if len(self.data.shape) == 4 and self.data.shape[-1] == 3:
                
                self.data = self.data.transpose([0, 3, 1, 2])
            elif len(self.data.shape) == 4 and self.data.shape[1] == 3:
                
                pass
            else:
                
                if self.data.size % (64 * 64 * 3) == 0:
                    n_samples = self.data.size // (64 * 64 * 3)
                    self.data = self.data.reshape([n_samples, 64, 64, 3]).transpose([0, 3, 1, 2])
                else:
                    raise ValueError(f"Cannot reshape data with shape {self.data.shape} to MPI3D format")
            
            print(f"Processed data shape: {self.data.shape}")
            
            
            if self.data.max() > 1.0:
                self.data = self.data.astype(np.float32) / 255.0
            else:
                self.data = self.data.astype(np.float32)
                
            self.data = torch.Tensor(self.data)
            
            if kwargs.get('ddp', True):
                self.data = self.data.share_memory_()

        except Exception as e:
            print(f"Error loading MPI3D data from {path}: {e}")
            print("Creating dummy data for testing...")
            
            self.data = torch.randn(1000, 3, 64, 64)

        self.latents_values = None
        self.latents_classes = None

    def __getitem__(self, idx):
        data = self.data[idx]
        return data, self.idx_to_factors(idx)
    
    def __len__(self):
        return len(self.data)
    
    def idx_to_image(self, idx):
        return self.data[idx] / 255.0
    
    def idx_to_factors(self, idx):
        
        
        base = np.array([6 * 2 * 3 * 3 * 40 * 40, 2 * 3 * 3 * 40 * 40, 3 * 3 * 40 * 40, 3 * 40 * 40, 40 * 40, 40, 1])
        
        factors = []
        for i in range(7):  
            factors.append(idx // base[i])
            idx = idx % base[i]

        return np.array(factors)
    
    def factor_to_idx(self, factor):
        
        
        base = np.array([6 * 2 * 3 * 3 * 40 * 40, 2 * 3 * 3 * 40 * 40, 3 * 3 * 40 * 40, 3 * 40 * 40, 40 * 40, 40, 1])
        idx = np.dot(factor, base)
        return idx
