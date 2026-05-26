import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from utils import set_seed
from data.dataset import Dataset


class DSprites(Dataset):
    def __init__(self, path, shuffle=True, random_seed=42, split_ratio=0.0):
        super(DSprites, self).__init__(path, shuffle, random_seed, split_ratio)

        dataset_zip = np.load(path, allow_pickle=True, encoding="bytes")

        self.data = np.expand_dims(dataset_zip["imgs"], axis=1)
        self.latents_values = dataset_zip["latents_values"]
        self.latents_classes = dataset_zip["latents_classes"][:, 1:]
        self.factor_num = self.latents_values[:, 1:].shape[-1]

        assert self.factor_num == 5

    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        set_seed(self.random_seed)
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2**32)))
        idxs = g.choice(len(self.data), batch_size, replace=replace)
        return torch.Tensor(self.data[idxs])

    def sampling_factors_and_img(self, batch_size, num_train):
        dataset_size = len(self.data)
        idx = list(range(dataset_size))
        factors, imgs = [], []
        for _ in range(batch_size):
            np.random.shuffle(idx)
            factors_idx = idx[:batch_size]
            factors.append(torch.Tensor(self.latents_classes[factors_idx]))
            imgs.append(torch.Tensor(self.data[factors_idx]))

        return torch.stack(factors, dim=0), torch.stack(imgs, dim=0)

    def __getitem__(self, idx):
        data = torch.Tensor(self.data[idx])
        classes = torch.Tensor(self.latents_classes[idx])
        return data, classes

    def __len__(self):
        return len(self.data)

    def idx_to_image(self, idx):
        return self.data[idx]

    def idx_to_factors(self, idx):
        return self.latents_classes[idx]

    def factor_to_idx(self, factor):
        base = np.array([6 * 40 * 32 * 32, 40 * 32 * 32, 32 * 32, 32, 1])
        idx = np.dot(factor, base)
        return idx

    def dataset_sample_batch(self, num_samples, mode, replace=False):
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2**32)))
        idx = g.choice(len(self.data), num_samples, replace=replace)
        return self.dataset_batch_from_idx(idx, mode)

    def dataset_batch_from_idx(self, idx, mode):
        return default_collate([self.dataset_get(i, mode=mode) for i in idx])

    def dataset_get(self, idx, mode: str):
        try:
            idx = int(idx)
        except:
            raise TypeError("idx must be int")

        if mode == "numpy":
            return self.data[idx]
        elif mode == "tensor":
            return torch.Tensor(self.data[idx])
        elif mode == "dataset":
            return self.__getitem__(idx)
        else:
            raise ValueError("mode must be numpy, tensor, or dataset")
