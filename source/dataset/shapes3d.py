import os
import pdb
import numpy as np
import torch
from torchvision.transforms import ToTensor
import PIL

from source.seed import manual_seed
from source.dataset.dataset import Dataset

TRANSFORM = ToTensor()


"""3D Shapes dataset wrapper for image-directory based storage."""


class Shapes3d(Dataset):
    def __init__(self, path, shuffle=True, random_seed=42, split_ratio=0.0):
        """Load image paths and precomputed factor labels."""
        super(Shapes3d, self).__init__(path, shuffle, random_seed, split_ratio)

        IMGS = "imgs"
        self.img_dir = os.path.join(self.path, IMGS)
        self.data = [os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir)]
        self.latents_values = np.load(os.path.join(self.path, "labels.npy"))
        self.factor_num = self.latents_values.shape[-1]

    def __getitem__(self, index):
        """Read and normalize one RGB image and return paired factor labels."""
        x = PIL.Image.open(self.data[index])
        x = np.array(x) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.Tensor(x)

        classes = torch.Tensor(self.latents_values[index])
        return x, classes

    def __len__(self):
        return len(self.data)

    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        """Sample images for global-variance style disentanglement metrics."""
        manual_seed(self.random_seed)
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2**32)))
        idxs = g.choice(len(self.data), batch_size, replace=replace)

        resized_imgs = []
        for idx in idxs:
            resized_imgs.append(self.__getitem__(idx)[0])

        return torch.stack(resized_imgs, dim=0)

    def sampling_factors_and_img(self, batch_size, num_train):
        """Sample batches of factors and images for evaluation utilities."""
        dataset_size = len(self.data)
        idx = list(range(dataset_size))
        factors, imgs = [], []
        for _ in range(batch_size):
            np.random.shuffle(idx)
            factors_idx = idx[:batch_size]
            factors.append(torch.Tensor(self.latents_classes[factors_idx]))
            resized_img = list()
            for idx in factors_idx:
                resized_img.append(self.__getitem__(idx)[0])
            imgs.append(torch.stack(resized_img, dim=0))

        return torch.stack(factors, dim=0), torch.stack(imgs, dim=0)

    def idx_to_image(self, idx):
        """Convert linear index to image file path."""
        return self.data[idx]

    def idx_to_factors(self, idx):
        """Convert linear index to latent factor tuple."""
        return self.latents_classes[idx]

    def factor_to_idx(self, factor):
        """Map factor tuple to flattened index in canonical ordering."""
        base = np.array(
            [10 * 10 * 8 * 4 * 15, 10 * 8 * 4 * 15, 8 * 4 * 15, 4 * 15, 15, 1]
        )
        idx = np.dot(factor, base)
        return idx
