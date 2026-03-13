from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from morphomnist import io


"""Dataset helpers for MorphoMNIST toy experiments."""


class MorphoMNISTDataset(Dataset):
    """Load MorphoMNIST images, labels, and perturbation annotations."""

    def __init__(self, train=True, concat=False, train_set=None, test_set=None):
        """Initialize from train/test files or by concatenating two sets."""
        super().__init__()

        if concat:
            assert train_set is not None and test_set is not None
            self.images = np.concatenate((train_set.images, test_set.images), axis=0)
            self.labels = np.concatenate((train_set.labels, test_set.labels), axis=0)
            self.pert = np.concatenate((train_set.pert, test_set.pert), axis=0)
        else:
            if train:
                self.images = io.load_idx("train-images-idx3-ubyte.gz") / 255.0
                self.labels = io.load_idx("train-labels-idx1-ubyte.gz")
                self.pert = io.load_idx("train-pert-idx1-ubyte.gz")
            else:
                self.images = io.load_idx("t10k-images-idx3-ubyte.gz") / 255.0
                self.labels = io.load_idx("t10k-labels-idx1-ubyte.gz")
                self.pert = io.load_idx("t10k-pert-idx1-ubyte.gz")

    def __len__(self):
        """Return total number of examples."""
        return len(self.images)

    def __getitem__(self, index):
        """Return one `(image, digit_label, perturbation_label)` tuple."""
        return (
            torch.tensor(self.images[index], dtype=torch.float32), 
            torch.tensor(self.labels[index], dtype=torch.float32), 
            torch.tensor(self.pert[index], dtype=torch.float32)
        )


def r2r(dataset: MorphoMNISTDataset):
    """Create random-to-random split by holding out a label/perturbation slice."""
    drop_list = list()
    for idx in range(len(dataset)):
        _, label, pert = dataset.__getitem__(idx)
        if label == 0 and pert == 2:
            drop_list.append(idx)
    
    train_set = deepcopy(dataset)
    test_set = deepcopy(dataset)

    train_set.images = np.delete(train_set.images, drop_list, axis=0)
    train_set.labels = np.delete(train_set.labels, drop_list, axis=0)
    train_set.pert = np.delete(train_set.pert, drop_list, axis=0)

    mask = np.zeros(len(dataset), dtype=bool)
    mask[drop_list] = True

    test_set.images = test_set.images[mask]
    test_set.labels = test_set.labels[mask]
    test_set.pert = test_set.pert[mask]

    return train_set, test_set

