import torch.utils.data as data


"""Base dataset abstraction shared by benchmark-specific loaders."""


class Dataset(data.Dataset):
    def __init__(self, path, shuffle=True, random_seed=42, split_ratio=0):
        """Store common dataset attributes.

        Args:
            path: Root path or file path to the dataset source.
            shuffle: Whether downstream samplers may shuffle examples.
            random_seed: Seed for deterministic sampling helpers.
            split_ratio: Optional train/test split ratio.
        """
        self.path = path
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.split_ratio = split_ratio

        self.data, self.latents_values, self.latents_classes = None, None, None
        self.train_idxs, self.test_idxs = None, None

        self.factor_num = None

    def __getitem__(self, index):
        """Return one sample and its factor labels."""
        raise NotImplementedError

    def __len__(self):
        """Return dataset size."""
        return len(self.data)
