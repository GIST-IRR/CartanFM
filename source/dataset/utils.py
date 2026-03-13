import torch.utils.data as data


"""Legacy dataset abstractions used by disentanglement evaluation code."""


class DisenDataLoader(data.Dataset):
    def __init__(self, path, shuffle_dataset=True, random_seed=42, split_ratio=0.2):
        """Store shared dataset configuration for legacy loader implementations."""
        self.path = path
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = random_seed
        self.split_ratio = split_ratio

        self.data, self.latents_values, self.latents_classes = None, None, None
        self.train_idxs, self.test_idxs = None, None
        # self.data, self.latents, self.classes = None, None, None
        # self.train_sampler, self.test_sampler = None, None
        self.factor_num = None

    # def __call__(self, *args, **kwargs):
    #    raise NotImplementedError("Build call function")

    def __getitem__(self, item):
        """Return one sample by index."""
        raise NotImplementedError("Build getitem function")

    def dataset_sample_batch(self, num_samples: int, mode: str, replace: bool):
        """Return a sampled batch according to downstream evaluation mode."""
        raise NotImplementedError("Build dataset_sample_batch function")

    def __len__(self):
        """Return number of samples currently loaded."""
        return len(self.data)
