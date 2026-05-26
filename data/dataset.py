import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, path, shuffle=True, random_seed=42, split_ratio=0):
        self.path = path
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.split_ratio = split_ratio

        self.data, self.latents_values, self.latents_classes = None, None, None
        self.train_idxs, self.test_idxs = None, None

        self.factor_num = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)
