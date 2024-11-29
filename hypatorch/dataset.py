import torch

class DictedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_keys):
        self.dataset = dataset
        self.sample_keys = sample_keys

    def __getitem__(self, idx):
        data_tuple = self.dataset[idx]
        return {k: data_tuple[i] for i, k in enumerate(self.sample_keys)}

    def __len__(self):
        return len(self.dataset)