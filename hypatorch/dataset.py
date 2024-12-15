import torch
from .utils import get_rank, get_world_size

class DictedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_keys):
        self.dataset = dataset
        self.sample_keys = sample_keys

    def __getitem__(self, idx):
        data_tuple = self.dataset[idx]
        return {k: data_tuple[i] for i, k in enumerate(self.sample_keys)}

    def __len__(self):
        return len(self.dataset)

class DistributedDataset:

    def rank_split(self, rank, world_size):
        raise NotImplementedError("rank_split(rank, world_size) not implemented")

    def get_rank_dataset(self):
        world_size = get_world_size() 
        rank=get_rank()

        if world_size == 1:
            return self
        
        if rank >= world_size or rank < 0:
            raise ValueError("rank %d invalid for world size %d", rank, world_size)
    
        return self.rank_split(rank=rank, world_size=world_size)

