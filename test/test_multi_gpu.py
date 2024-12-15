import unittest
import os
import tempfile
import torch
import torch.multiprocessing as mp

from hydra import initialize, compose
from hydra.utils import instantiate

import hypatorch

from shared import add_path, add_envs

class Dataset(torch.utils.data.Dataset, hypatorch.DistributedDataset):
    def __init__(self, size, offset=0):
        self.size = size
        self.offset = offset
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx >= self.size:
            raise StopIteration()

        sample_idx = self.offset + idx

        sample = {
            'image': torch.ones(1, 28, 28) * sample_idx,
            'class': sample_idx % 10           
        }   
        return sample

    def rank_split(self, rank, world_size):
        split_size = self.size//world_size
        return Dataset(split_size, offset=rank * split_size)


class TestMultiGPU(unittest.TestCase):

    def setUp(self):
        self.training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        rel_config_dir = os.path.relpath(self.training_path,os.path.dirname(__file__))
        with initialize(
            config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
        ):
            self.cfg = compose(config_name="config.yaml", overrides=['experiment=mnist_linear'])

    
    def test_distributed_dataset(self):
        dataset = Dataset(8)

        with add_envs(RANK=0, WORLD_SIZE=2):
            rank_0_dataset = dataset.get_rank_dataset()

            assert len(rank_0_dataset) == 4

        with add_envs(RANK=1, WORLD_SIZE=2):
            rank_1_dataset = dataset.get_rank_dataset()

            assert len(rank_1_dataset) == 4

        data_iter = iter(dataset)

        for ref_sample, split_sample in zip(iter(rank_0_dataset), data_iter):
            assert ref_sample['class'] == split_sample['class']

        for ref_sample, split_sample in zip(iter(rank_1_dataset), data_iter):
            assert ref_sample['class'] == split_sample['class']


