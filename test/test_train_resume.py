import unittest
import os
import tempfile
import torch
from hydra import initialize, compose
from hydra.utils import instantiate

import hypatorch

from shared import add_path

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        batch = {
            'image': torch.ones(1, 28, 28) * idx,
            'class': idx % 10           
        }   
        return batch

class TestTrainResume(unittest.TestCase):


    def setUp(self):
        self.training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        rel_config_dir = os.path.relpath(self.training_path,os.path.dirname(__file__))
        with initialize(
            config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
        ):
            self.cfg = compose(config_name="config.yaml", overrides=['experiment=mnist_linear'])

    
    def test_train_resume(self):
        with add_path(self.training_path):
            train_dataset = TestDataset(64)

            # Create a temp directory 
            with tempfile.TemporaryDirectory() as tmpdirname:
                model = instantiate(self.cfg.model)
                resumed_model = instantiate(self.cfg.model)
                resumed_model.load_state_dict(model.state_dict())

                trainer = hypatorch.Trainer(**self.cfg.trainer)
                trainer.max_epochs = 1
                trainer.train(model=model, train_dataset=train_dataset, loader_args=self.cfg.dataloader, checkpoint_path=tmpdirname)

                resumed_trainer = hypatorch.Trainer(**self.cfg.trainer)
                resumed_trainer.resume_training(model=resumed_model, chkpt_name='last.ckpt', train_dataset=train_dataset, loader_args=self.cfg.dataloader, checkpoint_path=tmpdirname)

                trainer.max_epochs = 2
                trainer.continue_training(train_dataset=train_dataset, loader_args=self.cfg.dataloader, checkpoint_path=tmpdirname)

                # Verify that the two models are the same
                for k,v in model.state_dict().items():
                    assert torch.allclose(v, resumed_model.state_dict()[k])

    def test_max_samples_stops_training(self):
        with add_path(self.training_path):
            train_dataset = TestDataset(64)

            trainer_kwargs = dict(self.cfg.trainer)
            trainer_kwargs["max_epochs"] = 5
            trainer_kwargs["max_samples"] = 16
            trainer_kwargs["save_last"] = False
            trainer = hypatorch.Trainer(**trainer_kwargs)
            model = instantiate(self.cfg.model)
            trainer.train(
                model=model,
                train_dataset=train_dataset,
                loader_args={"batch_size": 8},
            )

            assert trainer.train_samples == 16
            assert trainer.train_step == 2
            assert trainer.epoch_idx == 1

    def test_time_based_checkpointing_writes_periodic_checkpoints(self):
        with add_path(self.training_path):
            train_dataset = TestDataset(64)

            with tempfile.TemporaryDirectory() as tmpdirname:
                trainer_kwargs = dict(self.cfg.trainer)
                trainer_kwargs["checkpoint_interval_seconds"] = 0.0
                trainer = hypatorch.Trainer(**trainer_kwargs)
                model = instantiate(self.cfg.model)
                trainer.train(
                    model=model,
                    train_dataset=train_dataset,
                    loader_args={"batch_size": 8},
                    checkpoint_path=tmpdirname,
                )

                checkpoint_files = sorted(
                    filename for filename in os.listdir(tmpdirname) if filename.endswith(".ckpt")
                )
                assert "last.ckpt" in checkpoint_files
                assert len(checkpoint_files) > 1

    def test_multi_device_config_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            hypatorch.Trainer(max_epochs=1, devices=2)
