import unittest
import os
import tempfile
import torch
from hydra import initialize, compose
from hydra.utils import instantiate

import hypatorch

from shared import add_path

class TestCheckpoint(unittest.TestCase):


    def setUp(self):
        self.training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        rel_config_dir = os.path.relpath(self.training_path,os.path.dirname(__file__))
        with initialize(
            config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
        ):
            self.cfg = compose(config_name="config.yaml", overrides=['experiment=mnist_linear'])

    def test_save_checkpoint(self):
        with add_path(self.training_path):
            model = instantiate(self.cfg.model)
            optimizer, scheduler, _ = model.configure_optimizers()

            # Create a temp directory 
            with tempfile.TemporaryDirectory() as tmpdirname:
                trainer = hypatorch.Trainer(**self.cfg.trainer)
                trainer.save_checkpoint('test.ckpt', model, optimizer, scheduler, chkpt_dir=tmpdirname)

                # Verify that the checkpoint was saved
                assert os.path.exists(os.path.join(tmpdirname, 'test.ckpt'))

                # Load the checkpoint via torch
                checkpoint = torch.load(os.path.join(tmpdirname, 'test.ckpt'))

                assert checkpoint.keys() == {'hypatorch_version', 'state_dict', 'optimizers', 'global_step', 'train_step', 'epoch_idx', 'val_step', 'train_samples', 'rng_state'}

                assert checkpoint['hypatorch_version'] == hypatorch.__version__
                assert checkpoint['global_step'] == 0
                assert checkpoint['train_step'] == 0
                assert checkpoint['val_step'] == 0
                assert checkpoint['epoch_idx'] == 0
                assert checkpoint['train_samples'] == 0
                assert checkpoint['state_dict'].keys() == model.state_dict().keys()
                assert checkpoint['optimizers'].keys() == optimizer.keys()
                assert checkpoint['rng_state'].keys() == trainer.get_rng_state_dict().keys()

    def test_save_checkpoint_respects_exclude_from_checkpoint(self):
        with add_path(self.training_path):
            self.cfg.model.exclude_from_checkpoint = ['image_encoder']
            model = instantiate(self.cfg.model)
            optimizer, scheduler, _ = model.configure_optimizers()

            with tempfile.TemporaryDirectory() as tmpdirname:
                trainer = hypatorch.Trainer(**self.cfg.trainer)
                trainer.save_checkpoint('test.ckpt', model, optimizer, scheduler, chkpt_dir=tmpdirname)

                checkpoint = torch.load(os.path.join(tmpdirname, 'test.ckpt'))
                assert all(not key.startswith('image_encoder.') for key in checkpoint['state_dict'].keys())
    
    def test_load_checkpoint(self):
        with add_path(self.training_path):
            model = instantiate(self.cfg.model)
            optimizer, scheduler, _ = model.configure_optimizers()

            batch = {
                'image': torch.randn(32, 1, 28, 28),
                'class': torch.randint(0, 10, (32,))                
            }   

            # Create a temp directory 
            with tempfile.TemporaryDirectory() as tmpdirname:
                trainer = hypatorch.Trainer(**self.cfg.trainer)

                trainer.step('train', model, batch, optimizer, scheduler, logger=None)

                trainer.save_checkpoint('test.ckpt', model, optimizer, scheduler, chkpt_dir=tmpdirname)

                trainer.step('train', model, batch, optimizer, scheduler, logger=None)

                # Create a new trainer and model
                restored_model = instantiate(self.cfg.model)
                restored_optimizer, restored_scheduler, _ = restored_model.configure_optimizers()
                restored_trainer = hypatorch.Trainer(**self.cfg.trainer)

                # Load the checkpoint
                restored_trainer.load_checkpoint('test.ckpt', restored_model, restored_optimizer, restored_scheduler, chkpt_dir=tmpdirname)

                restored_trainer.step('train', restored_model, batch, restored_optimizer, restored_scheduler, logger=None)

                # Verify the states of restored and original are the same

                assert trainer.global_step == restored_trainer.global_step
                assert trainer.train_step == restored_trainer.train_step
                assert trainer.val_step == restored_trainer.val_step
                assert trainer.epoch_idx == restored_trainer.epoch_idx

                for k in model.state_dict().keys():
                    assert torch.allclose(model.state_dict()[k], restored_model.state_dict()[k])

                for k in optimizer.keys():
                    for k2 in optimizer[k].state_dict()['state']:
                        for state in optimizer[k].state_dict()['state'][k2].keys():
                            assert torch.allclose(optimizer[k].state_dict()['state'][k2][state], restored_optimizer[k].state_dict()['state'][k2][state])
                    
