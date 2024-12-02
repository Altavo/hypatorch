import unittest
import os
import sys
import torch
from hydra import initialize, compose
from hydra.utils import instantiate

import hypatorch
from hypatorch.utils import shared_dict

from shared import add_path

class TestOptimizer(unittest.TestCase):


    def setUp(self):
        self.training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        rel_config_dir = os.path.relpath(self.training_path,os.path.dirname(__file__))
        with initialize(
            config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
        ):
            self.cfg = compose(config_name="config.yaml", overrides=['+experiment=mnist_lenet'])

    def test_configure_optimizers(self):
        with add_path(self.training_path):
            model = instantiate(self.cfg.model)
            model.train()   

            optimizer, scheduler, gradient_clipping = model.configure_optimizers()

            assert(optimizer.keys() == {'update_encoder'})
            assert(len(scheduler.keys()) == 0)
            assert(gradient_clipping.keys() == {'update_encoder'})

    def test_gradient_clipping(self):
        with add_path(self.training_path):
            model = instantiate(self.cfg.model)
            model.operations.update_encoder.optimizer.lr = 1.0

            model.train()   

            optimizer, scheduler, gradient_clipping = model.configure_optimizers()

            batch = {
                'image': torch.randn(32, 1, 28, 28),
                'class': torch.randint(0, 10, (32,))                
            }   

            optimizer['update_encoder'].zero_grad()

            for i in range(100):
                output_dict = model(batch, 'update_encoder', 'train')         
                loss = model.compute_loss(shared_dict(batch, output_dict), 'update_encoder', 'train')
                loss.backward()

            # Verify that the gradients are greater than the clipping value
            for param in model.image_encoder.parameters():
                assert param.grad.max() > self.cfg.model.operations.update_encoder.gradient_clipping.clip_value

            gradient_clipping['update_encoder']()

            # Verify that the gradients have been clipped
            for param in model.image_encoder.parameters():
                assert param.grad.max() <= self.cfg.model.operations.update_encoder.gradient_clipping.clip_value