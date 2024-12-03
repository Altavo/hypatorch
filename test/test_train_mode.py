import unittest
import os
import sys
import torch
from hydra import initialize, compose
from hydra.utils import instantiate

import hypatorch

from shared import add_path

class TestTrainMode(unittest.TestCase):


    def setUp(self):
        self.training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        rel_config_dir = os.path.relpath(self.training_path,os.path.dirname(__file__))
        with initialize(
            config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
        ):
            self.cfg = compose(config_name="config.yaml", overrides=['+experiment=mnist_linear'])

    def test_train(self):
        with add_path(self.training_path):
            model = instantiate(self.cfg.model)
            model.train()   

            assert(model.image_encoder.training)

    def test_eval(self):
        with add_path(self.training_path):
            model = instantiate(self.cfg.model)
            model.eval()

            assert(not model.image_encoder.training)

    def test_submodule_eval(self):
        with add_path(self.training_path):
            # Modify the config to not calculate gradients
            self.cfg.model.submodules_eval = ['image_encoder']

            model = instantiate(self.cfg.model)
            model.train()

            assert(not model.image_encoder.training)