import unittest
import os
import sys
import torch
from hydra import initialize, compose
from hydra.utils import instantiate

import hypatorch

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

class TestComputeGrad(unittest.TestCase):


    def setUp(self):
        self.training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        rel_config_dir = os.path.relpath(self.training_path,os.path.dirname(__file__))
        with initialize(
            config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
        ):
            self.cfg = compose(config_name="config.yaml", overrides=['+experiment=mnist_lenet'])

    def test_compute_grad(self):
        with add_path(self.training_path):
            model = instantiate(self.cfg.model)

            batch = {
                'image': torch.randn(32, 1, 28, 28),
                'label': torch.randint(0, 10, (32,))                
            }

            train_output = model(batch, 'update_encoder', 'train')

            assert(train_output.keys() == {'logits'})
            assert(train_output['logits'].requires_grad)

            val_output = model(batch, 'update_encoder', 'val')

            assert(val_output.keys() == {'logits'})
            assert(not val_output['logits'].requires_grad)

            test_output = model(batch, 'update_encoder', 'test')

            assert(test_output.keys() == {'logits'})
            assert(not test_output['logits'].requires_grad)

            predict_output = model(batch, 'update_encoder', 'predict')

            assert(predict_output.keys() == {'logits'})
            assert(not predict_output['logits'].requires_grad)

