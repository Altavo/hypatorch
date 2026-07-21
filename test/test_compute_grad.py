import unittest
import os
import sys
import torch
from hydra import initialize, compose
from hydra.utils import instantiate

import hypatorch

from shared import add_path


class _CaptureLogger:
    """Records log_value calls so tests can inspect the metric names emitted."""

    def __init__(self):
        self.values = {}

    def log_value(self, name, value):
        self.values[name] = value


class TestComputeGrad(unittest.TestCase):


    def setUp(self):
        self.training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        rel_config_dir = os.path.relpath(self.training_path,os.path.dirname(__file__))
        with initialize(
            config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
        ):
            self.cfg = compose(config_name="config.yaml", overrides=['experiment=mnist_linear'])

    def test_modes(self):
        with add_path(self.training_path):
            model = instantiate(self.cfg.model)
            model.train()

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


    def test_assessment_metrics_are_slash_namespaced_by_mode(self):
        from hypatorch.utils import shared_dict

        with add_path(self.training_path):
            model = instantiate(self.cfg.model)
            model.train()

            batch = {
                'image': torch.randn(32, 1, 28, 28),
                'class': torch.randint(0, 10, (32,))
            }

            # Losses/metrics are logged under a "<mode>/" section so the
            # tracking backend groups them (e.g. "train/mean_cross_entropy").
            train_logger = _CaptureLogger()
            output_dict = model(batch, 'update_encoder', 'train')
            model.compute_loss(
                shared_dict(batch, output_dict), 'update_encoder', 'train',
                logger=train_logger,
            )
            assert train_logger.values, "expected at least one logged metric"
            assert all(name.startswith('train/') for name in train_logger.values), (
                train_logger.values
            )
            assert 'train/mean_cross_entropy' in train_logger.values

            val_logger = _CaptureLogger()
            val_output = model(batch, 'update_encoder', 'val')
            model.compute_loss(
                shared_dict(batch, val_output), 'update_encoder', 'val',
                logger=val_logger,
            )
            assert all(name.startswith('val/') for name in val_logger.values), (
                val_logger.values
            )

    def test_no_optimized_submodules(self):
        with add_path(self.training_path):
            # Modify the config to not optimize any submodules
            self.cfg.model.operations.update_encoder.optimize_submodules = []

            model = instantiate(self.cfg.model)
            model.train()

            batch = {
                'image': torch.randn(32, 1, 28, 28),
                'label': torch.randint(0, 10, (32,))                
            }

            train_output = model(batch, 'update_encoder', 'train')

            assert(train_output.keys() == {'logits'})
            assert(not train_output['logits'].requires_grad)

    def test_calculate_grad_false(self):
        with add_path(self.training_path):
            # Modify the config to not calculate gradients
            self.cfg.model.operations.update_encoder.mappings[0].image_encoder.calculate_grad = False

            model = instantiate(self.cfg.model)
            model.train()

            batch = {
                'image': torch.randn(32, 1, 28, 28),
                'label': torch.randint(0, 10, (32,))                
            }

            train_output = model(batch, 'update_encoder', 'train')

            assert(train_output.keys() == {'logits'})
            assert(not train_output['logits'].requires_grad)