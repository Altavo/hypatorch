import os
import unittest

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from shared import add_path


class TestApplyValidation(unittest.TestCase):

    def setUp(self):
        self.training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
        rel_config_dir = os.path.relpath(self.training_path, os.path.dirname(__file__))
        with initialize(
            config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
        ):
            self.cfg = compose(config_name="config.yaml", overrides=['experiment=mnist_linear'])
        OmegaConf.set_struct(self.cfg, False)

    def _mapping(self):
        return self.cfg.model.operations.update_encoder.mappings[0].image_encoder

    def test_a_mode_list_is_accepted(self):
        with add_path(self.training_path):
            self._mapping().apply = ['train', 'val']

            model = instantiate(self.cfg.model)

            assert model is not None

    def test_a_misspelled_mode_is_rejected(self):
        with add_path(self.training_path):
            self._mapping().apply = ['trian']

            with self.assertRaises(Exception) as context:
                instantiate(self.cfg.model)

            assert 'trian' in str(context.exception)

    def test_a_string_naming_no_mode_is_rejected(self):
        with add_path(self.training_path):
            self._mapping().apply = 'inference'

            with self.assertRaises(Exception) as context:
                instantiate(self.cfg.model)

            assert 'inference' in str(context.exception)

    def test_a_string_is_matched_by_containment(self):
        # 'evaluation' contains 'val', so it runs in validation. The string form
        # is a substring test, and this is what that costs.
        with add_path(self.training_path):
            self._mapping().apply = 'evaluation'

            model = instantiate(self.cfg.model)

            assert model is not None

    def test_a_string_naming_a_mode_is_accepted(self):
        with add_path(self.training_path):
            self._mapping().apply = 'train_val'

            model = instantiate(self.cfg.model)

            assert model is not None

    def test_an_assessment_mode_is_validated(self):
        with add_path(self.training_path):
            self.cfg.model.operations.update_encoder.losses[0].apply = ['trian']

            with self.assertRaises(Exception) as context:
                instantiate(self.cfg.model)

            assert 'trian' in str(context.exception)
