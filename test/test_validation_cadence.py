"""When validation runs, relative to the training epochs it is supposed to measure.

`_finalize_last_checkpoint` keeps the LAST epoch's weights, so those are the weights a
run's reported metrics have to describe. Validating at the top of the epoch loop instead
of the bottom shifts the whole series one epoch early: it opens on the untrained model and
stops one short of the checkpoint. At `check_val_every_n_epoch == max_epochs` the series is
that single untrained row -- which is how an untrained CER reached a published model.
"""

import unittest

from hypatorch.train import Trainer


class _Recorder:
    """A Trainer whose epochs do nothing but record that they happened."""

    @staticmethod
    def build(max_epochs, check_val_every_n_epoch=1):
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            check_val_every_n_epoch=check_val_every_n_epoch,
        )
        calls = []

        def epoch(mode, model, epoch, dataset, **kwargs):
            calls.append((mode, epoch))

        trainer.epoch = epoch
        trainer._as_dataloader = lambda dataset, **kwargs: dataset
        trainer._finalize_last_checkpoint = lambda **kwargs: None
        trainer.get_rng_state_dict = lambda: {}
        trainer.model = _NullModel()
        trainer.optimizers = {}
        trainer.schedulers = {}
        trainer.gradient_clipping = None
        trainer.epoch_idx = 0
        trainer.should_stop = False
        return trainer, calls


class _NullModel:
    def train(self):
        return self

    def eval(self):
        return self


def _run(max_epochs, check_val_every_n_epoch):
    trainer, calls = _Recorder.build(max_epochs, check_val_every_n_epoch)
    trainer._training_loop(train_dataset=["train"], loader_args={}, val_dataset=["val"])
    return calls


class TestValidationCadence(unittest.TestCase):
    def test_validation_follows_the_epoch_it_measures(self):
        """Train then validate, not validate then train."""
        self.assertEqual(
            _run(max_epochs=3, check_val_every_n_epoch=1),
            [("train", 0), ("val", 0), ("train", 1), ("val", 1), ("train", 2), ("val", 2)],
        )

    def test_validating_once_measures_the_trained_model(self):
        """`check_val_every_n_epoch == max_epochs` is how a caller pays for the metric once.

        The one pass has to be the one that describes the saved checkpoint. Before this,
        it fired at epoch 0 -- before a single optimizer step -- and the run published the
        base model's score as the finetuned model's.
        """
        calls = _run(max_epochs=50, check_val_every_n_epoch=50)
        self.assertEqual([c for c in calls if c[0] == "val"], [("val", 49)])
        self.assertEqual(calls[-1], ("val", 49))
        self.assertEqual(len([c for c in calls if c[0] == "train"]), 50)

    def test_the_last_epoch_always_validates(self):
        """`save_last` keeps its weights whether or not N divides into the run."""
        calls = _run(max_epochs=10, check_val_every_n_epoch=4)
        self.assertEqual(
            [c for c in calls if c[0] == "val"],
            [("val", 3), ("val", 7), ("val", 9)],
        )

    def test_no_validation_without_a_val_dataset(self):
        trainer, calls = _Recorder.build(max_epochs=3)
        trainer._training_loop(train_dataset=["train"], loader_args={}, val_dataset=None)
        self.assertEqual(calls, [("train", 0), ("train", 1), ("train", 2)])


if __name__ == "__main__":
    unittest.main()
