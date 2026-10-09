"""Prediction logging shares collectors without invoking training assessments."""

import types

import pytest
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from hypatorch import Model, Trainer
from hypatorch.train import Callback
from hypatorch.logger import WandbLogger
from test_validation_histograms import ENTRY, RecordingLogger, Samples, Stream, make_model


class Recorder(Callback):
    def __init__(self):
        self.events = []
        self.outputs = []

    def on_predict_start(self):
        self.events.append("start")

    def on_predict_batch_end(self, output, batch, batch_idx):
        self.events.append(batch_idx)
        self.outputs.append(output)

    def on_predict_end(self):
        self.events.append("end")


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_full_prediction_pass_includes_tail_and_resets_between_calls(batch_size):
    logger = RecordingLogger()
    callback = Recorder()
    trainer = Trainer(device="cpu", callbacks=[callback], max_val_samples=1)
    loader = DataLoader(Samples([0, .5, 1, 2, 3]), batch_size=batch_size)
    model = make_model([ENTRY, {"fn": "log_images", "log_image_keys": [{"key": "score"}]}])
    for _ in range(2):
        trainer.predict(model, loader, logger=logger)
    assert [h[1]["counts"] for h in logger.histograms] == [[2, 2], [2, 2]]
    assert [h[1]["total_count"] for h in logger.histograms] == [5, 5]
    assert logger.histograms[0][1]["global_step"] < logger.histograms[1][1]["global_step"]
    assert len(logger.images) == 2
    assert logger.events == ["histogram", "epoch", "histogram", "epoch"]
    assert callback.events == (["start", *range(len(loader)), "end"] * 2)
    assert trainer.train_step == trainer.val_step == trainer.train_samples == 0
    assert trainer.optimizers is None and trainer.last_checkpoint_path is None


def test_no_logger_keeps_callback_only_behavior_even_with_training_logger(monkeypatch):
    callback = Recorder()
    logger = RecordingLogger()
    trainer = Trainer(device="cpu", logger=logger, callbacks=[callback])
    model = make_model([dict(ENTRY, bins=None)])  # ignored without prediction logging
    monkeypatch.setattr(model, "iter_logging_entries", lambda: pytest.fail("logging inspected"))
    trainer.should_stop = True  # historically ignored by callback-only prediction
    trainer.predict(model, DataLoader(Samples([0, 1, 2]), batch_size=2, drop_last=True))
    assert callback.events == ["start", 0, "end"]
    assert callback.outputs[0]["score"].tolist() == [0, 1]
    assert not logger.histograms and not logger.events
    assert trainer.global_step == 0


class PredictionModel(Model):
    def __init__(self):
        super().__init__(submodules={}, operations={"evaluate": {"logging": [ENTRY]}})
        self.weight = torch.nn.Parameter(torch.tensor(2.))

    def forward(self, input_dict, operation_name, mode):
        assert mode == "predict" and not self.training
        assert not torch.is_grad_enabled()
        return {"score": input_dict["value"] * self.weight}

    def compute_loss(self, *args, **kwargs):
        pytest.fail("prediction must not run losses")

    def compute_metrics(self, *args, **kwargs):
        pytest.fail("prediction must not run metric assessments")

    def configure_optimizers(self):
        pytest.fail("prediction must not configure optimizers")


def test_collects_merged_model_outputs_without_assessments_or_gradients():
    logger = RecordingLogger()
    callback = Recorder()
    model = PredictionModel()
    Trainer(device="cpu", callbacks=[callback]).predict(
        model, [{"value": torch.tensor(0.), "id": "a"},
                {"value": torch.tensor(1.), "id": "b"}],
        loader_args={"batch_size": 2}, logger=logger,
    )
    assert logger.histograms[0][1]["counts"] == [1, 1]
    assert callback.outputs[0]["id"] == ["a", "b"]
    assert not callback.outputs[0]["score"].requires_grad
    assert model.weight.grad is None and model.weight.item() == 2.


@pytest.mark.parametrize("empty", [False, True])
def test_stream_exhaustion_and_empty_pass(empty):
    logger = RecordingLogger()
    loader = DataLoader(Samples([]) if empty else Stream(), batch_size=2)
    Trainer(device="cpu").predict(make_model(), loader, logger=logger)
    assert logger.histograms[0][1]["total_count"] == (0 if empty else 5)


@pytest.mark.parametrize("loader", [
    DataLoader(Samples([0, 1, 2]), batch_size=2, drop_last=True),
    DataLoader(Samples([0, 1, 2]), batch_size=2, sampler=SubsetRandomSampler([0, 1])),
    DataLoader(Samples([0, 1, 2]), batch_size=None),
])
def test_incomplete_loader_fails_before_callbacks(loader):
    callback = Recorder()
    logger = RecordingLogger()
    with pytest.raises(ValueError, match="Full prediction"):
        Trainer(device="cpu", callbacks=[callback]).predict(make_model(), loader, logger=logger)
    assert not callback.events and not logger.histograms


def test_dropping_collate_detected_at_end():
    logger = RecordingLogger()
    loader = DataLoader(Samples([0, 1, 2, 3]), batch_size=2,
                        collate_fn=lambda rows: {"score": torch.stack([rows[0]["score"]])})
    with pytest.raises(RuntimeError, match="Incomplete prediction coverage"):
        Trainer(device="cpu").predict(make_model(), loader, logger=logger)
    assert not logger.histograms


@pytest.mark.parametrize("failure", ["stop", "raise", "end"])
def test_callback_failure_or_stop_prevents_histogram_and_retry_starts_fresh(failure):
    logger = RecordingLogger()
    trainer = Trainer(device="cpu")
    class Failing(Callback):
        def on_predict_batch_end(self, output, batch, batch_idx):
            if failure == "stop":
                trainer.should_stop = True
            elif failure == "raise":
                raise RuntimeError("callback failed")
        def on_predict_end(self):
            raise RuntimeError("end failed")
    trainer.callbacks = [Failing()]
    with pytest.raises(RuntimeError):
        trainer.predict(make_model(), Samples([0, 1]), logger=logger)
    assert not logger.histograms
    trainer.should_stop = False
    trainer.callbacks = []
    trainer.predict(make_model(), Samples([2]), logger=logger)
    assert logger.histograms[0][1]["total_count"] == 1


def test_distributed_logging_rejected_before_callbacks():
    trainer = Trainer(device="cpu")
    trainer.distributed = types.SimpleNamespace(enabled=True)
    with pytest.raises(ValueError, match="single-process"):
        trainer.predict(make_model(), Samples([0]), logger=RecordingLogger())


@pytest.mark.parametrize("with_audio", [False, True])
def test_real_wandb_prediction_logs_histogram_and_table(tmp_path, monkeypatch, with_audio):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    columns = [{"name": "score", "key": "score"}]
    if with_audio:
        pytest.importorskip("soundfile")
        columns.append({"name": "audio", "key": "audio", "media_type": "audio",
                        "sample_rate": 16000, "len_key": "audio_len"})
    entries = [ENTRY, {"fn": "log_table", "name": "predict/examples", "sample_index": None,
                       "columns": columns}]
    with wandb.init(mode="offline", dir=str(tmp_path), project="hypatorch-predict-test") as run:
        trainer = Trainer(device="cpu")
        logger = WandbLogger()
        samples = [{"score": score, "audio": torch.zeros(1, 160), "audio_len": 80}
                   for score in [0., .5, 1., 2., 3.]]
        trainer.predict(make_model(entries), samples,
                        loader_args={"batch_size": 2}, logger=logger)
        run.log({}, commit=True)
        assert run.summary["val/distribution"]["values"] == [2, 2]
        assert run.summary["val/distribution/total_count"] == 5
        assert run.summary["predict/examples"]["nrows"] == 2


def test_later_missing_key_never_emits_partial_histogram():
    class BrokenStream(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield {"score": torch.tensor(0.)}
            yield {"other": torch.tensor(1.)}
    logger = RecordingLogger()
    with pytest.raises(KeyError, match="score"):
        Trainer(device="cpu").predict(make_model(), BrokenStream(), logger=logger)
    assert not logger.histograms
