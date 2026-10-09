"""Conventional histogram snapshots retain exact per-example values per pass."""
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from hypatorch import Trainer
from hypatorch.logger import DataLogger, DistributedLogger, WandbLogger
from hypatorch.validation_logging import DistributionCollector, validation_collectors
from test_validation_histograms import Samples, make_model, evaluate, ENTRY as HISTOGRAM

ENTRY = {"fn": "log_distribution", "name": "val/distribution", "key": "score"}


class Recorder(DataLogger):
    def __init__(self):
        super().__init__()
        self.distributions = []

    def report_step(self): pass
    def report_epoch(self): pass

    def log_distribution(self, name, **kwargs):
        self.distributions.append((name, kwargs))


@pytest.mark.parametrize("prediction", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_complete_pass_and_tail_reset_across_passes(prediction, batch_size):
    logger = Recorder()
    trainer = Trainer(device="cpu", save_last=False)
    model = make_model([ENTRY])
    for values in ([0.1, 0.2, 0.3, 2., 8.], [9., 10.]):
        loader = DataLoader(Samples(values), batch_size=batch_size)
        if prediction:
            trainer.predict(model, loader, logger=logger)
        else:
            evaluate(model, loader, logger, trainer)
        assert logger.distributions[-1][1]["values"] == values
    assert len(logger.distributions) == 2
    assert logger.distributions[0][1]["global_step"] < logger.distributions[1][1]["global_step"]


def test_collector_owns_cpu_values_without_graph_or_storage_aliasing():
    collector = DistributionCollector("dist", "score")
    tensor = torch.tensor([[0.125], [0.5]], dtype=torch.float64, requires_grad=True)
    collector.update({"score": tensor}, 2)
    with torch.no_grad(): tensor.fill_(42)
    collector.update({"score": [0.123456789012345]}, 1)
    assert collector.values == [0.125, 0.5, 0.123456789012345]


@pytest.mark.parametrize("value,error", [
    ([float('nan')], ValueError), ([float('inf')], ValueError),
    ([True], TypeError), ([1j], TypeError), (1., ValueError), ([[1., 2.]], ValueError),
])
def test_invalid_observations_rejected_without_mutating_collector(value, error):
    collector = DistributionCollector("dist", "score")
    with pytest.raises(error): collector.update({"score": value}, 1)
    assert collector.values == []


@pytest.mark.parametrize("prediction", [False, True])
def test_no_chart_for_oversized_or_incomplete_pass(prediction):
    for entry, drop_last in [(dict(ENTRY, max_values=2), False), (ENTRY, True)]:
        logger = Recorder()
        model = make_model([entry])
        loader = DataLoader(Samples([1., 2., 3.]), batch_size=2, drop_last=drop_last)
        with pytest.raises(ValueError):
            if prediction: Trainer(device="cpu").predict(model, loader, logger=logger)
            else: evaluate(model, loader, logger)
        assert logger.distributions == []


@pytest.mark.parametrize("changes", [{"name": ""}, {"name": "samples"}, {"key": ""},
                                       {"max_values": 0}, {"max_values": True}, {"max_values": 1.5},
                                       {"bins": {"edges": [0, 1]}}])
def test_bad_configuration_fails_early(changes):
    with pytest.raises((ValueError, TypeError)):
        validation_collectors([dict(ENTRY, **changes)])


def test_duplicate_names_shared_with_histograms_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validation_collectors([ENTRY, HISTOGRAM])


def test_no_first_batch_or_training_emission():
    logger = Recorder()
    model = make_model([ENTRY])
    model.log_data(logger, 0, {"score": torch.tensor([1.])})
    Trainer(device="cpu").epoch(mode="train", model=model, epoch=0,
                                dataset=DataLoader(Samples([1.])), logger=logger)
    Trainer(device="cpu").predict(model, DataLoader(Samples([1.])))
    assert not logger.distributions


def test_wandb_sink_creates_fresh_table_and_chart_and_guards_serialization(monkeypatch):
    calls = []
    class Table:
        MAX_ROWS = 3
        MAX_ARTIFACT_ROWS = 4
        def __init__(self, columns, data): self.columns, self.data = columns, data
    sdk = types.SimpleNamespace(
        Table=Table, plot=types.SimpleNamespace(histogram=lambda table, value, title: (table, value, title)),
        run=types.SimpleNamespace(define_metric=lambda *a, **k: None,
                                 log=lambda payload, **kw: calls.append((payload, kw))))
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    logger = WandbLogger()
    for step in [5, 10]: logger.log_distribution("dist", values=[0.25, 0.75], global_step=step)
    first, second = [call[0]["dist"] for call in calls]
    assert first[0] is not second[0]
    assert first[0].columns == ["value"] and first[0].data == [[0.25], [0.75]]
    assert first[1:] == ("value", "dist")
    assert [kw for _, kw in calls] == [{"step": 5, "commit": False}, {"step": 10, "commit": False}]
    with pytest.raises(ValueError, match="table limit"): logger.log_distribution("dist", values=[0.] * 5)
    assert len(calls) == 2
    logger.log_distribution("above_preview_limit", values=[0.] * 4)
    assert len(calls[-1][0]["above_preview_limit"][0].data) == 4
    logger.log_distribution("empty", values=[])
    assert calls[-1][0]["empty"][0].data == []


def test_distributed_logger_only_forwards_on_rank_zero():
    logger = Recorder()
    for rank_zero in [False, True]:
        DistributedLogger(logger, types.SimpleNamespace(is_rank_zero=rank_zero)).log_distribution(
            "dist", values=[1.], global_step=3)
    assert logger.distributions == [("dist", {"values": [1.], "global_step": 3})]


def test_real_wandb_offline_serialization(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    with wandb.init(mode="offline", project="hypatorch-distributions", dir=str(tmp_path)) as run:
        from unittest.mock import Mock
        config_calls = Mock(wraps=run._config_callback)
        monkeypatch.setattr(run, "_config_callback", config_calls)
        logger = WandbLogger()
        logger.log_distribution("val/dist", values=[0.1, 0.2, 0.3], global_step=1)
        logger.log_distribution("val/dist", values=[0.4, 0.5], global_step=2)
        files = list((Path(run.dir) / 'media/table/val').glob('*.json'))
        tables = [json.loads(path.read_text()) for path in files]
        assert sorted(len(table['data']) for table in tables) == [2, 3]
        assert all(table['columns'] == ['value'] for table in tables)
        assert any(table['data'] == [[0.1], [0.2], [0.3]] for table in tables)
        charts = [call.kwargs["val"]["panel_config"] for call in config_calls.call_args_list
                  if call.kwargs.get("key") == ("_wandb", "visualize", "val/dist")]
        assert len(charts) == 2
        chart = charts[-1]
        assert chart['panelDefId'] == 'wandb/histogram/v0'
        assert chart['fieldSettings']['value'] == 'value'


@pytest.mark.parametrize("prediction", [False, True])
def test_failed_pass_does_not_publish_partial_distribution(prediction):
    logger = Recorder()
    trainer = Trainer(device="cpu")
    model = make_model([ENTRY])
    class FailingDataset(IterableDataset):
        def __iter__(self):
            yield {"score": torch.tensor(1.)}
            raise RuntimeError("dataset failed")
    loader = DataLoader(FailingDataset(), batch_size=1)
    with pytest.raises(RuntimeError, match="dataset failed"):
        if prediction:
            trainer.predict(model, loader, logger=logger)
        else:
            evaluate(model, loader, logger, trainer)
    assert logger.distributions == []


def test_default_limit_accepts_20000_and_rejects_next_value():
    collector = DistributionCollector("dist", "score")
    collector.update({"score": [0.5] * 20000}, 20000)
    assert len(collector.values) == 20000
    with pytest.raises(ValueError, match="max_values=20000"):
        collector.update({"score": [0.5]}, 1)
    assert len(collector.values) == 20000


def test_real_wandb_20000_values_reach_chart_artifact(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    serialized = []
    original = wandb.Table.to_json
    def capture(table, destination):
        result = original(table, destination)
        if isinstance(destination, wandb.Artifact):
            serialized.append(result)
        return result
    monkeypatch.setattr(wandb.Table, "to_json", capture)
    values = [float(i) for i in range(20000)]
    with wandb.init(mode="offline", project="hypatorch-distributions", dir=str(tmp_path)):
        WandbLogger().log_distribution("dist", values=values, global_step=1)
    assert serialized
    assert any(table["columns"] == ["value"] and table["data"] == [[v] for v in values]
               for table in serialized)
