"""Pass-level logging uses generic model outputs and ordinary Torch loaders."""

import sys
import types

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SubsetRandomSampler

from hypatorch import Model, Trainer
from hypatorch.logger import DataLogger, WandbLogger
from hypatorch.validation_logging import HistogramCollector, validation_collectors


ENTRY = {
    "fn": "log_histogram",
    "name": "val/distribution", "key": "score", "bins": {"edges": [0.0, 1.0, 2.0]},
}


class RecordingLogger(DataLogger):
    def __init__(self):
        super().__init__()
        self.histograms = []
        self.images = []
        self.events = []

    def report_step(self):
        pass

    def report_epoch(self):
        self.events.append("epoch")

    def log_histogram(self, name, **kwargs):
        self.events.append("histogram")
        self.histograms.append((name, kwargs))

    def log_images(self, **kwargs):
        self.images.append(kwargs)


class Samples(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"score": torch.tensor(self.values[index], dtype=torch.float64)}


class Stream(IterableDataset):
    def __iter__(self):
        yield from Samples([0.0, 0.5, 1.0, 2.0, 3.0])


def make_model(entries=None):
    return instantiate(OmegaConf.create({
        "_target_": "hypatorch.core.Model", "submodules": {},
        "operations": {"evaluate": {"logging": entries if entries is not None else [ENTRY]}},
    }))


def evaluate(model, loader, logger=None, trainer=None):
    trainer = trainer or Trainer(device="cpu", save_last=False)
    model.eval()
    trainer.epoch(mode="val", model=model, epoch=0, dataset=loader, logger=logger)
    return trainer


def test_counts_are_batch_invariant_and_account_for_every_observation():
    values = torch.tensor([-1, 0, 0.5, 1, 1.5, 2, 3, float("nan"), float("inf"), -float("inf")], dtype=torch.float64)
    results = []
    for batch_size in [1, 3, len(values)]:
        collector = HistogramCollector("distribution", "score", {"edges": [0, 1, 2]})
        for chunk in values.split(batch_size):
            collector.update({"score": chunk.reshape(-1, 1).requires_grad_()}, len(chunk))
        assert collector.counts.tolist() == [2, 3]
        assert collector.total_count == 10
        assert collector.underflow_count == collector.overflow_count == 1
        assert collector.invalid_count == 3
        assert int(collector.counts.sum()) + collector.invalid_count + collector.underflow_count + collector.overflow_count == collector.total_count
        assert collector.counts.numel() == 2 and collector.counts.grad_fn is None
        results.append(collector.counts)
    assert all(torch.equal(results[0], result) for result in results)


@pytest.mark.parametrize("edges", [[], [0], [0, 0], [1, 0], [0, float("inf")], [0, float("nan")], list(range(514))])
def test_invalid_edges_are_rejected(edges):
    with pytest.raises(ValueError, match="bins"):
        HistogramCollector("distribution", "score", {"edges": edges})


@pytest.mark.parametrize("value", [torch.tensor(1.), torch.ones(2, 2), torch.ones(1), torch.ones(2, 1, 1)])
def test_collector_rejects_scalar_reductions_and_wrong_shapes(value):
    collector = HistogramCollector("distribution", "score", {"edges": [0, 1, 2]})
    with pytest.raises(ValueError, match="one value per example"):
        collector.update({"score": value}, batch_size=2)
    assert collector.total_count == 0


@pytest.mark.parametrize("value", [torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.complex64)])
def test_non_real_metrics_are_rejected(value):
    with pytest.raises(TypeError, match="real numeric"):
        HistogramCollector("distribution", "score", {"edges": [0, 1]}).update({"score": value}, 2)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_hydra_configuration_collects_entire_pass_and_preserves_first_batch_logging(batch_size):
    model = make_model([ENTRY, {"fn": "log_images", "log_image_keys": [{"key": "score"}]}])
    logger = RecordingLogger()
    loader = DataLoader(Samples([0, 0.5, 1, 2, 3]), batch_size=batch_size)
    trainer = evaluate(model, loader, logger)
    assert len(logger.histograms) == len(logger.images) == 1
    name, result = logger.histograms[0]
    assert name == "val/distribution"
    assert result["counts"] == [2, 2] and result["overflow_count"] == 1
    assert result["total_count"] == 5
    assert result["global_step"] == trainer.global_step - 1
    assert logger.events == ["histogram", "epoch"]
    evaluate(model, loader, logger, trainer)
    assert logger.histograms[1][1]["total_count"] == 5  # no accumulation across passes


def test_iterable_exhaustion_includes_final_partial_batch_and_flushes_only_at_end():
    logger = RecordingLogger()
    loader = DataLoader(Stream(), batch_size=2)
    def batches():
        for batch in loader:
            assert not logger.histograms
            yield batch
        assert not logger.histograms
    evaluate(make_model(), batches(), logger)
    assert logger.histograms[0][1]["total_count"] == 5


def test_empty_validation_logs_zero_coverage():
    logger = RecordingLogger()
    evaluate(make_model(), DataLoader(Samples([]), batch_size=2), logger)
    assert logger.histograms[0][1]["total_count"] == 0
    assert logger.histograms[0][1]["counts"] == [0, 0]


@pytest.mark.parametrize("cap", [0, 2, 100])
def test_any_validation_cap_fails_before_reading(cap):
    with pytest.raises(ValueError, match="no cap"):
        evaluate(make_model(), [], trainer=Trainer(device="cpu", max_val_samples=cap))


@pytest.mark.parametrize("loader", [
    DataLoader(Samples([0, 1, 2]), batch_size=2, drop_last=True),
    DataLoader(Samples([0, 1, 2]), batch_size=2, sampler=SubsetRandomSampler([0, 1])),
    DataLoader(Samples([0, 1, 2]), batch_size=2, sampler=RandomSampler(Samples([0, 1, 2]), replacement=True)),
    DataLoader(Samples([0, 1, 2]), batch_size=None),
])
def test_sample_dropping_or_unverifiable_loaders_are_rejected(loader):
    logger = RecordingLogger()
    with pytest.raises(ValueError, match="Full validation"):
        evaluate(make_model(), loader, logger)
    assert not logger.histograms


def test_standard_shuffled_loader_still_covers_dataset():
    logger = RecordingLogger()
    evaluate(make_model(), DataLoader(Samples([0, 1, 2]), batch_size=2, shuffle=True), logger)
    assert logger.histograms[0][1]["total_count"] == 3


def test_collate_dropping_samples_fails_coverage_check():
    logger = RecordingLogger()
    loader = DataLoader(Samples([0, 1, 2, 3]), batch_size=2,
                        collate_fn=lambda samples: {"score": torch.stack([s["score"] for s in samples[:1]])})
    with pytest.raises(RuntimeError, match="Incomplete validation coverage"):
        evaluate(make_model(), loader, logger)
    assert not logger.histograms


def test_interrupted_or_failed_pass_never_publishes_and_next_pass_starts_fresh():
    trainer = Trainer(device="cpu")
    logger = RecordingLogger()
    model = make_model()
    def interrupted():
        yield {"score": torch.tensor([0., 1.])}
        trainer.should_stop = True
        yield {"score": torch.tensor([2.])}
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate(model, interrupted(), logger, trainer)
    assert not logger.histograms
    trainer.should_stop = False
    def failed():
        yield {"score": torch.tensor([0.])}
        raise RuntimeError("dataset failed")
    with pytest.raises(RuntimeError, match="dataset failed"):
        evaluate(model, failed(), logger, trainer)
    assert not logger.histograms
    evaluate(model, [{"score": torch.tensor([1.])}], logger, trainer)
    assert logger.histograms[0][1]["total_count"] == 1


def test_missing_key_on_later_batch_does_not_log_partial_histogram():
    logger = RecordingLogger()
    with pytest.raises(KeyError, match="score"):
        evaluate(make_model(), [{"score": torch.tensor([0.])}, {"other": torch.tensor([1.])}], logger)
    assert not logger.histograms


def test_distributed_execution_is_rejected_before_training_or_rank_zero_branch():
    trainer = Trainer(device="cpu")
    trainer.state_model = make_model()
    trainer.distributed = types.SimpleNamespace(enabled=True, is_rank_zero=False)
    with pytest.raises(ValueError, match="single-process"):
        trainer._training_loop(train_dataset=[], val_dataset=[], loader_args={})


@pytest.mark.parametrize("changes", [
    {"aggregate": "epoch"}, {"aggregate": None}, {"aggregate": "validation_pass"},
    {"sample_index": 0}, {"bins": {"edges": [True, 1]}},
])
def test_unsupported_configuration_is_rejected(changes):
    entry = dict(ENTRY, **changes)
    with pytest.raises((ValueError, TypeError)):
        validation_collectors([entry])


def test_duplicate_names_across_operations_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validation_collectors([ENTRY, ENTRY])


def test_training_does_not_collect_histograms():
    logger = RecordingLogger()
    Trainer(device="cpu").epoch(mode="train", model=make_model(), epoch=0,
                                dataset=[{"score": torch.tensor([0.])}], logger=logger)
    assert not logger.histograms


def test_wandb_receives_precomputed_counts_and_diagnostics(monkeypatch):
    calls = []
    run = types.SimpleNamespace(define_metric=lambda *args, **kwargs: None,
                                log=lambda data, **kwargs: calls.append((data, kwargs)))
    sdk = types.SimpleNamespace(run=run, Histogram=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    logger = WandbLogger()
    logger.log_value("samples", 20)
    collector = HistogramCollector("val/distribution", "score", {"edges": [0, 1, 2]})
    collector.update({"score": [-1, 0, 1, 2, 3, float("nan")]}, 6)
    collector.emit(logger, 7)
    payload, options = calls[0]
    assert payload["val/distribution"] == {"np_histogram": ([1, 2], [0., 1., 2.])}
    assert payload["val/distribution/total_count"] == 6
    assert payload["val/distribution/underflow_count"] == payload["val/distribution/overflow_count"] == payload["val/distribution/invalid_count"] == 1
    assert payload["samples"] == 20 and payload["global_step"] == 7
    assert options == {"step": 7, "commit": False}
    assert logger._epoch_log == {}  # counts are not scalar-averaged


def test_real_wandb_full_validation_histogram(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    with wandb.init(mode="offline", dir=str(tmp_path), project="hypatorch-histogram-test") as run:
        logger = WandbLogger()
        evaluate(make_model(), DataLoader(Samples([-1, 0, 0.5, 1, 2, 3, float("nan")]), batch_size=2), logger)
        run.log({}, commit=True)
        result = run.summary["val/distribution"]
        assert result["_type"] == "histogram"
        assert result["values"] == [2, 2]
        assert result["bins"] == [0., 1., 2.]
        assert run.summary["val/distribution/total_count"] == 7


class Scores(torch.nn.Module):
    def forward(self, values):
        score = values * 2
        return score


def test_histogram_can_collect_model_outputs_without_metric_assessments():
    model = Model(submodules={"scores": Scores()}, operations={"evaluate": {
        "logging": [ENTRY], "mappings": [{"scores": {
            "inputs": {"values": "inputs"}, "outputs": {"score": "score"},
            "calculate_grad": False, "apply": ["val"],
        }}],
    }})
    logger = RecordingLogger()
    evaluate(model, [{"inputs": torch.tensor([0., 0.5])}, {"inputs": torch.tensor([1.])}], logger)
    assert logger.histograms[0][1]["counts"] == [1, 2]
    assert logger.histograms[0][1]["total_count"] == 3


def test_two_collectors_are_independent_and_a_later_error_prevents_both_emissions():
    second = dict(ENTRY, name="val/other", key="other", bins={"edges": [-2., 0., 2.]})
    model = make_model([ENTRY, second])
    logger = RecordingLogger()
    evaluate(model, [{"score": torch.tensor([0., 1.]), "other": torch.tensor([-1., 1.])}], logger)
    assert [record[0] for record in logger.histograms] == ["val/distribution", "val/other"]
    assert [record[1]["counts"] for record in logger.histograms] == [[1, 1], [1, 1]]
    logger.histograms.clear()
    with pytest.raises(KeyError):
        evaluate(model, [{"score": torch.tensor([0.]), "other": torch.tensor([1.])},
                         {"score": torch.tensor([1.])}], logger)
    assert not logger.histograms


def test_no_validation_dataset_or_loader_cap_is_rejected_before_training():
    trainer = Trainer(device="cpu")
    trainer.state_model = make_model()
    with pytest.raises(ValueError, match="require a validation dataset"):
        trainer._training_loop(train_dataset=[], loader_args={})
    with pytest.raises(ValueError, match="drop_last"):
        trainer._training_loop(train_dataset=[], val_dataset=Stream(), loader_args={"drop_last": True})


def test_collection_has_no_dependency_on_consumer_training_packages():
    # A fresh interpreter blocks even indirect imports of consumer packages.
    import subprocess
    code = '''
import importlib.abc
import sys
class NoConsumers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"altavo_mltraining", "mlops_components", "datavo_sdk"}:
            raise AssertionError("Unexpected consumer dependency: " + fullname)
sys.meta_path.insert(0, NoConsumers())
from hypatorch import Model, Trainer
from hypatorch.validation_logging import HistogramCollector
collector = HistogramCollector("distribution", "values", {"edges": [0, 1, 2]})
collector.update({"values": [0, 1, 2]}, 3)
assert collector.counts.tolist() == [1, 2]
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_python_float_observations_keep_precision_near_bin_edges():
    collector = HistogramCollector("distribution", "score", {"edges": [0., 1., 2.]})
    collector.update({"score": [1. - 1e-10, 1., 1. + 1e-10]}, 3)
    assert collector.counts.tolist() == [1, 2]


@pytest.mark.parametrize("bins", [
    {"range": [0., 2.], "count": 2}, {"edges": [0., 1., 2.]},
])
def test_both_bin_specs_resolve_before_validation_and_produce_identical_counts(bins):
    entry = dict(ENTRY, bins=bins)
    model = make_model([entry])
    logger = RecordingLogger()
    evaluate(model, DataLoader(Samples([0, 0.5, 1, 2, 3]), batch_size=2), logger)
    result = logger.histograms[0][1]
    assert result["bin_edges"] == [0., 1., 2.]
    assert result["counts"] == [2, 2]
    assert result["overflow_count"] == 1


@pytest.mark.parametrize("bins", [
    None, [], 100, {}, {"range": [0, 2]}, {"count": 100},
    {"edges": [0, 1], "range": [0, 2], "count": 2},
    {"edges": [0, 1], "count": 2}, {"edges": [0, 1], "extra": True},
    {"range": [0, 2], "count": 0}, {"range": [0, 2], "count": 513},
    {"range": [0, 2], "count": True}, {"range": [0, 2], "count": 2.0},
    {"range": [0, 2], "count": "2"}, {"range": [0, 1, 2], "count": 2},
    {"range": [2, 0], "count": 2}, {"range": [1, 1], "count": 2},
    {"range": [0, float("inf")], "count": 2},
    {"range": [0, float("nan")], "count": 2},
    {"range": [False, 2], "count": 2}, {"range": "02", "count": 2},
    {"range": [1.0, 1.0000000000000002], "count": 2},
])
def test_invalid_bin_specs_fail_before_reading_validation(bins):
    trainer = Trainer(device="cpu")
    trainer.state_model = make_model([dict(ENTRY, bins=bins)])
    with pytest.raises((ValueError, TypeError), match="bins"):
        trainer._training_loop(train_dataset=[], val_dataset=[], loader_args={})


@pytest.mark.parametrize("count", [1, 100, 512])
def test_uniform_bin_counts_and_endpoints(count):
    collector = HistogramCollector("distribution", "score", OmegaConf.create({"range": [-1., 2.], "count": count}))
    assert len(collector.bin_edges) == count + 1
    assert collector.bin_edges[0] == -1. and collector.bin_edges[-1] == 2.
    collector.update({"score": [-1., 2.]}, 2)
    assert collector.counts.sum() == 2
    assert collector.underflow_count == collector.overflow_count == 0


def test_old_bin_edges_config_is_rejected():
    entry = {key: value for key, value in ENTRY.items() if key != "bins"}
    entry["bin_edges"] = [0, 1, 2]
    with pytest.raises(TypeError, match="bin_edges"):
        validation_collectors([entry])
