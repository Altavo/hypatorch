import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


LOGGER_PATH = Path(__file__).resolve().parents[1] / "hypatorch" / "logger.py"
LOGGER_SPEC = importlib.util.spec_from_file_location("hypatorch_logger_for_test", LOGGER_PATH)
LOGGER_MODULE = importlib.util.module_from_spec(LOGGER_SPEC)
assert LOGGER_SPEC is not None and LOGGER_SPEC.loader is not None
LOGGER_SPEC.loader.exec_module(LOGGER_MODULE)
WandbLogger = LOGGER_MODULE.WandbLogger


class _FakeRun:
    def __init__(self):
        self.id = "run-123"
        self.logged = []
        self.artifacts = []
        self.finished = []
        self.defined_metrics = []

    def define_metric(self, name, step_metric=None):
        self.defined_metrics.append((name, step_metric))

    def log(self, data, step=None):
        self.logged.append((data, step))

    def log_artifact(self, artifact, aliases=None):
        self.artifacts.append((artifact, aliases))
        return artifact

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class _FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, local_path, name=None):
        self.files.append((local_path, name))


def _fake_wandb(run):
    return types.SimpleNamespace(
        run=run,
        Artifact=_FakeArtifact,
        Image=lambda figure: ("image", figure),
        Html=lambda text: ("html", text),
    )


def test_wandb_logger_logs_step_and_epoch_metrics():
    run = _FakeRun()
    original = sys.modules.get("wandb")
    sys.modules["wandb"] = _fake_wandb(run)
    try:
        logger = WandbLogger(log_every_n_steps=1)
        logger.log_value("loss", 0.5)
        logger.log_value("train_step", 3)
        logger.log_value("global_step", 7)
        logger.log_value("samples", 32)
        logger.step_done()
        logger.epoch_done()
    finally:
        if original is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original

    # `samples` is declared as the default display x-axis; global_step remains
    # the monotonic commit step.
    assert ("samples", None) in run.defined_metrics
    assert ("*", "samples") in run.defined_metrics

    # Progress counters are stamped verbatim as coordinates (not suffixed to
    # loss_step-style keys); only the assessment metric is suffixed. global_step
    # is the commit step passed to wandb.
    assert run.logged[0] == (
        {"loss_step": 0.5, "samples": 32, "global_step": 7, "train_step": 3},
        7,
    )
    assert run.logged[1][1] == 7
    assert run.logged[1][0]["loss_epoch"] == 0.5
    assert run.logged[1][0]["samples"] == 32


def test_wandb_logger_validation_is_one_aggregated_point_on_samples_axis():
    # Validation must produce a single aggregated point per pass, plotted at the
    # training-progress coordinate (samples) reached when the eval ran -- not
    # one point per val batch, and not at the restarted val_step. The commit
    # step (global_step) stays monotonic so nothing is dropped.
    run = _FakeRun()
    original = sys.modules.get("wandb")
    sys.modules["wandb"] = _fake_wandb(run)
    try:
        logger = WandbLogger(log_every_n_steps=1)

        # One training step: samples advanced to 320.
        logger.log_value("loss", 1.0)
        logger.log_value("train_step", 10)
        logger.log_value("global_step", 10)
        logger.log_value("samples", 320)
        logger.step_done()
        logger.epoch_done()

        # A validation pass of two batches: val_step restarts, global_step keeps
        # rising, samples is frozen at 320 (no training happened).
        logger.log_value("cer", 0.3)
        logger.log_value("val_step", 0)
        logger.log_value("global_step", 11)
        logger.log_value("samples", 320)
        logger.step_done()
        logger.log_value("cer", 0.5)
        logger.log_value("val_step", 1)
        logger.log_value("global_step", 12)
        logger.log_value("samples", 320)
        logger.step_done()
        logger.epoch_done()
    finally:
        if original is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original

    payloads = [data for data, _ in run.logged]
    steps = [step for _, step in run.logged]

    # Commit step never rewinds.
    assert steps == sorted(steps), f"commit steps not monotonic: {steps}"

    # Validation contributed exactly one point (no per-batch cer_step writes).
    assert not any("cer_step" in p for p in payloads)
    val_points = [p for p in payloads if "cer_epoch" in p]
    assert len(val_points) == 1

    # It is the aggregate over the pass, plotted at the frozen samples axis.
    assert val_points[0]["cer_epoch"] == 0.4
    assert val_points[0]["samples"] == 320


def test_wandb_logger_preserves_slash_namespaced_metric_keys():
    # Slash-namespaced assessment keys (train/…, val/…) must flow through as a
    # groupable section name, while flat progress coordinates stay flat.
    run = _FakeRun()
    original = sys.modules.get("wandb")
    sys.modules["wandb"] = _fake_wandb(run)
    try:
        logger = WandbLogger(log_every_n_steps=1)
        logger.log_value("train/mean_cross_entropy", 0.5)
        logger.log_value("train_step", 3)
        logger.log_value("global_step", 7)
        logger.log_value("samples", 32)
        logger.step_done()
    finally:
        if original is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original

    payload = run.logged[0][0]
    # The section-bearing key keeps its slash and gains the per-step suffix.
    assert "train/mean_cross_entropy_step" in payload
    # Coordinates remain flat (they are x-axes, not grouped metrics).
    assert payload["samples"] == 32
    assert payload["global_step"] == 7
    assert payload["train_step"] == 3
    assert "train_step_step" not in payload


def test_wandb_logger_logs_file_and_directory_artifacts(tmp_path):
    run = _FakeRun()
    original = sys.modules.get("wandb")
    sys.modules["wandb"] = _fake_wandb(run)
    try:
        logger = WandbLogger()
        checkpoint_path = tmp_path / "last.ckpt"
        checkpoint_path.write_text("checkpoint", encoding="utf-8")
        export_dir = tmp_path / "exports"
        nested_dir = export_dir / "nested"
        nested_dir.mkdir(parents=True)
        (export_dir / "root.txt").write_text("root", encoding="utf-8")
        (nested_dir / "child.txt").write_text("child", encoding="utf-8")

        logged_checkpoint = logger.log_artifact(
            str(checkpoint_path), artifact_path="checkpoints"
        )
        logger.log_artifact(str(export_dir), artifact_path="exports")
    finally:
        if original is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original

    checkpoint_artifact, checkpoint_aliases = run.artifacts[0]
    assert checkpoint_artifact.name == "run-run-123-checkpoints"
    assert checkpoint_artifact.type == "model"
    assert checkpoint_artifact.files == [(str(checkpoint_path), "last.ckpt")]
    assert checkpoint_aliases == ["latest"]
    assert logged_checkpoint is checkpoint_artifact

    export_artifact, export_aliases = run.artifacts[1]
    assert export_artifact.name == "run-run-123-exports"
    assert export_artifact.type == "artifact"
    assert export_artifact.files == [
        (str(export_dir / "nested" / "child.txt"), "exports/nested/child.txt"),
        (str(export_dir / "root.txt"), "exports/root.txt"),
    ]
    assert export_aliases == ["latest"]


def test_wandb_logger_finalize_uses_status_exit_code():
    run = _FakeRun()
    original = sys.modules.get("wandb")
    sys.modules["wandb"] = _fake_wandb(run)
    try:
        logger = WandbLogger()
        logger.finalize("FAILED")
        logger.finalize("FINISHED")
    finally:
        if original is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original

    assert run.finished == [1, 0]


# Rich media must never enter scalar aggregation or commit a step before metrics.


@pytest.mark.parametrize("media_type,class_name,options", [
    ("image", "Image", {"masks": {"prediction": {}}, "boxes": {"truth": {}}}),
    ("audio", "Audio", {"sample_rate": 16000}),
    ("video", "Video", {"fps": 24}),
    ("html", "Html", {"inject": False}),
    ("object3d", "Object3D", {}),
    ("molecule", "Molecule", {}),
    ("histogram", "Histogram", {"num_bins": 32}),
    ("plotly", "Plotly", {}),
])
def test_media_constructors_preserve_options_and_coordinates(monkeypatch, media_type, class_name, options):
    run = _FakeRun()
    calls = []
    run.log = lambda data, **kwargs: calls.append((data, kwargs))
    sdk = _fake_wandb(run)
    constructed = object()
    inputs = []
    setattr(sdk, class_name, lambda value, **kwargs: inputs.append((value, kwargs)) or constructed)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    logger = WandbLogger(log_every_n_steps=100)
    logger.log_value("global_step", 7)
    logger.log_value("samples", 32)
    logger.log_media("val/example", "source", media_type=media_type, **options)
    assert inputs == [("source", options)]
    assert calls == [({"global_step": 7, "samples": 32, "val/example": constructed},
                      {"step": 7, "commit": False})]
    assert "val/example" not in logger._step_log
    assert not logger._epoch_log


def test_media_table_tensor_and_prebuilt_graph(monkeypatch):
    run = _FakeRun()
    calls = []
    run.log = lambda data, **kwargs: calls.append((data, kwargs))
    sdk = _fake_wandb(run)
    sdk.Table = lambda **kwargs: kwargs
    sdk.Audio = lambda value, **kwargs: value
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    logger = WandbLogger()
    logger.log_media("examples", [[1]], media_type="table", columns=["id"])
    assert calls[-1][0]["examples"] == {"data": [[1]], "columns": ["id"]}
    logger.log_media("audio", torch.ones(8, requires_grad=True), media_type="audio", sample_rate=16000)
    assert calls[-1][0]["audio"].shape == (8,)
    graph = object()
    logger.log_media("model", graph, global_step=9)
    assert calls[-1] == ({"model": graph, "global_step": 9}, {"step": 9, "commit": False})
    with pytest.raises(ValueError, match="Unsupported"):
        logger.log_media("bad", media_type="unknown")
    with pytest.raises(ValueError, match="constructed"):
        logger.log_media("bad", media_type="graph")
    with pytest.raises(ValueError, match="reserved"):
        logger.log_media("samples", graph)
    with pytest.raises(TypeError, match="media_type"):
        logger.log_media("bad", graph, caption="unused")


def test_distributed_media_is_constructed_only_on_rank_zero(monkeypatch):
    run = _FakeRun()
    calls = []
    run.log = lambda *args, **kwargs: calls.append((args, kwargs))
    sdk = _fake_wandb(run)
    constructed = []
    sdk.Audio = lambda *args, **kwargs: constructed.append(args) or "audio"
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    runtime = types.SimpleNamespace(is_rank_zero=False)
    logger = LOGGER_MODULE.DistributedLogger(WandbLogger(), runtime)
    logger.log_media("audio", [0], media_type="audio", sample_rate=16000)
    assert not constructed and not calls
    runtime.is_rank_zero = True
    logger.log_media("audio", [0], media_type="audio", sample_rate=16000)
    assert len(constructed) == len(calls) == 1
    # Console / MLflow inherit the optional no-op without importing wandb.
    LOGGER_MODULE.ConsoleLogger().log_media("audio", [0], media_type="audio")


def test_real_wandb_graph_and_media_share_scalar_step(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    with wandb.init(mode="offline", dir=str(tmp_path), project="hypatorch-test") as run:
        logger = WandbLogger()
        logger.log_value("global_step", 0)
        logger.log_value("samples", 16)
        graph = wandb.Graph()
        source = graph.add_node(id="input", name="Input")
        target = graph.add_node(id="output", name="Output")
        graph.add_edge(source, target)
        logger.log_media("model/graph", graph)
        logger.log_media("distribution", [1, 2, 3], media_type="histogram")
        logger.log_media("examples", [[1, "example"]], media_type="table", columns=["id", "text"])
        logger.log_media("html_samples", data_dict={"html": ["<p>first</p>", "<p>second</p>"]},
                         key="html", sample_index=[1, 0], media_type="html")
        assert run.step == 0
        logger.log_value("loss", 0.5)
        logger.step_done()
        # Explicit-step scalar reports also leave the W&B history row open.
        assert run.step == 0
        run.log({}, commit=True)
        assert run.step == 1
        assert run.summary["loss_step"] == 0.5
        assert run.summary["model/graph"]["_type"] == "graph-file"
        assert run.summary["html_samples"]["count"] == 2


@pytest.mark.parametrize("selection", [1, [1], [1, 0, 1]])
def test_hydra_model_logging_selects_trims_and_orients_audio(monkeypatch, selection):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import numpy as np

    run = _FakeRun()
    calls = []
    run.log = lambda data, **kwargs: calls.append((data, kwargs))
    sdk = _fake_wandb(run)
    audio = []
    sdk.Audio = lambda value, **kwargs: audio.append((value, kwargs)) or "clip"
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    cfg = OmegaConf.create('''
_target_: hypatorch.core.Model
submodules: {}
operations:
  decoder:
    logging:
      - fn: log_media
        name: val/prediction
        media_type: audio
        key: prediction
        len_key: lengths
        sample_index: 1
        sample_rate: 16000
''')
    cfg.operations.decoder.logging[0].sample_index = selection
    model = instantiate(cfg)
    logger = WandbLogger()
    logger.log_value("samples", 32)
    waveforms = torch.arange(24, dtype=torch.float32).reshape(2, 2, 6).requires_grad_()
    data = {"prediction": waveforms, "lengths": torch.tensor([5, 3])}
    model.log_data(logger, 7, data)
    indices = selection if isinstance(selection, list) else [selection]
    for (clip, options), index in zip(audio, indices, strict=True):
        np.testing.assert_array_equal(clip, waveforms[index, :, :data["lengths"][index]].detach().numpy().T)
        assert options == {"sample_rate": 16000}
    expected = ["clip"] * len(indices) if isinstance(selection, list) else "clip"
    assert calls == [({"samples": 32, "global_step": 7, "val/prediction": expected},
                      {"step": 7, "commit": False})]
    assert waveforms.shape == (2, 2, 6) and waveforms.requires_grad

    # Other backends can use exactly the same model config without W&B calls.
    model.log_data(LOGGER_MODULE.ConsoleLogger(), 7, data)
    runtime = types.SimpleNamespace(is_rank_zero=False)
    model.log_data(LOGGER_MODULE.DistributedLogger(logger, runtime), 7, {})
    assert len(audio) == len(indices)


def test_hydra_media_whole_table_and_nested_options(monkeypatch):
    from omegaconf import OmegaConf

    run = _FakeRun()
    calls = []
    run.log = lambda data, **kwargs: calls.append(data)
    sdk = _fake_wandb(run)
    sdk.Table = lambda **kwargs: kwargs
    sdk.Image = lambda value, **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    logger = WandbLogger()
    rows = [[1, "one"], [2, "two"]]
    logger.log_media(data_dict={"rows": rows}, key="rows", sample_index=None,
                     media_type="table", columns=OmegaConf.create(["id", "text"]))
    assert calls[-1]["rows"] == {"data": rows, "columns": ["id", "text"]}
    logger.log_media(data_dict={"pictures": torch.zeros(1, 3, 4, 4)},
                     key="pictures", media_type="image",
                     boxes=OmegaConf.create({"truth": {"class_labels": {0: "cat"}}}))
    assert type(calls[-1]["pictures"]["boxes"]) is dict
    assert calls[-1]["pictures"]["boxes"]["truth"]["class_labels"] == {0: "cat"}


@pytest.mark.parametrize("options,error", [
    ({"key": "missing"}, KeyError),
    ({"key": "audio", "sample_index": 2}, IndexError),
    ({"key": "audio", "sample_index": -1}, ValueError),
    ({"key": "audio", "value": [1]}, ValueError),
    ({"key": "audio", "time_axis": 3, "len_key": "lengths"}, ValueError),
    ({"key": "audio", "len_key": "too_long"}, ValueError),
    ({"key": "audio", "len_key": "fractional"}, TypeError),
    ({"len_key": "lengths"}, ValueError),
])
def test_hydra_media_invalid_selection_fails_before_logging(monkeypatch, options, error):
    run = _FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run))
    logger = WandbLogger()
    with pytest.raises(error):
        logger.log_media(data_dict={"audio": torch.zeros(1, 4), "lengths": [2],
                                    "too_long": [5], "fractional": [2.5]}, **options)
    assert not run.logged


@pytest.mark.parametrize("shape,time_axis,expected", [
    ((1, 6), -1, (3,)),
    ((1, 1, 6), -1, (3, 1)),
    ((1, 6, 2), 0, (3, 2)),
])
def test_hydra_audio_layouts(monkeypatch, shape, time_axis, expected):
    run = _FakeRun()
    sdk = _fake_wandb(run)
    clips = []
    sdk.Audio = lambda value, **kwargs: clips.append(value)
    run.log = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    logger = WandbLogger()
    logger.log_media(data_dict={"x": torch.zeros(shape), "n": [3]}, key="x",
                     len_key="n", time_axis=time_axis, media_type="audio", sample_rate=16000)
    assert clips[0].shape == expected


@pytest.mark.parametrize("selection,media_type,key,error", [
    ([], "audio", "x", ValueError),
    ([0, -1], "audio", "x", ValueError),
    ([0, True], "audio", "x", ValueError),
    ([0, 1.5], "audio", "x", ValueError),
    ([0, None], "audio", "x", ValueError),
    ([0, 2], "audio", "x", IndexError),
    ([0], "table", "x", ValueError),
    ([0], "graph", "x", ValueError),
    ([0], "plotly", "x", ValueError),
    ([0], "histogram", "x", ValueError),
    ([0], None, "x", ValueError),
    ([0], "audio", None, ValueError),
])
def test_sample_list_rejects_invalid_requests_without_partial_logs(monkeypatch, selection, media_type, key, error):
    run = _FakeRun()
    sdk = _fake_wandb(run)
    constructed = []
    sdk.Audio = lambda *args, **kwargs: constructed.append(args)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    logger = WandbLogger()
    with pytest.raises(error):
        logger.log_media("example", data_dict={"x": torch.zeros(2, 4)}, key=key,
                         sample_index=selection, media_type=media_type)
    assert not constructed
    assert not run.logged


def test_sample_list_invalid_later_length_is_atomic(monkeypatch):
    run = _FakeRun()
    sdk = _fake_wandb(run)
    constructed = []
    sdk.Audio = lambda *args, **kwargs: constructed.append(args)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    with pytest.raises(ValueError, match="length"):
        WandbLogger().log_media(data_dict={"x": torch.zeros(2, 4), "n": [2, 5]},
                               key="x", len_key="n", sample_index=[0, 1], media_type="audio")
    assert not constructed and not run.logged


def test_sample_list_images_preserve_order_and_options(monkeypatch):
    from omegaconf import OmegaConf
    run = _FakeRun()
    calls = []
    run.log = lambda data, **kwargs: calls.append(data)
    sdk = _fake_wandb(run)
    sdk.Image = lambda value, **kwargs: (value, kwargs)
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    WandbLogger().log_media(data_dict={"images": ["a.png", "b.png", "c.png"]},
                           key="images", sample_index=OmegaConf.create([2, 0]),
                           media_type="image", caption="prediction")
    assert calls == [{"images": [("c.png", {"caption": "prediction"}),
                                  ("a.png", {"caption": "prediction"})]}]
