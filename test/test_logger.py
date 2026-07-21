import importlib.util
import sys
import types
from pathlib import Path


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

        logger.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
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
