import json
import multiprocessing
import os
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

import hypatorch
from hypatorch.distributed import DistributedRuntime
from hypatorch.logger import ConsoleLogger

from shared import add_path


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.ones(1, 28, 28) * idx,
            "class": idx % 10,
        }


class TestIterableDataset(IterableDataset):
    def __iter__(self):
        for idx in range(8):
            yield {"image": torch.ones(1, 28, 28) * idx, "class": idx % 10}


class UnevenRankIterableDataset(IterableDataset):
    def __iter__(self):
        rank = int(os.environ.get("RANK", "0"))
        size = 12 if rank == 0 else 8
        for idx in range(size):
            yield {"image": torch.ones(1, 28, 28) * idx, "class": idx % 10}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _set_dist_env(rank: int, world_size: int, port: int) -> dict[str, str | None]:
    keys = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _example_training_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))


def _compose_example_model():
    training_path = _example_training_root()
    rel_config_dir = os.path.relpath(training_path, os.path.dirname(__file__))
    with initialize(
        config_path=os.path.join(rel_config_dir, "conf"),
        version_base="1.1",
    ):
        cfg = compose(config_name="config.yaml", overrides=["experiment=mnist_linear"])
    return training_path, cfg


def _runtime_worker(rank: int, port: int, output_dir: str) -> None:
    previous = _set_dist_env(rank, 2, port)
    try:
        runtime = DistributedRuntime(
            devices=2,
            accelerator="cpu",
            strategy="ddp",
            backend="gloo",
            allow_cpu=True,
        )
        runtime.initialize(device_type="cpu")
        runtime.barrier()
        reduced = runtime.reduce_mean_scalar(rank + 1)
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": runtime.rank,
                    "local_rank": runtime.local_rank,
                    "world_size": runtime.world_size,
                    "rank_zero": runtime.is_rank_zero,
                    "reduced": reduced,
                }
            ),
            encoding="utf-8",
        )
        runtime.cleanup()
    finally:
        _restore_env(previous)


def _ddp_train_worker(rank: int, port: int, checkpoint_dir: str) -> None:
    previous = _set_dist_env(rank, 2, port)
    try:
        training_path, cfg = _compose_example_model()
        dataset = TestDataset(32)
        with add_path(training_path):
            trainer = hypatorch.Trainer(
                max_epochs=1,
                accelerator="cpu",
                devices=2,
                strategy="ddp",
                save_last=True,
                _allow_cpu_distributed=True,
                _distributed_backend="gloo",
            )
            model = instantiate(cfg.model)
            model.register_buffer("sync_buffer", torch.zeros(1))
            trainer.train(
                model=model,
                train_dataset=dataset,
                val_dataset=dataset,
                loader_args={"batch_size": 8},
                checkpoint_path=checkpoint_dir,
            )
            trainer.distributed.cleanup()
    finally:
        _restore_env(previous)


def _ddp_resume_worker(rank: int, port: int, checkpoint_dir: str) -> None:
    previous = _set_dist_env(rank, 2, port)
    try:
        training_path, cfg = _compose_example_model()
        dataset = TestDataset(32)
        with add_path(training_path):
            trainer = hypatorch.Trainer(
                max_epochs=2,
                accelerator="cpu",
                devices=2,
                strategy="ddp",
                save_last=True,
                _allow_cpu_distributed=True,
                _distributed_backend="gloo",
            )
            model = instantiate(cfg.model)
            trainer.resume_training(
                model=model,
                chkpt_name="last.ckpt",
                train_dataset=dataset,
                val_dataset=dataset,
                loader_args={"batch_size": 8},
                checkpoint_path=checkpoint_dir,
                strict=True,
            )
            trainer.distributed.cleanup()
    finally:
        _restore_env(previous)


def _ddp_uneven_iterable_worker(rank: int, port: int, output_dir: str) -> None:
    previous = _set_dist_env(rank, 2, port)
    try:
        training_path, cfg = _compose_example_model()
        dataset = UnevenRankIterableDataset()
        with add_path(training_path):
            trainer = hypatorch.Trainer(
                max_epochs=1,
                accelerator="cpu",
                devices=2,
                strategy="ddp",
                save_last=False,
                logger=ConsoleLogger(log_every_n_steps=1),
                _allow_cpu_distributed=True,
                _distributed_backend="gloo",
            )
            model = instantiate(cfg.model)
            model.register_buffer("sync_buffer", torch.zeros(1))
            trainer.train(
                model=model,
                train_dataset=dataset,
                val_dataset=dataset,
                loader_args={"batch_size": 4, "drop_last": True},
            )
            Path(output_dir, f"uneven-rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "epoch_idx": trainer.epoch_idx,
                        "global_step": trainer.global_step,
                    }
                ),
                encoding="utf-8",
            )
            trainer.distributed.cleanup()
    finally:
        _restore_env(previous)


def _run_processes(worker, *, port: int, checkpoint_dir: str | None = None, output_dir: str | None = None):
    ctx = multiprocessing.get_context("spawn")
    processes = []
    for rank in range(2):
        kwargs = {"rank": rank, "port": port}
        if checkpoint_dir is not None:
            kwargs["checkpoint_dir"] = checkpoint_dir
        if output_dir is not None:
            kwargs["output_dir"] = output_dir
        process = ctx.Process(target=worker, kwargs=kwargs)
        process.start()
        processes.append(process)
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0


def test_validate_execution_config_accepts_gpu_ddp():
    trainer = hypatorch.Trainer.__new__(hypatorch.Trainer)
    trainer._validate_execution_config(
        devices=2,
        accelerator="gpu",
        strategy="ddp",
        compile_model=False,
    )


def test_validate_execution_config_rejects_invalid_multi_device_settings():
    trainer = hypatorch.Trainer.__new__(hypatorch.Trainer)
    with pytest.raises(NotImplementedError, match="accelerator='gpu'"):
        trainer._validate_execution_config(devices=2, accelerator="cpu", strategy="ddp")
    with pytest.raises(NotImplementedError, match="only strategy=None, 'auto', or 'ddp'"):
        trainer._validate_execution_config(devices=2, accelerator="gpu", strategy="fsdp")
    with pytest.raises(NotImplementedError, match="compile_model"):
        trainer._validate_execution_config(
            devices=2,
            accelerator="gpu",
            strategy="ddp",
            compile_model=True,
        )


def test_distributed_runtime_reports_rank_and_reduces_scalar(tmp_path):
    port = _free_port()
    _run_processes(_runtime_worker, port=port, output_dir=str(tmp_path))

    payloads = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert payloads[0]["rank_zero"] is True
    assert payloads[1]["rank_zero"] is False
    assert payloads[0]["world_size"] == payloads[1]["world_size"] == 2
    assert payloads[0]["reduced"] == pytest.approx(1.5)
    assert payloads[1]["reduced"] == pytest.approx(1.5)


def test_as_dataloader_uses_distributed_sampler_and_sets_epoch():
    trainer = hypatorch.Trainer.__new__(hypatorch.Trainer)
    trainer.distributed = SimpleNamespace(enabled=True, world_size=2, rank=1)

    loader = trainer._as_dataloader(
        TestDataset(32),
        shuffle=True,
        loader_args={"batch_size": 4},
        epoch=3,
    )

    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.num_replicas == 2
    assert loader.sampler.rank == 1
    assert loader.sampler.epoch == 3


def test_as_dataloader_rebuilds_prebuilt_loader_with_distributed_sampler():
    trainer = hypatorch.Trainer.__new__(hypatorch.Trainer)
    trainer.distributed = SimpleNamespace(enabled=True, world_size=2, rank=0)
    collate_fn = lambda batch: {"count": len(batch)}
    loader = DataLoader(
        TestDataset(16),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    rebuilt = trainer._as_dataloader(loader, shuffle=True, loader_args=None, epoch=2)

    assert rebuilt is not loader
    assert isinstance(rebuilt.sampler, DistributedSampler)
    assert rebuilt.sampler.rank == 0
    assert rebuilt.sampler.num_replicas == 2
    assert rebuilt.sampler.epoch == 2
    assert rebuilt.batch_size == 4
    assert rebuilt.collate_fn is collate_fn


def test_as_dataloader_keeps_iterable_loader_without_distributed_sampler():
    trainer = hypatorch.Trainer.__new__(hypatorch.Trainer)
    trainer.distributed = SimpleNamespace(enabled=True, world_size=2, rank=0)
    loader = DataLoader(TestIterableDataset(), batch_size=4)

    rebuilt = trainer._as_dataloader(loader, shuffle=True, loader_args=None, epoch=2)

    assert rebuilt is loader
    assert not isinstance(rebuilt.sampler, DistributedSampler)


def test_ddp_checkpoint_loads_in_single_process(tmp_path):
    port = _free_port()
    checkpoint_dir = str(tmp_path)
    _run_processes(_ddp_train_worker, port=port, checkpoint_dir=checkpoint_dir)

    checkpoint_path = tmp_path / "last.ckpt"
    assert checkpoint_path.exists()

    training_path, cfg = _compose_example_model()
    with add_path(training_path):
        trainer = hypatorch.Trainer(max_epochs=1)
        model = instantiate(cfg.model)
        trainer._prepare_model_training(model)
        trainer.load_checkpoint(
            "last.ckpt",
            model,
            optimizers=trainer.optimizers,
            schedulers=trainer.schedulers,
            chkpt_dir=checkpoint_dir,
            strict=True,
        )
        assert trainer.global_step > 0


def test_single_process_checkpoint_resumes_in_ddp(tmp_path):
    training_path, cfg = _compose_example_model()
    dataset = TestDataset(32)
    checkpoint_dir = str(tmp_path)

    with add_path(training_path):
        trainer = hypatorch.Trainer(max_epochs=1, save_last=True)
        model = instantiate(cfg.model)
        trainer.train(
            model=model,
            train_dataset=dataset,
            val_dataset=dataset,
            loader_args={"batch_size": 8},
            checkpoint_path=checkpoint_dir,
        )

    port = _free_port()
    _run_processes(_ddp_resume_worker, port=port, checkpoint_dir=checkpoint_dir)
    assert (tmp_path / "last.ckpt").exists()


def test_ddp_handles_uneven_iterable_inputs(tmp_path):
    port = _free_port()
    _run_processes(_ddp_uneven_iterable_worker, port=port, output_dir=str(tmp_path))

    payloads = [
        json.loads((tmp_path / f"uneven-rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert [payload["epoch_idx"] for payload in payloads] == [1, 1]
