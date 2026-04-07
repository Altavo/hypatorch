from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.distributed as dist


_REQUIRED_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


def _env_int(env: Mapping[str, str], name: str) -> int | None:
    raw = env.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Distributed environment value {name} must be an integer.") from exc


@dataclass(frozen=True)
class DistributedEnvironment:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int | None = None


def detect_distributed_environment(
    env: Mapping[str, str] | None = None,
) -> DistributedEnvironment | None:
    current_env = os.environ if env is None else env
    present = [name for name in _REQUIRED_ENV_KEYS if current_env.get(name)]
    if not present:
        return None
    if len(present) != len(_REQUIRED_ENV_KEYS):
        missing = ", ".join(name for name in _REQUIRED_ENV_KEYS if name not in present)
        raise RuntimeError(
            "Incomplete distributed environment detected; missing: " + missing
        )

    rank = _env_int(current_env, "RANK")
    world_size = _env_int(current_env, "WORLD_SIZE")
    local_rank = _env_int(current_env, "LOCAL_RANK")
    local_world_size = _env_int(current_env, "LOCAL_WORLD_SIZE")
    if rank is None or world_size is None or local_rank is None:
        raise RuntimeError("Distributed environment is missing required rank values.")
    if rank < 0 or local_rank < 0 or world_size < 1:
        raise RuntimeError("Distributed environment values must be non-negative.")
    return DistributedEnvironment(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )


class DistributedRuntime:
    def __init__(
        self,
        *,
        devices: int = 1,
        accelerator: str | None = None,
        strategy: str | None = None,
        backend: str | None = None,
        allow_cpu: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.devices = devices
        self.accelerator = accelerator
        self.strategy = strategy
        self.backend = backend
        self.allow_cpu = allow_cpu
        self.environment = detect_distributed_environment(env)
        self.rank = self.environment.rank if self.environment is not None else 0
        self.world_size = (
            self.environment.world_size if self.environment is not None else 1
        )
        self.local_rank = (
            self.environment.local_rank if self.environment is not None else 0
        )
        self.local_world_size = (
            self.environment.local_world_size if self.environment is not None else None
        )
        self.enabled = devices > 1
        self._initialized_here = False

        if not self.enabled:
            if self.environment is not None and self.world_size > 1:
                raise RuntimeError(
                    "Distributed environment detected but trainer.devices is not greater than 1."
                )
            return

        if self.environment is None:
            raise RuntimeError(
                "Distributed training requires a torchrun environment. "
                "Launch with `python -m torch.distributed.run --standalone --nproc_per_node=<devices>`."
            )
        if self.world_size != self.devices:
            raise RuntimeError(
                "Only single-node distributed training is supported; WORLD_SIZE must equal trainer.devices."
            )
        if self.local_world_size is not None and self.local_world_size != self.devices:
            raise RuntimeError(
                "Only single-node distributed training is supported; LOCAL_WORLD_SIZE must equal trainer.devices."
            )
        if accelerator != "gpu" and not allow_cpu:
            raise NotImplementedError(
                "Distributed training currently requires accelerator='gpu'."
            )

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def initialize(self, *, device_type: str) -> None:
        if not self.enabled:
            return
        if dist.is_initialized():
            return
        backend = self.backend
        if backend is None:
            backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        self._initialized_here = True

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.enabled:
            return model
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(model, DistributedDataParallel):
            return model
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group must be initialized before wrapping the model.")
        if next(model.parameters(), None) is None:
            raise RuntimeError("DistributedDataParallel requires a model with parameters.")
        if self.accelerator == "gpu":
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        return DistributedDataParallel(model)

    def barrier(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def reduce_mean_scalar(self, value: float | int) -> float:
        if not self.enabled:
            return float(value)
        device = "cpu"
        if self.accelerator == "gpu":
            device = torch.device("cuda", self.local_rank)
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(self.world_size)
        return float(tensor.item())

    def cleanup(self) -> None:
        if self._initialized_here and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized_here = False
