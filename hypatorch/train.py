import os
import signal
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from .core import Model
from .distributed import DistributedRuntime
from .logger import DistributedLogger
from .utils import shared_dict, update_output


class Trainer:
    @staticmethod
    def _normalize_requested_devices(devices):
        if isinstance(devices, str) and devices.isdigit():
            return int(devices)
        return devices

    def __init__(
        self,
        max_epochs=None,
        device=None,
        accelerator=None,
        devices=1,
        strategy=None,
        precision=None,
        log_every_n_steps=25,
        logger=None,
        seed=1234,
        float32_matmul_precision="high",
        compile_model=False,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=True,
        autocast_dtype=None,
        grad_accum_steps=1,
        max_samples=None,
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
        checkpoint_interval_seconds=None,
        checkpoint_artifact_path="checkpoints",
        save_last=True,
        **kwargs,
    ):
        distributed_backend = kwargs.pop("_distributed_backend", None)
        allow_cpu_distributed = bool(kwargs.pop("_allow_cpu_distributed", False))
        del kwargs

        devices = self._normalize_requested_devices(devices)
        self._validate_execution_config(
            devices=devices,
            strategy=strategy,
            accelerator=accelerator,
            compile_model=compile_model,
            allow_cpu_distributed=allow_cpu_distributed,
        )
        self.devices = devices
        self.strategy = strategy
        self.distributed = DistributedRuntime(
            devices=devices if isinstance(devices, int) else 1,
            accelerator=accelerator,
            strategy=strategy,
            backend=distributed_backend,
            allow_cpu=allow_cpu_distributed,
            ddp_find_unused_parameters=ddp_find_unused_parameters,
            ddp_broadcast_buffers=ddp_broadcast_buffers,
        )

        # Device setup
        self.device = self._resolve_device(
            device=device,
            accelerator=accelerator,
            local_rank=self.distributed.local_rank,
        )
        self.distributed.initialize(device_type=self.device.type)

        # Computation setup
        self.float32_matmul_precision = float32_matmul_precision
        self.seed = seed
        self.autocast_dtype = self._resolve_autocast_dtype(
            autocast_dtype=autocast_dtype,
            precision=precision,
        )
        self.compile_model = compile_model

        # Logging setup
        self.log_every_n_steps = log_every_n_steps
        self.logger = logger

        # Optimizer setup
        self.grad_accum_steps = grad_accum_steps
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        if self.gradient_clip_val is not None and self.gradient_clip_algorithm not in {
            "norm",
            "value",
        }:
            raise ValueError(
                "gradient_clip_algorithm must be either 'norm' or 'value'."
            )

        # Termination / checkpointing
        self.max_epochs = max_epochs
        self.max_samples = None if max_samples is None or max_samples < 0 else max_samples
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        self.checkpoint_artifact_path = checkpoint_artifact_path
        self.save_last = save_last

        # Step state
        self.global_step = 0
        self.train_step = 0
        self.val_step = 0
        self.epoch_idx = 0
        self.train_samples = 0

        # Training state
        self.optimizers = None
        self.schedulers = None
        self.gradient_clipping = None
        self.model = None
        self.state_model = None
        self.rng_state_dict = None
        self.last_checkpoint_path = None

        # Internal execution state
        self.should_stop = False
        self._stop_reason = None
        self._last_checkpoint_monotonic = None
        self._signal_handlers = {}

    def _validate_execution_config(
        self,
        devices=1,
        strategy=None,
        accelerator=None,
        compile_model=False,
        allow_cpu_distributed=False,
    ):
        if isinstance(devices, (list, tuple)):
            if len(devices) > 1:
                raise NotImplementedError("List-style device selection is not supported.")

        if isinstance(devices, int) and devices > 1:
            if accelerator != "gpu" and not allow_cpu_distributed:
                raise NotImplementedError(
                    "Multi-device training currently requires accelerator='gpu'."
                )
            if strategy not in (None, "auto", "ddp"):
                raise NotImplementedError(
                    "Multi-device training currently supports only strategy=None, 'auto', or 'ddp'."
                )
            if compile_model:
                raise NotImplementedError(
                    "compile_model is not supported with distributed training."
                )
            return

        if strategy not in (None, "auto", "single_device"):
            raise NotImplementedError(
                "Distributed strategies are not supported in hypatorch."
            )

    def _resolve_device(self, device=None, accelerator=None, local_rank=0):
        if isinstance(device, torch.device):
            return device

        if isinstance(device, str) and device:
            return torch.device(device)

        if accelerator in (None, "auto"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if accelerator == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested accelerator='gpu' but CUDA is not available.")
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)

        if accelerator == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise RuntimeError("Requested accelerator='mps' but MPS is not available.")
            return torch.device("mps")

        if accelerator == "cpu":
            return torch.device("cpu")

        raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _resolve_autocast_dtype(self, autocast_dtype=None, precision=None):
        if autocast_dtype is None:
            precision_map = {
                16: torch.float16,
                "16": torch.float16,
                "16-mixed": torch.float16,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "bf16-mixed": torch.bfloat16,
            }
            autocast_dtype = precision_map.get(precision)

        if isinstance(autocast_dtype, str):
            normalized = autocast_dtype.replace("-", "_")
            alias_map = {
                "fp16": "float16",
                "float16": "float16",
                "half": "float16",
                "bf16": "bfloat16",
                "bfloat16": "bfloat16",
            }
            normalized = alias_map.get(normalized, normalized)
            autocast_dtype = getattr(torch, normalized)

        return autocast_dtype

    def _stateful_model(self, model):
        candidate = getattr(model, "_orig_mod", model)
        if candidate.__class__.__name__ == "DistributedDataParallel":
            candidate = candidate.module
        return candidate

    def _wrap_logger(self, logger):
        if logger is None or not self.distributed.enabled:
            return logger
        if isinstance(logger, DistributedLogger):
            return logger
        return DistributedLogger(logger, self.distributed)

    def _reset_steps(self):
        self.global_step = 0
        self.train_step = 0
        self.val_step = 0
        self.epoch_idx = 0
        self.train_samples = 0
        self.should_stop = False
        self._stop_reason = None
        self.last_checkpoint_path = None

    def _reset_random_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def get_rng_state_dict(self):
        rng_state_dict = {
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state_dict["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
        return rng_state_dict

    def set_rng_state_dict(self, rng_state_dict):
        torch_rng_state = rng_state_dict["torch_rng_state"].to(
            device="cpu",
            dtype=torch.uint8,
        )
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch_cuda_rng_state = [
                state.to(device="cpu", dtype=torch.uint8)
                for state in rng_state_dict["torch_cuda_rng_state"]
            ]
            torch.cuda.set_rng_state_all(torch_cuda_rng_state)

    def _forward_context(self, mode):
        del mode
        forward_context = ExitStack()

        if self.autocast_dtype is not None:
            forward_context.enter_context(
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                )
            )

        return forward_context

    def _input_to_device(self, input_dict):
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].to(self.device)

        return input_dict

    def _infer_batch_size(self, input_dict):
        for value in input_dict.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)) and value:
                return len(value)
        return 1

    def _is_optimizer_step_complete(self, epoch_step):
        return (epoch_step + 1) % self.grad_accum_steps == 0

    def _backward_context(self, model, epoch_step):
        if hasattr(model, "no_sync") and not self._is_optimizer_step_complete(epoch_step):
            return model.no_sync()
        return nullcontext()

    def _next_step(self, mode):
        current_step = self.train_step if mode == "train" else self.val_step
        current_global_step = self.global_step

        if mode == "train":
            self.train_step += 1
        else:
            self.val_step += 1

        self.global_step += 1

        return current_step, current_global_step

    def _verify_checkpoint(self, checkpoint):
        if "hypatorch_version" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain 'hypatorch_version'. This might be an incompatible checkpoint."
            )

        if "state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain 'state_dict'. This is required to load the model."
            )

        required_steps = {"global_step", "train_step", "val_step", "epoch_idx"}
        if not required_steps.issubset(checkpoint):
            raise ValueError(
                "Checkpoint does not contain step information. This is required for training state."
            )

    def save_checkpoint(
        self,
        name,
        model,
        optimizers=None,
        schedulers=None,
        chkpt_dir=None,
    ):
        from . import __version__

        checkpoint = {}
        checkpoint["hypatorch_version"] = __version__
        checkpoint["state_dict"] = self._stateful_model(model).checkpoint_state_dict()
        if optimizers:
            checkpoint["optimizers"] = {k: v.state_dict() for k, v in optimizers.items()}
        if schedulers:
            checkpoint["lr_schedulers"] = {
                k: v.state_dict() for k, v in schedulers.items()
            }
        checkpoint["global_step"] = self.global_step
        checkpoint["train_step"] = self.train_step
        checkpoint["val_step"] = self.val_step
        checkpoint["epoch_idx"] = self.epoch_idx
        checkpoint["train_samples"] = self.train_samples
        checkpoint["rng_state"] = self.get_rng_state_dict()

        if chkpt_dir:
            os.makedirs(chkpt_dir, exist_ok=True)
            name = os.path.join(chkpt_dir, name)

        torch.save(checkpoint, name)
        self.last_checkpoint_path = name
        return name

    def load_checkpoint(
        self,
        name,
        model,
        optimizers=None,
        schedulers=None,
        chkpt_dir=None,
        strict=True,
        set_rng_state=False,
    ):
        if chkpt_dir:
            name = os.path.join(chkpt_dir, name)
        checkpoint = torch.load(name, map_location=self.device, weights_only=False)

        self._verify_checkpoint(checkpoint)

        self._stateful_model(model).load_checkpoint_state_dict(
            checkpoint["state_dict"],
            strict=strict,
        )

        if optimizers:
            if "optimizers" not in checkpoint:
                raise ValueError(
                    "Checkpoint does not contain 'optimizers'. This is required to load the optimizers."
                )
            for k, v in optimizers.items():
                if k not in checkpoint["optimizers"]:
                    raise ValueError(f"Optimizer {k} not found in checkpoint")

                v.load_state_dict(checkpoint["optimizers"][k])

        if schedulers:
            if "lr_schedulers" not in checkpoint:
                raise ValueError(
                    "Checkpoint does not contain 'lr_schedulers'. This is required to load the learning rate schedulers."
                )
            for k, v in schedulers.items():
                if k not in checkpoint["lr_schedulers"]:
                    raise ValueError(f"LR Scheduler {k} not found in checkpoint")

                v.load_state_dict(checkpoint["lr_schedulers"][k])

        self.global_step = checkpoint["global_step"]
        self.train_step = checkpoint["train_step"]
        self.val_step = checkpoint["val_step"]
        self.epoch_idx = checkpoint["epoch_idx"]
        self.train_samples = checkpoint.get("train_samples", 0)
        self.last_checkpoint_path = name

        if set_rng_state:
            if "rng_state" not in checkpoint:
                raise ValueError(
                    "Checkpoint does not contain 'rng_state'. This is required to set the random state."
                )
            self.set_rng_state_dict(checkpoint["rng_state"])

    def _apply_trainer_gradient_clipping(self, optimizer):
        if self.gradient_clip_val is None:
            return

        parameters = []
        for group in optimizer.param_groups:
            parameters.extend(
                param for param in group["params"] if getattr(param, "grad", None) is not None
            )

        if not parameters:
            return

        if self.gradient_clip_algorithm == "value":
            torch.nn.utils.clip_grad_value_(parameters, self.gradient_clip_val)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_val)

    def _optimizer_step(
        self,
        operation_name,
        optimizers=None,
        schedulers=None,
        gradient_clipping=None,
    ):
        optimizers = self.optimizers if optimizers is None else optimizers
        schedulers = self.schedulers if schedulers is None else schedulers
        gradient_clipping = (
            self.gradient_clipping if gradient_clipping is None else gradient_clipping
        )

        opt = optimizers[operation_name]

        if gradient_clipping and operation_name in gradient_clipping:
            gradient_clipping[operation_name]()

        self._apply_trainer_gradient_clipping(opt)

        opt.step()

        if schedulers and operation_name in schedulers:
            schedulers[operation_name].step_done()

        opt.zero_grad()

    def _flush_pending_gradients(self):
        if not self.optimizers:
            return

        state_model = self._stateful_model(self.model)
        for operation_name in state_model.operations.keys():
            optimizer = self.optimizers[operation_name]
            has_grad = any(
                param.grad is not None for group in optimizer.param_groups for param in group["params"]
            )
            if has_grad:
                self._optimizer_step(operation_name)

    def _maybe_request_stop_for_samples(self):
        if self.max_samples is not None and self.train_samples >= self.max_samples:
            self.should_stop = True
            self._stop_reason = "max_samples"

    def _periodic_checkpoint_name(self):
        return f"epoch={self.epoch_idx:04d}-step={self.global_step:08d}.ckpt"

    def _log_checkpoint_artifact(self, checkpoint_file, logger):
        if logger is None or not hasattr(logger, "log_artifact"):
            return
        logger.log_artifact(
            checkpoint_file,
            artifact_path=self.checkpoint_artifact_path,
        )

    def _save_periodic_checkpoint(self, logger=None, checkpoint_path=None):
        if self.distributed.enabled and not self.distributed.is_rank_zero:
            return
        checkpoint_file = self.save_checkpoint(
            self._periodic_checkpoint_name(),
            self.model,
            self.optimizers,
            self.schedulers,
            chkpt_dir=checkpoint_path,
        )
        self._last_checkpoint_monotonic = time.monotonic()
        self._log_checkpoint_artifact(checkpoint_file, logger)

    def _maybe_save_periodic_checkpoint(self, logger=None, checkpoint_path=None):
        if self.checkpoint_interval_seconds is None or checkpoint_path is None:
            return

        now = time.monotonic()
        if self._last_checkpoint_monotonic is None:
            self._last_checkpoint_monotonic = now
            return

        if now - self._last_checkpoint_monotonic >= self.checkpoint_interval_seconds:
            self._save_periodic_checkpoint(logger=logger, checkpoint_path=checkpoint_path)

    def _finalize_last_checkpoint(self, logger=None, checkpoint_path=None):
        if not self.save_last:
            return None
        if self.distributed.enabled and not self.distributed.is_rank_zero:
            return None

        checkpoint_file = self.save_checkpoint(
            "last.ckpt",
            self.model,
            self.optimizers,
            self.schedulers,
            chkpt_dir=checkpoint_path,
        )
        self._log_checkpoint_artifact(checkpoint_file, logger)
        return checkpoint_file

    def _request_stop(self, reason):
        self.should_stop = True
        self._stop_reason = reason

    @contextmanager
    def _signal_handler_context(self):
        previous_handlers = {}

        def _handler(signum, _frame):
            try:
                signame = signal.Signals(signum).name
            except ValueError:
                signame = str(signum)
            self._request_stop(f"signal:{signame}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _handler)
            except (ValueError, OSError, RuntimeError):
                continue

        try:
            yield
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

    def step(
        self,
        mode,
        model,
        input_dict,
        optimizers=None,
        schedulers=None,
        gradient_clipping=None,
        logger=None,
        epoch_step=0,
    ):
        step, global_step = self._next_step(mode)

        if logger:
            logger.log_value(f"{mode}_step", step)
            logger.log_value("global_step", global_step)

        input_dict = self._input_to_device(input_dict)
        output_dict = {}
        metrics = {}
        state_model = self._stateful_model(model)

        for operation_name in state_model.operations.keys():
            with self._forward_context(mode):
                operation_output = model(
                    input_dict=shared_dict(input_dict, output_dict),
                    operation_name=operation_name,
                    mode=mode,
                )

                update_output(
                    operation_output,
                    output_dict,
                    operation_name,
                )

                loss = state_model.compute_loss(
                    x=shared_dict(input_dict, output_dict),
                    operation_name=operation_name,
                    mode=mode,
                    logger=logger,
                )

                if loss is not None and optimizers:
                    loss = loss / self.grad_accum_steps
                    with self._backward_context(model, epoch_step):
                        loss.backward()

            if optimizers and self._is_optimizer_step_complete(epoch_step):
                self._optimizer_step(
                    operation_name,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    gradient_clipping=gradient_clipping,
                )

            metrics = state_model.compute_metrics(
                x=shared_dict(input_dict, output_dict),
                operation_name=operation_name,
                mode=mode,
                logger=logger,
            )

        if logger:
            logger.step_done()

        return output_dict, metrics, step, global_step

    def epoch(
        self,
        mode,
        model,
        epoch,
        dataset,
        optimizers=None,
        schedulers=None,
        gradient_clipping=None,
        logger=None,
        checkpoint_path=None,
    ):
        state_model = self._stateful_model(model)
        operations = state_model.operations.keys()

        if optimizers:
            for operation_name in operations:
                optimizers[operation_name].zero_grad()

        if logger:
            logger.log_value(f"{mode}_epoch", epoch)

        with self.distributed.join_context(model, enable=mode == "train"):
            for epoch_step, input_dict in enumerate(dataset):
                batch_size = self._infer_batch_size(input_dict) if mode == "train" else 0
                output_dict, metrics, step, global_step = self.step(
                    mode=mode,
                    model=model,
                    input_dict=input_dict,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    gradient_clipping=gradient_clipping,
                    logger=logger,
                    epoch_step=epoch_step,
                )
                del metrics, step

                if logger and epoch_step == 0 and mode == "val":
                    state_model.log_data(
                        data_dict=shared_dict(input_dict, output_dict),
                        step=global_step,
                        logger=logger,
                    )

                if mode == "train":
                    self.train_samples += batch_size
                    self._maybe_request_stop_for_samples()
                    self._maybe_save_periodic_checkpoint(
                        logger=logger,
                        checkpoint_path=checkpoint_path,
                    )

                if self.should_stop:
                    break

            if mode == "train" and optimizers:
                self._flush_pending_gradients()

        for operation_name in operations:
            if schedulers and operation_name in schedulers:
                schedulers[operation_name].epoch_done()

        if logger:
            logger.epoch_done()

    @staticmethod
    def _safe_len(dataset):
        if isinstance(dataset, DataLoader) and isinstance(dataset.dataset, IterableDataset):
            return None
        if isinstance(dataset, IterableDataset):
            return None
        if not hasattr(dataset, "__len__"):
            return None
        try:
            return len(dataset)
        except TypeError:
            return None

    def _prepare_model_training(self, model: Model):
        model.to(self.device)
        torch.set_float32_matmul_precision(self.float32_matmul_precision)

        self.optimizers, self.schedulers, self.gradient_clipping = (
            model.configure_optimizers()
        )

        self.state_model = model
        prepared_model = torch.compile(model) if self.compile_model else model
        self.model = self.distributed.wrap_model(prepared_model)

    @staticmethod
    def _loader_kwargs_from_dataloader(loader):
        kwargs = {
            "batch_size": loader.batch_size,
            "num_workers": loader.num_workers,
            "collate_fn": loader.collate_fn,
            "pin_memory": loader.pin_memory,
            "drop_last": loader.drop_last,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "multiprocessing_context": loader.multiprocessing_context,
            "generator": loader.generator,
        }
        prefetch_factor = getattr(loader, "prefetch_factor", None)
        if prefetch_factor is not None and loader.num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
        persistent_workers = getattr(loader, "persistent_workers", False)
        if loader.num_workers > 0:
            kwargs["persistent_workers"] = persistent_workers
        pin_memory_device = getattr(loader, "pin_memory_device", "")
        if pin_memory_device:
            kwargs["pin_memory_device"] = pin_memory_device
        if hasattr(loader, "in_order"):
            kwargs["in_order"] = loader.in_order
        return kwargs

    def _iterable_loader_from_dataloader(self, loader):
        kwargs = self._loader_kwargs_from_dataloader(loader)
        return DataLoader(
            loader.dataset,
            shuffle=False,
            **kwargs,
        )

    def _distributed_loader_from_dataloader(self, loader, *, shuffle, epoch):
        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=self.distributed.world_size,
            rank=self.distributed.rank,
            shuffle=shuffle,
            drop_last=loader.drop_last,
        )
        sampler.set_epoch(epoch)
        kwargs = self._loader_kwargs_from_dataloader(loader)
        kwargs["sampler"] = sampler
        return DataLoader(loader.dataset, **kwargs)

    def _as_dataloader(self, dataset_or_loader, *, shuffle, loader_args, epoch):
        if isinstance(dataset_or_loader, DataLoader):
            if isinstance(dataset_or_loader.dataset, IterableDataset):
                return self._iterable_loader_from_dataloader(dataset_or_loader)
            if self.distributed.enabled:
                sampler = dataset_or_loader.sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)
                    return dataset_or_loader
                return self._distributed_loader_from_dataloader(
                    dataset_or_loader,
                    shuffle=shuffle,
                    epoch=epoch,
                )
            return dataset_or_loader
        resolved_loader_args = dict(loader_args or {})
        is_iterable_dataset = isinstance(dataset_or_loader, IterableDataset)
        if self.distributed.enabled:
            if is_iterable_dataset:
                if resolved_loader_args.get("sampler") is not None:
                    raise RuntimeError(
                        "Distributed iterable datasets do not support explicit loader_args['sampler']."
                    )
                return DataLoader(
                    dataset_or_loader,
                    shuffle=False,
                    **resolved_loader_args,
                )
            if resolved_loader_args.get("sampler") is not None:
                raise RuntimeError(
                    "Distributed training does not support explicit loader_args['sampler']."
                )
            sampler = DistributedSampler(
                dataset_or_loader,
                num_replicas=self.distributed.world_size,
                rank=self.distributed.rank,
                shuffle=shuffle,
                drop_last=bool(resolved_loader_args.get("drop_last", False)),
            )
            sampler.set_epoch(epoch)
            resolved_loader_args["sampler"] = sampler
            shuffle = False
        return DataLoader(
            dataset_or_loader,
            shuffle=shuffle,
            **resolved_loader_args,
        )

    def _training_loop(
        self,
        train_dataset,
        loader_args,
        val_dataset=None,
        logger=None,
        checkpoint_path=None,
    ):
        logger = logger or self.logger

        max_epochs = self.max_epochs
        if max_epochs is None:
            max_epochs = float("inf")

        with self._signal_handler_context():
            while self.epoch_idx < max_epochs and not self.should_stop:
                current_epoch = self.epoch_idx

                if val_dataset is not None:
                    if not self.distributed.enabled or self.distributed.is_rank_zero:
                        self.model.eval()
                        validation_model = (
                            self.state_model if self.distributed.enabled else self.model
                        )
                        val_loader = self._as_dataloader(
                            val_dataset,
                            shuffle=False,
                            loader_args=loader_args,
                            epoch=current_epoch,
                        )
                        self.epoch(
                            mode="val",
                            model=validation_model,
                            epoch=current_epoch,
                            dataset=val_loader,
                            logger=logger,
                            checkpoint_path=checkpoint_path,
                        )
                    if self.distributed.enabled:
                        self.distributed.barrier()

                if self.should_stop:
                    break

                self.model.train()
                train_loader = self._as_dataloader(
                    train_dataset,
                    shuffle=True,
                    loader_args=loader_args,
                    epoch=current_epoch,
                )
                self.epoch(
                    mode="train",
                    model=self.model,
                    epoch=current_epoch,
                    dataset=train_loader,
                    optimizers=self.optimizers,
                    schedulers=self.schedulers,
                    gradient_clipping=self.gradient_clipping,
                    logger=logger,
                    checkpoint_path=checkpoint_path,
                )

                self.epoch_idx = current_epoch + 1

        self._finalize_last_checkpoint(logger=logger, checkpoint_path=checkpoint_path)
        self.distributed.barrier()
        self.rng_state_dict = self.get_rng_state_dict()

    def train(
        self,
        model: Model,
        train_dataset,
        loader_args,
        val_dataset=None,
        logger=None,
        checkpoint_path=None,
    ):
        self._reset_random_seed()
        self._reset_steps()
        self._prepare_model_training(model)
        logger = self._wrap_logger(logger or self.logger)
        self._training_loop(
            train_dataset=train_dataset,
            loader_args=loader_args,
            val_dataset=val_dataset,
            logger=logger,
            checkpoint_path=checkpoint_path,
        )

    def resume_training(
        self,
        model: Model,
        chkpt_name: str,
        train_dataset,
        loader_args,
        val_dataset=None,
        logger=None,
        checkpoint_path=None,
        strict=True,
    ):
        self.should_stop = False
        self._stop_reason = None
        self.last_checkpoint_path = None
        self._prepare_model_training(model)
        self.load_checkpoint(
            name=chkpt_name,
            model=model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            chkpt_dir=checkpoint_path,
            strict=strict,
            set_rng_state=True,
        )
        logger = self._wrap_logger(logger or self.logger)
        self._training_loop(
            train_dataset=train_dataset,
            loader_args=loader_args,
            val_dataset=val_dataset,
            logger=logger,
            checkpoint_path=checkpoint_path,
        )

    def continue_training(
        self,
        train_dataset,
        loader_args,
        val_dataset=None,
        logger=None,
        checkpoint_path=None,
    ):
        if not self.optimizers or not self.model or not self.rng_state_dict:
            raise ValueError(
                "No training can be continued. Please call train() or resume_training() before contine_training()"
            )

        self.should_stop = False
        self._stop_reason = None
        self.last_checkpoint_path = None
        self.set_rng_state_dict(self.rng_state_dict)
        logger = self._wrap_logger(logger or self.logger)
        self._training_loop(
            train_dataset=train_dataset,
            loader_args=loader_args,
            val_dataset=val_dataset,
            logger=logger,
            checkpoint_path=checkpoint_path,
        )
