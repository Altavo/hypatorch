from abc import ABC, abstractmethod
import html
from pathlib import Path

import torch


class DataLogger(ABC):

    def __init__(self, log_every_n_steps:int=1):
        self.log_every_n_steps = log_every_n_steps

        self._reset()

    def _reset(self):
        self._step_log = {}
        self._epoch_log = {}
        self._epoch_no_log = []
        self._epoch_step = 0

    def log_value(self, name:str, value:float|int):
        self._step_log[name] = value

    def log_images(self, *args, **kwargs):
        del args, kwargs

    def log_text(self, *args, **kwargs):
        del args, kwargs

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        del local_path, artifact_path

    def finalize(self, status: str):
        del status

    @abstractmethod
    def report_step(self):
        pass

    @abstractmethod
    def report_epoch(self):
        pass

    def _update_epoch_log(self):
        for k, v in self._step_log.items():
            if k in self._epoch_no_log:
                continue
            
            if isinstance(v, int):
                if k in self._epoch_log and self._epoch_log[k] != v:
                    self._epoch_no_log.append(k)
                    self._epoch_log.pop(k)
                else:
                    self._epoch_log[k] = v
            else:
                if k not in self._epoch_log:
                    self._epoch_log[k] = []
                self._epoch_log[k].append(v)

    def step_done(self):
        if self._epoch_step % self.log_every_n_steps == 0:
            self.report_step()

        self._epoch_step += 1

        self._update_epoch_log()

    def epoch_done(self):
        self.report_epoch()

        self._reset()    
    

    def step_items(self):
        # Generator that yields key value pairs
        for k, v in self._step_log.items():
            if not k.endswith("_step") and not k.endswith("_epoch"):
                k = f"{k}_step"
            
            yield k, v

    def epoch_items(self):
        # Generator that yields key value pairs
        for k, v in self._epoch_log.items():
            if not k.endswith("_step") and not k.endswith("_epoch"):
                k = f"{k}_epoch"
            
            if isinstance(v, list):
                v = sum(v) / len(v)

            yield k, v


class ConsoleLogger(DataLogger):
    def __init__(self, log_every_n_steps:int=1, float_precision:int=4):
        super().__init__(log_every_n_steps=log_every_n_steps)
        self.float_precision = float_precision


    def _format(self, value) -> str:
        if isinstance(value, int):
            return str(value)

        if isinstance(value, float):
            return f"{value:.{self.float_precision}f}"
        
        if isinstance(value, list):
            mean_value = sum(value) / len(value)
            return f"{mean_value:.{self.float_precision}f}"

    def report_step(self):
        # Create a string with | separated key value pairs
        log_str = " | ".join([f"{k}={self._format(v)}" for k, v in self.step_items()])
        print("step > " + log_str)


    def report_epoch(self):
        # Create a string with | separated key value pairs
        log_str = " | ".join([f"{k}={self._format(v)}" for k, v in self.epoch_items()])
        print("epoch > " + log_str)


class MLflowLogger(DataLogger):
    def __init__(self, log_every_n_steps: int = 1):
        super().__init__(log_every_n_steps=log_every_n_steps)
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflowLogger requires the optional 'mlflow' dependency."
            ) from exc
        self._mlflow = mlflow

    def _metric_step(self) -> int:
        for candidate in ("train_step", "val_step", "global_step"):
            value = self._step_log.get(candidate)
            if isinstance(value, int):
                return value
        return self._epoch_step

    def report_step(self):
        metrics = {}
        for key, value in self.step_items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        if metrics:
            self._mlflow.log_metrics(metrics, step=self._metric_step())

    def report_epoch(self):
        metrics = {}
        for key, value in self.epoch_items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        if metrics:
            self._mlflow.log_metrics(metrics, step=self._metric_step())

    def _plot_images(self, data_dict, log_image_keys):
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(len(log_image_keys), 1, figsize=(15, 10))
        if not isinstance(axes, (list, tuple)):
            try:
                axes = list(axes)
            except TypeError:
                axes = [axes]

        for index, spec in enumerate(log_image_keys):
            key = spec["key"]
            len_key = spec.get("len_key")
            output = data_dict[key][0].float().detach().cpu().squeeze().numpy()

            if len_key is not None:
                valid_length = data_dict[len_key][0]
                if len(output.shape) == 1:
                    output = output[:valid_length]
                elif len(output.shape) == 2:
                    output = output[..., :valid_length]

            if len(output.shape) == 1:
                axes[index].plot(output)
                axes[index].set_xlim(0, output.shape[0])
            else:
                axes[index].imshow(
                    output,
                    aspect="auto",
                    origin="lower",
                    interpolation="none",
                )
            axes[index].set_title(key)

        figure.tight_layout()
        return figure

    def log_images(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        if kwargs:
            data_dict = kwargs["data_dict"]
            global_step = kwargs["global_step"]
            log_image_keys = kwargs["log_image_keys"]
        else:
            _, data_dict, log_image_keys = args
            global_step = self._metric_step()

        if not log_image_keys:
            return

        figure = self._plot_images(data_dict, log_image_keys)
        self._mlflow.log_figure(figure, f"images/log_step_{global_step}.png")
        plt.close(figure)

    def log_text(self, *args, **kwargs):
        if kwargs:
            data_dict = kwargs.get("data_dict")
            global_step = kwargs.get("global_step", self._metric_step())
            log_text_keys = kwargs.get("log_text_keys")
        elif len(args) == 2:
            name, text = args
            self._mlflow.log_text(text, f"{name}.txt")
            return
        else:
            data_dict = None
            global_step = self._metric_step()
            log_text_keys = None

        if not log_text_keys or data_dict is None:
            return

        text_lines = []
        for spec in log_text_keys:
            key = spec["key"]
            text_lines.append(f"{key}: {data_dict[key][0]}")
        self._mlflow.log_text("\n".join(text_lines), f"text/log_step_{global_step}.txt")

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        path = Path(local_path)
        if path.is_file():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
            return
        if path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)
            return
        raise FileNotFoundError(f"Artifact path does not exist: {local_path}")

    def finalize(self, status: str):
        self._mlflow.end_run(status=status)


class WandbLogger(DataLogger):
    def __init__(self, log_every_n_steps: int = 1):
        super().__init__(log_every_n_steps=log_every_n_steps)
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandbLogger requires the optional 'wandb' dependency."
            ) from exc
        self._wandb = wandb
        self._run = getattr(wandb, "run", None)
        if self._run is None:
            raise RuntimeError(
                "WandbLogger requires an active wandb run. Call wandb.init() before constructing the logger."
            )

    def _metric_step(self) -> int:
        for candidate in ("train_step", "val_step", "global_step"):
            value = self._step_log.get(candidate)
            if isinstance(value, int):
                return value
        return self._epoch_step

    def report_step(self):
        metrics = {}
        for key, value in self.step_items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                metrics[key] = value

        if metrics:
            self._run.log(metrics, step=self._metric_step())

    def report_epoch(self):
        metrics = {}
        for key, value in self.epoch_items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                metrics[key] = value

        if metrics:
            self._run.log(metrics, step=self._metric_step())

    def _plot_images(self, data_dict, log_image_keys):
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(len(log_image_keys), 1, figsize=(15, 10))
        if not isinstance(axes, (list, tuple)):
            try:
                axes = list(axes)
            except TypeError:
                axes = [axes]

        for index, spec in enumerate(log_image_keys):
            key = spec["key"]
            len_key = spec.get("len_key")
            output = data_dict[key][0].float().detach().cpu().squeeze().numpy()

            if len_key is not None:
                valid_length = data_dict[len_key][0]
                if len(output.shape) == 1:
                    output = output[:valid_length]
                elif len(output.shape) == 2:
                    output = output[..., :valid_length]

            if len(output.shape) == 1:
                axes[index].plot(output)
                axes[index].set_xlim(0, output.shape[0])
            else:
                axes[index].imshow(
                    output,
                    aspect="auto",
                    origin="lower",
                    interpolation="none",
                )
            axes[index].set_title(key)

        figure.tight_layout()
        return figure

    def log_images(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        if kwargs:
            data_dict = kwargs["data_dict"]
            global_step = kwargs["global_step"]
            log_image_keys = kwargs["log_image_keys"]
        else:
            _, data_dict, log_image_keys = args
            global_step = self._metric_step()

        if not log_image_keys:
            return

        figure = self._plot_images(data_dict, log_image_keys)
        self._run.log(
            {f"images/log_step_{global_step}": self._wandb.Image(figure)},
            step=global_step,
        )
        plt.close(figure)

    def log_text(self, *args, **kwargs):
        if kwargs:
            data_dict = kwargs.get("data_dict")
            global_step = kwargs.get("global_step", self._metric_step())
            log_text_keys = kwargs.get("log_text_keys")
        elif len(args) == 2:
            name, text = args
            self._run.log(
                {name: self._wandb.Html(f"<pre>{html.escape(str(text))}</pre>")},
                step=self._metric_step(),
            )
            return
        else:
            data_dict = None
            global_step = self._metric_step()
            log_text_keys = None

        if not log_text_keys or data_dict is None:
            return

        text_lines = []
        for spec in log_text_keys:
            key = spec["key"]
            text_lines.append(f"{key}: {data_dict[key][0]}")
        self._run.log(
            {
                f"text/log_step_{global_step}": self._wandb.Html(
                    f"<pre>{html.escape(chr(10).join(text_lines))}</pre>"
                )
            },
            step=global_step,
        )

    def _artifact_name(self, local_path: Path, artifact_path: str | None) -> str:
        normalized_path = (artifact_path or "").strip("/")
        if normalized_path.startswith("checkpoints") or local_path.suffix == ".ckpt":
            return f"run-{self._run.id}-checkpoints"
        root_name = normalized_path.split("/", 1)[0] if normalized_path else "artifacts"
        return f"run-{self._run.id}-{root_name}"

    def _artifact_type(self, local_path: Path, artifact_path: str | None) -> str:
        normalized_path = (artifact_path or "").strip("/")
        if normalized_path.startswith("checkpoints") or local_path.suffix == ".ckpt":
            return "model"
        return "artifact"

    def _artifact_entry_name(self, local_path: Path, artifact_path: str | None) -> str:
        normalized_path = (artifact_path or "").strip("/")
        if not normalized_path:
            return local_path.name
        if normalized_path == "checkpoints" and local_path.is_file():
            return local_path.name
        return normalized_path

    def _add_path_to_artifact(
        self,
        artifact,
        local_path: Path,
        artifact_path: str | None = None,
    ) -> None:
        if local_path.is_file():
            artifact.add_file(
                str(local_path),
                name=self._artifact_entry_name(local_path, artifact_path),
            )
            return

        if local_path.is_dir():
            base_name = self._artifact_entry_name(local_path, artifact_path)
            for child in sorted(local_path.rglob("*")):
                if not child.is_file():
                    continue
                relative_name = child.relative_to(local_path).as_posix()
                if base_name:
                    relative_name = f"{base_name}/{relative_name}"
                artifact.add_file(str(child), name=relative_name)
            return

        raise FileNotFoundError(f"Artifact path does not exist: {local_path}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        path = Path(local_path)
        artifact = self._wandb.Artifact(
            self._artifact_name(path, artifact_path),
            type=self._artifact_type(path, artifact_path),
        )
        self._add_path_to_artifact(artifact, path, artifact_path=artifact_path)
        self._run.log_artifact(artifact, aliases=["latest"])

    def finalize(self, status: str):
        exit_code = 0 if status == "FINISHED" else 1
        self._run.finish(exit_code=exit_code)


class DistributedLogger:
    def __init__(self, logger, runtime):
        self._logger = logger
        self._runtime = runtime
        self.log_every_n_steps = getattr(logger, "log_every_n_steps", 1)

    def _forward_only_on_rank_zero(self, method_name: str, *args, **kwargs):
        if not self._runtime.is_rank_zero:
            return None
        method = getattr(self._logger, method_name)
        return method(*args, **kwargs)

    @staticmethod
    def _is_counter_key(name: str) -> bool:
        return name == "global_step" or name.endswith("_step") or name.endswith("_epoch")

    def log_value(self, name: str, value: float | int):
        if isinstance(value, torch.Tensor):
            value = value.item()
        return self._forward_only_on_rank_zero("log_value", name, value)

    def log_images(self, *args, **kwargs):
        return self._forward_only_on_rank_zero("log_images", *args, **kwargs)

    def log_text(self, *args, **kwargs):
        return self._forward_only_on_rank_zero("log_text", *args, **kwargs)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        return self._forward_only_on_rank_zero(
            "log_artifact",
            local_path,
            artifact_path=artifact_path,
        )

    def finalize(self, status: str):
        return self._forward_only_on_rank_zero("finalize", status)

    def report_step(self):
        return self._forward_only_on_rank_zero("report_step")

    def report_epoch(self):
        return self._forward_only_on_rank_zero("report_epoch")

    def step_done(self):
        return self._forward_only_on_rank_zero("step_done")

    def epoch_done(self):
        return self._forward_only_on_rank_zero("epoch_done")

    def __getattr__(self, name):
        return getattr(self._logger, name)
