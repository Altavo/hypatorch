from abc import ABC, abstractmethod
import html
import operator
from collections.abc import Mapping
from pathlib import Path

import torch
from omegaconf import OmegaConf


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

    def log_media(self, name=None, value=None, *, media_type=None, global_step=None,
                  data_dict=None, key=None, sample_index=0, len_key=None,
                  time_axis=-1, **kwargs):
        """Optional rich-media hook; unsupported backends ignore it."""
        del name, value, media_type, global_step, data_dict, key, sample_index, len_key, time_axis, kwargs

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        del local_path, artifact_path

    def log_table(self, name, columns, *, data_dict, sample_index=0, global_step=None):
        """Optional table hook; unsupported backends ignore it."""
        del name, columns, data_dict, sample_index, global_step

    def log_distribution(self, name, *, values, global_step=None):
        """Optional sink for finite per-example values from a completed pass."""
        del name, values, global_step

    def log_histogram(self, name, *, counts, bin_edges, total_count, invalid_count,
                      underflow_count, overflow_count, global_step=None):
        """Optional sink for a completed validation histogram."""
        del name, counts, bin_edges, total_count, invalid_count, underflow_count, overflow_count, global_step

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

    # Progress coordinates are logged verbatim (no _step/_epoch suffix, never
    # averaged) so the tracking backend can use them as x-axes. See
    # WandbLogger.__init__, which makes `samples` the default display axis.
    _COORDINATE_KEYS = frozenset({"samples", "global_step", "train_step", "val_step"})

    def _is_validation(self) -> bool:
        # A validation flush carries val_step but no train_step. Used to emit a
        # single aggregated validation point per pass (via report_epoch) instead
        # of one point per validation batch.
        return "val_step" in self._step_log and "train_step" not in self._step_log

    def _is_coordinate_key(self, key: str) -> bool:
        base = key
        if base.endswith("_step"):
            base = base[: -len("_step")]
        elif base.endswith("_epoch"):
            base = base[: -len("_epoch")]
        return key in self._COORDINATE_KEYS or base in self._COORDINATE_KEYS

    def _progress_coordinates(self) -> dict:
        # Current position, read live from the step log so it is correct for
        # both per-step and per-epoch flushes. During validation the training
        # coordinates (samples/train_step) are frozen at the value reached when
        # the eval started, which is exactly where the val point should land.
        coords = {}
        for key in self._COORDINATE_KEYS:
            value = self._step_log.get(key)
            if isinstance(value, int):
                coords[key] = value
        return coords


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
        # The tracking backend requires a single, monotonically-increasing step
        # axis. Only global_step is monotonic across both train and val;
        # train_step and val_step are independent per-mode counters that would
        # collide when both are mapped onto one step axis.
        value = self._step_log.get("global_step")
        if isinstance(value, int):
            return value
        return self._epoch_step

    def report_step(self):
        # Validation is emitted once per pass in report_epoch (a single
        # aggregated point at the training-progress coordinate), so skip the
        # per-batch validation flush here.
        if self._is_validation():
            return

        metrics = {}
        for key, value in self.step_items():
            if self._is_coordinate_key(key):
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        if metrics:
            metrics.update(
                {k: float(v) for k, v in self._progress_coordinates().items()}
            )
            self._mlflow.log_metrics(metrics, step=self._metric_step())

    def report_epoch(self):
        metrics = {}
        for key, value in self.epoch_items():
            if self._is_coordinate_key(key):
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        if metrics:
            metrics.update(
                {k: float(v) for k, v in self._progress_coordinates().items()}
            )
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
    _MEDIA_TYPES = {
        "image": "Image",
        "audio": "Audio",
        "video": "Video",
        "html": "Html",
        "object3d": "Object3D",
        "molecule": "Molecule",
        "table": "Table",
        "histogram": "Histogram",
        "plotly": "Plotly",
        "graph": "Graph",
    }

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

        # Decouple the human-facing x-axis from wandb's monotonic commit step:
        # plot every metric against cumulative training samples by default.
        # Validation is logged at the samples reached when it ran, so val and
        # train curves align, and validation batches (which do not advance
        # `samples`) no longer distort the axis.
        self._run.define_metric("samples")
        self._run.define_metric("*", step_metric="samples")

    # These W&B types serialize lists as a single media collection.
    _BATCH_MEDIA_TYPES = frozenset({"audio", "image", "video", "html", "object3d", "molecule"})

    @staticmethod
    def _select_media(data_dict, key, sample_index, len_key, time_axis, media_type):
        value = data_dict[key]
        if sample_index is not None:
            if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
                raise ValueError("sample_index must be a nonnegative integer or None")
            value = value[sample_index]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if len_key is not None and not hasattr(value, "shape"):
            raise TypeError("len_key requires tensor or array media")
        if len_key is not None or (media_type == "audio" and hasattr(value, "shape")):
            ndim = len(value.shape)
            axis = operator.index(time_axis)
            if not -ndim <= axis < ndim:
                raise ValueError("time_axis is outside the selected media dimensions")
            axis %= ndim
            if len_key is not None:
                length = data_dict[len_key]
                if sample_index is not None:
                    length = length[sample_index]
                if isinstance(length, torch.Tensor):
                    length = length.item()
                length = operator.index(length)
                if not 0 <= length <= value.shape[axis]:
                    raise ValueError("Media length is outside the selected time dimension")
                slices = [slice(None)] * ndim
                slices[axis] = slice(length)
                value = value[tuple(slices)]
            if media_type == "audio":
                # W&B audio expects [time] or [time, channels].
                value = value.transpose([axis] + [i for i in range(ndim) if i != axis])
        return value

    def _construct_media(self, value, media_type, kwargs):
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        kwargs = {
            k: OmegaConf.to_container(v, resolve=True) if OmegaConf.is_config(v) else v
            for k, v in kwargs.items()
        }
        if media_type is None:
            if kwargs:
                raise TypeError("Constructor options require media_type")
        else:
            if media_type not in self._MEDIA_TYPES:
                raise ValueError(f"Unsupported media_type: {media_type!r}")
            if media_type == "graph":
                raise ValueError("Pass a constructed wandb.Graph without media_type")
            constructor = getattr(self._wandb, self._MEDIA_TYPES[media_type])
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if media_type == "table":
                if value is not None:
                    kwargs["data"] = value
                value = constructor(**kwargs)
            elif value is None:
                value = constructor(**kwargs)
            else:
                value = constructor(value, **kwargs)
        return value

    def log_media(self, name=None, value=None, *, media_type=None, global_step=None,
                  data_dict=None, key=None, sample_index=0, len_key=None,
                  time_axis=-1, **kwargs):
        """Log rich media without scalar averaging or cadence throttling.

        Hydra selects data_dict[key][sample_index]; None selects the whole
        value. A nonempty list selects an ordered collection of batchable media.
        Each sample is independently trimmed using len_key and time_axis.
        Selected audio moves its time axis first. Direct values keep their shape.
        Constructor options are forwarded to W&B. Omit media_type for prebuilt
        objects, including graphs. The W&B step remains open (commit=False).
        """
        if OmegaConf.is_config(sample_index):
            sample_index = OmegaConf.to_container(sample_index, resolve=True)
        multiple = isinstance(sample_index, list)
        if multiple:
            if key is None:
                raise ValueError("A sample_index list requires key and data_dict")
            if not sample_index or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0
                for i in sample_index
            ):
                raise ValueError("sample_index must be a nonempty list of nonnegative integers")
            if media_type not in self._BATCH_MEDIA_TYPES:
                raise ValueError(
                    "A sample_index list supports only audio, image, video, html, object3d, or molecule"
                )
        if key is not None:
            if data_dict is None:
                raise ValueError("key requires data_dict")
            if value is not None:
                raise ValueError("Pass either value or key, not both")
            indices = sample_index if multiple else [sample_index]
            # Select all items before constructing or logging any media, so an
            # invalid later index or length cannot produce a partial log entry.
            values = [
                self._select_media(data_dict, key, i, len_key, time_axis, media_type)
                for i in indices
            ]
            if name is None:
                name = key
        elif len_key is not None:
            raise ValueError("len_key requires key")
        else:
            values = [value]
        if not isinstance(name, str) or not name:
            raise ValueError("Media requires a nonempty name or key")
        if name in self._COORDINATE_KEYS:
            raise ValueError(f"Media name is a reserved progress coordinate: {name}")
        converted = [self._construct_media(v, media_type, kwargs) for v in values]
        value = converted if multiple else converted[0]
        payload = self._progress_coordinates()
        payload[name] = value
        step = self._metric_step() if global_step is None else global_step
        if global_step is not None:
            payload["global_step"] = global_step
        # Leave this step open for more media and its scalar report. Committing
        # here would cause subsequent writes at this step to be dropped by W&B.
        self._run.log(payload, step=step, commit=False)

    @staticmethod
    def _render_spectrogram(value, time_axis=-1, *, cmap="magma", vmin=None, vmax=None):
        """Render a 2-D mel-by-time array without changing its amplitude scale."""
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        if len(value.shape) != 2:
            raise ValueError("render: spectrogram requires a 2-D selected sample")
        axis = operator.index(time_axis)
        if not -2 <= axis < 2:
            raise ValueError("Spectrogram time_axis must identify one of its two axes")
        value = np.moveaxis(value, axis, -1)
        if 0 in value.shape:
            raise ValueError("Cannot render an empty spectrogram")
        figure = Figure(figsize=(6, 3), layout="constrained")
        try:
            canvas = FigureCanvasAgg(figure)
            axes = figure.subplots()
            axes.imshow(value, origin="lower", aspect="auto", interpolation="none",
                        cmap=cmap, vmin=vmin, vmax=vmax)
            axes.set_xlabel("Frame")
            axes.set_ylabel("Mel bin")
            canvas.draw()
            return np.asarray(canvas.buffer_rgba()).copy()
        finally:
            figure.clear()

    def log_table(self, name, columns, *, data_dict, sample_index=0, global_step=None):
        """Build aligned rows from Hydra column specs and log one W&B table.

        Each column declares name, key, optional media_type, len_key, time_axis,
        and W&B constructor options. Image columns may use render='spectrogram'
        with render_options (cmap, vmin, vmax). An integer selects one row, a list
        preserves order, and None selects every row. All batch lengths must match.
        """
        if not isinstance(name, str) or not name or name in self._COORDINATE_KEYS:
            raise ValueError("Table name must be nonempty and not a progress coordinate")
        if OmegaConf.is_config(columns):
            columns = OmegaConf.to_container(columns, resolve=True)
        if not isinstance(columns, list) or not columns:
            raise ValueError("columns must be a nonempty list")
        specs, names = [], []
        batch_size = None
        for column in columns:
            if not isinstance(column, Mapping):
                raise TypeError("Each table column must be a mapping")
            spec = dict(column)
            column_name, key = spec.pop("name", None), spec.pop("key", None)
            if not isinstance(column_name, str) or not column_name or column_name in names:
                raise ValueError("Table columns require unique nonempty names")
            if not isinstance(key, str) or not key:
                raise ValueError(f"Column {column_name!r} requires a data key")
            media_type = spec.pop("media_type", None)
            if media_type is not None and media_type not in self._MEDIA_TYPES:
                raise ValueError(f"Unsupported media_type: {media_type!r}")
            if media_type == "graph":
                raise ValueError("Pass prebuilt graphs without media_type")
            len_key = spec.pop("len_key", None)
            time_axis = spec.pop("time_axis", -1)
            render = spec.pop("render", None)
            render_options = spec.pop("render_options", {})
            if render is not None and (render != "spectrogram" or media_type != "image"):
                raise ValueError("Only image columns support render: spectrogram")
            if not isinstance(render_options, Mapping) or set(render_options) - {"cmap", "vmin", "vmax"}:
                raise ValueError("render_options accepts cmap, vmin, and vmax")
            if render_options and render is None:
                raise ValueError("render_options requires render: spectrogram")
            if media_type is None and spec:
                raise ValueError("Plain table columns do not accept media constructor options")
            for data_key in [key] + ([len_key] if len_key is not None else []):
                batch = data_dict[data_key]
                if isinstance(batch, (str, bytes, Mapping)):
                    raise TypeError(f"Table data {data_key!r} must have a batch dimension")
                size = len(batch)
                if batch_size is None:
                    batch_size = size
                elif size != batch_size:
                    raise ValueError(f"Inconsistent batch length for {data_key!r}: {size} != {batch_size}")
            names.append(column_name)
            specs.append((key, media_type, len_key, time_axis, render, render_options, spec))
        if OmegaConf.is_config(sample_index):
            sample_index = OmegaConf.to_container(sample_index, resolve=True)
        indices = (list(range(batch_size)) if sample_index is None else
                   sample_index if isinstance(sample_index, list) else [sample_index])
        if not indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
            raise ValueError("sample_index must select one or more nonnegative integer indices")
        if any(i >= batch_size for i in indices):
            raise IndexError("Table sample_index is outside the batch")
        # Validate every selection before constructing media or emitting a table.
        selected = []
        for index in indices:
            row = []
            for key, media_type, len_key, axis, render, _, _ in specs:
                value = self._select_media(data_dict, key, index, len_key, axis, media_type)
                if render is not None and (getattr(value, "ndim", None) != 2 or 0 in value.shape):
                    raise ValueError("render: spectrogram requires a nonempty 2-D selected sample")
                row.append(value)
            selected.append(row)
        rows = []
        for selected_row in selected:
            row = []
            for value, (_, media_type, _, axis, render, render_options, options) in zip(selected_row, specs):
                if render is not None:
                    value = self._render_spectrogram(value, axis, **render_options)
                if media_type is None and hasattr(value, "tolist"):
                    value = value.tolist()
                row.append(self._construct_media(value, media_type, options))
            rows.append(row)
        table = self._wandb.Table(columns=names, data=rows)
        self.log_media(name, table, global_step=global_step)

    def log_distribution(self, name, *, values, global_step=None):
        """Log a fresh raw-value table and conventional W&B histogram per pass."""
        if not isinstance(name, str) or not name or name in self._COORDINATE_KEYS:
            raise ValueError("Distribution name must be nonempty and not a progress coordinate")
        import numpy as np

        # Custom charts query the artifact-backed table. MAX_ROWS only limits
        # the legacy media preview; enforce the artifact limit to avoid truncation.
        limit = self._wandb.Table.MAX_ARTIFACT_ROWS
        if len(values) > limit:
            raise ValueError(f"Distribution has {len(values)} values, exceeding W&B table limit {limit}")
        array = np.asarray(values)
        if array.ndim != 1 or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError("Distribution values must be a finite real numeric vector")
        table = self._wandb.Table(columns=["value"], data=[[float(value)] for value in array])
        chart = self._wandb.plot.histogram(table, "value", title=name)
        payload = self._progress_coordinates()
        payload[name] = chart
        if global_step is not None:
            payload["global_step"] = global_step
        step = self._metric_step() if global_step is None else global_step
        self._run.log(payload, step=step, commit=False)

    def log_histogram(self, name, *, counts, bin_edges, total_count, invalid_count,
                      underflow_count, overflow_count, global_step=None):
        """Publish precomputed counts and coverage diagnostics in one W&B row."""
        payload = self._progress_coordinates()
        payload[name] = self._wandb.Histogram(np_histogram=(counts, bin_edges))
        for key, value in {
            "total_count": total_count,
            "invalid_count": invalid_count,
            "underflow_count": underflow_count,
            "overflow_count": overflow_count,
        }.items():
            payload[f"{name}/{key}"] = value
        step = self._metric_step() if global_step is None else global_step
        if global_step is not None:
            payload["global_step"] = global_step
        self._run.log(payload, step=step, commit=False)

    def _metric_step(self) -> int:
        # wandb requires a single, monotonically-increasing step axis. Only
        # global_step is monotonic across both train and val; train_step and
        # val_step are independent per-mode counters that would collide (and get
        # silently dropped by wandb) when both are mapped onto wandb's step.
        value = self._step_log.get("global_step")
        if isinstance(value, int):
            return value
        return self._epoch_step

    def report_step(self):
        # Validation is emitted once per pass in report_epoch (a single
        # aggregated point at the training-progress coordinate), so skip the
        # per-batch validation flush here.
        if self._is_validation():
            return

        metrics = {}
        for key, value in self.step_items():
            if self._is_coordinate_key(key):
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                metrics[key] = value

        if metrics:
            metrics.update(self._progress_coordinates())
            self._run.log(metrics, step=self._metric_step())

    def report_epoch(self):
        metrics = {}
        for key, value in self.epoch_items():
            if self._is_coordinate_key(key):
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                metrics[key] = value

        if metrics:
            metrics.update(self._progress_coordinates())
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
        return self._run.log_artifact(artifact, aliases=["latest"])

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

    def log_media(self, *args, **kwargs):
        return self._forward_only_on_rank_zero("log_media", *args, **kwargs)

    def log_table(self, *args, **kwargs):
        return self._forward_only_on_rank_zero("log_table", *args, **kwargs)

    def log_distribution(self, *args, **kwargs):
        return self._forward_only_on_rank_zero("log_distribution", *args, **kwargs)

    def log_histogram(self, *args, **kwargs):
        return self._forward_only_on_rank_zero("log_histogram", *args, **kwargs)

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
