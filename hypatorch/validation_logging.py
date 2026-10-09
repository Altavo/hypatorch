"""Validation-pass collection independent of tracking SDKs and training wrappers."""

from collections.abc import Mapping, Sequence
from numbers import Real

import torch
from omegaconf import OmegaConf


def _resolve_bin_edges(bins):
    """Resolve one explicit bin specification before observing any samples."""
    if OmegaConf.is_config(bins):
        bins = OmegaConf.to_container(bins, resolve=True)
    if not isinstance(bins, Mapping) or set(bins) not in ({"edges"}, {"range", "count"}):
        raise ValueError("bins requires either edges or range + count, with no other keys")
    uniform = "range" in bins
    points = bins["range"] if uniform else bins["edges"]
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ValueError("bins edges/range must be a sequence of real numbers")
    if (uniform and len(points) != 2) or (not uniform and not 2 <= len(points) <= 513):
        raise ValueError("bins.range requires two endpoints; bins.edges requires 2 to 513 edges")
    if any(isinstance(point, bool) or not isinstance(point, Real) for point in points):
        raise TypeError("bins edges/range must contain real numbers")
    edges = torch.tensor(points, dtype=torch.float64)
    if not torch.isfinite(edges).all() or not (edges[1:] > edges[:-1]).all():
        raise ValueError("bins edges/range must be finite and strictly increasing")
    if uniform:
        count = bins["count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 512:
            raise ValueError("bins.count must be an integer between 1 and 512")
        edges = torch.linspace(edges[0], edges[1], count + 1, dtype=torch.float64)
        if not torch.isfinite(edges).all() or not (edges[1:] > edges[:-1]).all():
            raise ValueError("bins range/count cannot produce distinct finite float64 edges")
    return edges


class HistogramCollector:
    """Count one numeric observation per example using fixed finite bin edges.

    Intervals are left-closed/right-open, except the final interval includes its
    right edge. Non-finite values are invalid; finite outliers are counted
    separately. Persistent storage is O(number of bins), never O(dataset size).
    """

    def __init__(self, name, key, bins):
        if not isinstance(name, str) or not name:
            raise ValueError("Histogram name must be a nonempty string")
        if name in {"samples", "global_step", "train_step", "val_step"}:
            raise ValueError("Histogram name must not be a progress coordinate")
        if not isinstance(key, str) or not key:
            raise ValueError("Histogram key must be a nonempty string")
        edges = _resolve_bin_edges(bins)
        self.name = name
        self.key = key
        self.bin_edges = edges
        self.counts = torch.zeros(len(edges) - 1, dtype=torch.int64)
        self.total_count = 0
        self.invalid_count = 0
        self.underflow_count = 0
        self.overflow_count = 0

    def update(self, data_dict, batch_size):
        values = data_dict[self.key]
        if OmegaConf.is_config(values):
            values = OmegaConf.to_container(values, resolve=True)
        source = values
        values = torch.as_tensor(values)
        if values.dtype == torch.bool or values.is_complex():
            raise TypeError(f"Histogram {self.name!r} requires real numeric observations")
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or values.shape[0] != batch_size:
            raise ValueError(
                f"Histogram {self.name!r} requires one value per example: "
                f"expected [{batch_size}] or [{batch_size}, 1], got {tuple(values.shape)}"
            )
        # Convert from the original input: a Python float sequence must not
        # round through Torch's default float32 before comparison with edges.
        values = torch.as_tensor(source, device="cpu", dtype=torch.float64).detach().reshape(-1)
        finite = torch.isfinite(values)
        self.total_count += batch_size
        self.invalid_count += int((~finite).sum())
        values = values[finite]
        self.underflow_count += int((values < self.bin_edges[0]).sum())
        self.overflow_count += int((values > self.bin_edges[-1]).sum())
        values = values[(values >= self.bin_edges[0]) & (values <= self.bin_edges[-1])]
        indices = torch.bucketize(values, self.bin_edges, right=True) - 1
        indices.clamp_(max=len(self.counts) - 1)
        self.counts += torch.bincount(indices, minlength=len(self.counts))

    def emit(self, logger, global_step):
        logger.log_histogram(
            self.name,
            counts=self.counts.tolist(),
            bin_edges=self.bin_edges.tolist(),
            total_count=self.total_count,
            invalid_count=self.invalid_count,
            underflow_count=self.underflow_count,
            overflow_count=self.overflow_count,
            global_step=global_step,
        )


def validation_collectors(entries):
    """Create fresh collectors for log_histogram entries, always for a full pass."""
    collectors = []
    names = set()
    for entry in entries:
        if OmegaConf.is_config(entry):
            entry = OmegaConf.to_container(entry, resolve=True)
        if not isinstance(entry, Mapping):
            raise TypeError("Logging entries must be mappings")
        if "aggregate" in entry:
            raise ValueError("aggregate is not supported; log_histogram always collects the full validation pass")
        if entry.get("fn") != "log_histogram":
            continue
        kwargs = {key: value for key, value in entry.items() if key != "fn"}
        collector = HistogramCollector(**kwargs)
        if collector.name in names:
            raise ValueError(f"Duplicate validation histogram name: {collector.name!r}")
        names.add(collector.name)
        collectors.append(collector)
    return collectors
