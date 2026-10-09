# Losses and metrics

Use this when you declare what an operation optimizes, what it reports, or what
it logs beyond scalars.

## Assessments

Losses and metrics are the same object — a `HypaAssessment` wrapping any callable
that computes a value from dict keys. `losses` are summed and backpropagated;
`metrics` are computed and reported. Both are declared per operation.

```yaml
losses:
  - _target_: hypatorch.HypaAssessment
    assessment:
      _target_: torch.nn.CrossEntropyLoss
      reduction: mean
    name: mean_cross_entropy
    inputs:
      input: logits
      target: class
    weight: 1.0
```

| Field | Meaning |
| --- | --- |
| `assessment` | The module that computes the value |
| `name` | The reported name, logged as `<mode>/<name>` |
| `inputs` | `argument_name: dict_key`, with the same literal and list forms a mapping accepts |
| `weight` | Multiplies the result; the default is `1.0` |
| `apply` | Modes this assessment runs in; omitted means always. Same `train` / `val` / `predict` names a mapping uses, checked the same way |
| `harmonize_inputs` | Argument names whose lengths are aligned before the call |
| `masking` | `{apply: [...], len_key: ...}` — see below |
| `requires_model` | Passes the model itself as a `model` argument |

An operation's total loss is the plain sum of its weighted losses. There is no
separate weighting stage, so `weight` is the only lever.

## Masking

`masking` is optional, and it is the framework taking the job over rather than a
requirement to declare padding. With it, the assessment is wrapped so padded
positions do not contribute:

- `len_key` is a **dict key** holding the per-sample lengths.
- `apply` names the **arguments** to mask, resolved through `inputs`.
- The mask is built over the **last** dimension.
- The wrapped module must accept `reduction='sum'`, enforced at construction,
  because the mask decides the normalization: `mean` then divides by the unmasked
  element count rather than the padded one.

This is why a batcher's length keys matter downstream: without them a masked loss
has nothing to mask by.

### Masking inside the assessment instead

`masking: null` leaves the assessment unwrapped, and every entry in `inputs` is
passed to it as a plain keyword argument — lengths included. A module that
handles its own padding just takes the length as another input:

```yaml
    metrics:
      - _target_: hypatorch.HypaAssessment
        assessment:
          _target_: modules.metrics.Accuracy
          avg_acc: true
        name: accuracy
        inputs:
          preds: motor_indices
          target: motor_indices_pred
          target_len: in_motor_len
        harmonize_inputs: null
        masking: null
```

This is usually the right form for a metric written for this model: the
`reduction='sum'` contract exists so the wrapper can normalize, and a module that
normalizes itself has no reason to satisfy it. `harmonize_inputs` is an
independent opt-in and can be left null the same way.

## Length harmonization

`harmonize_inputs` truncates the named inputs (and the mask, when masking is on)
to the shortest last dimension before the assessment runs, for a prediction and a
target that differ by a frame. It is an alignment tool, not a correctness
argument, and it says so: a difference of more than `mismatch_threshold` samples
— 5 by default, set through `harmonizer_kwargs` — raises rather than truncates,
on the grounds that a bug is likelier than a rounding step.

## Ready-made losses

`hypatorch.losses` has `MAE_Loss`, `MSE_Loss` and their masked variants
`MMAE_Loss` and `MMSE_Loss`. They are thin `HypaAssessment` subclasses that fill
in the assessment, the harmonization and a derived name, so they take only
`inputs` and `weight`. Reach for them before hand-rolling the equivalent.

## What gets logged

Every computed assessment is logged as `<mode>/<name>` — `val/cer`,
`train/mean_cross_entropy` — and the logger appends its own per-step or per-epoch
suffix. Mode namespacing is automatic; uniqueness within a mode is yours.

## Custom logging entries

An operation may declare `logging` entries for what a scalar cannot carry.
Ordinary entries run on the **first validation batch only**, never during training.
`log_histogram` always collects every validation batch and
emits once after the complete pass (see below). `log_images` and `log_text` render the **first sample**; `log_media`
lets you choose a sample or the whole value.

```yaml
    logging:
      - fn: log_images
        log_image_keys:
          - key: spectrogram
            len_key: spectrogram_len
          - key: prediction
      - fn: log_text
        log_text_keys:
          - key: transcript
```

For ordinary entries, `fn` names a method on the active logger and every other key is forwarded to it,
together with `data_dict` and `global_step`. The set of usable methods is fixed
by the active logger; these methods fit this call shape:

| `fn` | Takes | Renders |
| --- | --- | --- |
| `log_images` | `log_image_keys: [{key, len_key?}]` | One subplot per key: a line for 1-D, an image for 2-D, truncated to `len_key` |
| `log_text` | `log_text_keys: [{key}]` | One line per key |
| `log_media` | `key`, `media_type`, optional selection and W&B options | Rich media (W&B only) |
| `log_table` | `name`, `columns`, optional `sample_index` | Aligned mixed-media table (W&B only) |

`log_value` is not usable here — assessments call it themselves, and its
signature is `(name, value)`, so routing it through this path raises. `log_artifact`
is driven by the trainer for checkpoints.

Both `WandbLogger` and `MLflowLogger` implement `log_images` and `log_text`. The base class defines
them as no-ops, so the same config under `ConsoleLogger` logs nothing at all,
without complaint. Omitting the keys argument is the opposite failure: the
implementations read it directly and raise a `KeyError`.

## W&B rich media

For Hydra, put entries under an operation's `logging` list:

```yaml
logging:
  - fn: log_media
    name: val/prediction
    media_type: audio
    key: predicted_audio
    len_key: predicted_audio_length
    sample_index: 0
    time_axis: -1
    sample_rate: 16000
```

The trainer supplies `data_dict` (inputs plus operation outputs) and `global_step`.
The default backend must be W&B to emit media; console and MLflow ignore this hook.

| Field | Meaning |
| --- | --- |
| `key` | Exact key in `data_dict`; required for selecting runtime data |
| `name` | W&B field name; defaults to `key` |
| `sample_index` | Nonnegative batch index (default `0`), nonempty list of indices, or `null` for the whole value |
| `len_key` | Optional key containing integer valid lengths, selected with the same sample index |
| `time_axis` | Axis to trim in the selected sample, default `-1`; also identifies audio's time axis |
| `media_type` | Constructor type from the table below; omit for prebuilt W&B objects |

Other options go to the W&B constructor. Nested Hydra lists and dictionaries
are resolved to ordinary Python containers. Missing keys, out-of-range sample
indices, and lengths outside the selected time dimension raise errors.

For `[batch, channels, time]` audio, defaults produce `[time, channels]` clips.
For `[batch, time, channels]`, set `time_axis: 0`. `[batch, time]` mono audio
produces `[time]`. Lengths are absolute sample counts, not relative fractions.
Trimming happens before audio axis conversion. Other media types retain their
selected layout, so set `time_axis` explicitly when trimming video or images.
Audio files may be selected by key without `len_key`; no reshaping is applied.

Use `sample_index: null` for a whole table or a prebuilt graph in `data_dict`.
To log several samples under one W&B key, use a list:

```yaml
logging:
  - fn: log_media
    name: val/predictions
    media_type: audio
    key: predicted_audio
    len_key: predicted_audio_length
    sample_index: [0, 2, 4]
    sample_rate: 16000
```

Each selected sample uses its own length and is converted independently. The
collection preserves the requested order, including repeated indices. A one-item
list still emits a collection; an integer emits one object as before. Constructor
options (such as `caption`) are shared across all selected samples.

List selection requires `key` and an explicit `media_type` of `audio`, `image`,
`video`, `html`, `object3d`, or `molecule`. Tables, histograms, Plotly figures,
graphs, and prebuilt objects do not support list selection; log those separately
or use `sample_index: null` to pass the whole value. Empty lists, negative or
noninteger indices, and indices outside the batch raise errors. All selections
and lengths are checked before constructing media or logging a collection.

For separate W&B keys, use entries with distinct `name` values instead.
Entries run on the first validation batch of each pass. Their sample indices must
exist in that batch. Media options such as image masks are literal constructor
options; only `key` and `len_key` resolve runtime data keys.

The Python API remains available:

Initialize a W&B run before constructing `hypatorch.logger.WandbLogger`.
Use `logger.log_media(name, value, media_type=..., **options)` for rich data.
Constructor options are forwarded to W&B; omit `media_type` to pass an already
constructed W&B object or list of objects. Existing `log_images` (waveform /
spectrogram plots), `log_text`, and scalar logging remain available.

```python
import wandb
from hypatorch.logger import WandbLogger

with wandb.init(project="audio-experiments"):
    logger = WandbLogger()
    logger.log_value("global_step", 0)
    logger.log_value("samples", 16)
    logger.log_media("val/audio", waveform, media_type="audio", sample_rate=16000)
    logger.log_media("val/spectrogram", spectrogram_image, media_type="image")
    logger.log_value("loss", 0.5)
    logger.step_done()
```

Log media alongside the scalar report (`step_done` / `epoch_done`) for the
same global step. Media uses `commit=False` so multiple media calls and the
scalar report can share that step. A later step, an explicit W&B commit, or run
finalization flushes pending media. Do not write to an already committed step. `global_step=...`
overrides the step; otherwise the logger uses its current progress coordinates.
Media is never averaged and is not throttled by `log_every_n_steps`; callers
control its frequency. `DistributedLogger` constructs and logs media only on
rank zero. Other current backends ignore this optional hook.

| `media_type` | Input and typical options |
|---|---|
| `image` | Image data/path; `caption`, `masks`, `boxes` for overlays |
| `audio` | Waveform/path; `sample_rate`, `caption` |
| `video` | Video data/path; `fps`, `format` |
| `html` | HTML string/file; `inject` |
| `object3d` | Point cloud or supported 3D file |
| `molecule` | Molecular data/file; `caption` |
| `table` | Rows with `columns`, or keyword-only `dataframe` |
| `histogram` | Values with `num_bins`, or keyword-only `np_histogram` |
| `plotly` | Plotly figure |

Top-level tensors are detached and copied to CPU NumPy arrays without reshaping.
Supply the shape and dtype required by W&B (for example mono audio as `[time]`,
multichannel audio as `[time, channels]`) for direct Python values. Hydra
key selection handles trimming and audio layout as described above. Nested
media in table cells must be constructed explicitly, e.g. `wandb.Audio(...)`.
W&B's optional dependencies for the selected type must be installed.

### Model graphs

Pass an explicitly constructed `wandb.Graph` without `media_type`:

```python
graph = wandb.Graph()
source = graph.add_node(id="encoder", name="Encoder")
target = graph.add_node(id="decoder", name="Decoder")
graph.add_edge(source, target)
logger.log_media("model/graph", graph)  # before the scalar flush
```

This records the supplied nodes and edges; it does not infer Hypatorch's dynamic
data-dictionary connections. For automatic PyTorch model capture, W&B exposes
`wandb.run.watch(model, log=None, log_graph=True)`, called before model execution
and only on rank zero. That installs execution hooks; capture depends on the
executed model path and is separate from `log_media`. Use Plotly or HTML when you
need a custom rendering of Hypatorch's configuration/data-flow graph.

## Mixed-media tables

Use `log_table` under an operation's `logging` list to assemble one table from
aligned input/output keys. The tracking backend must provide `WandbLogger`.
Console and MLflow ignore this optional hook. The trainer supplies `data_dict`
and `global_step`; logging runs on the first validation batch, only on rank zero.

```yaml
logging:
  - fn: log_table
    name: val/examples
    sample_index: [0, 2, 4]
    columns:
      - name: prediction
        key: predicted_audio
        media_type: audio
        len_key: predicted_audio_length
        sample_rate: 16000
      - name: ground_truth_text
        key: transcript
      - name: ground_truth_audio
        key: target_audio
        media_type: audio
        len_key: target_audio_length
        sample_rate: 16000
      - name: mel
        key: predicted_mel
        media_type: image
        len_key: predicted_mel_length
        time_axis: -1
        render: spectrogram
        render_options:
          cmap: magma
```

`name` is the W&B table key. `columns` is a nonempty list with unique nonempty
column names and exact `data_dict` keys. Column order is preserved. Every data
key and length key must have the same batch length; no broadcasting occurs.
`sample_index` defaults to `0` (one row), accepts a nonempty list (preserving order
and repetitions), or `null` (all rows of this batch). All selected indices must
exist. Missing keys, inconsistent batch lengths, invalid lengths, and invalid
selections fail before media construction or table logging.

Each column accepts `name`, `key`, optional `media_type`, `len_key`, `time_axis`,
and W&B constructor options such as `sample_rate` or `caption`. Selection,
trimming and audio orientation follow `log_media`. Omit `media_type` for ordinary
text, numbers or prebuilt W&B objects. Selected numeric tensors/arrays become
Python scalars or lists for plain cells. A plain column cannot carry media
constructor options. Graph cells must contain prebuilt objects, with no
`media_type`. Nested tables can be constructed with `media_type: table` and
column names for their own rows.

For image columns, `render: spectrogram` plots a **2-D selected sample** after
trimming, with time horizontally and mel bins vertically (low bins at the bottom).
`time_axis` defaults to `-1`; use `0` for `[time, mel]` samples. Remove any singleton
channel dimension in model preprocessing. Empty or non-2-D samples fail.
The renderer uses a headless Matplotlib canvas and does not apply logarithms,
dB conversion, or amplitude normalization. Supply the desired values upstream.
`render_options` accepts `cmap` (default `magma`), `vmin`, and `vmax`. Matplotlib
scales colors per image unless limits are supplied; use fixed limits for visual
comparisons. The image carries Frame and Mel bin labels, not physical units.
Without `render`, image values go directly to W&B.

Install the optional media dependencies in the training environment: Matplotlib
and Pillow for spectrogram images, and soundfile for W&B audio from arrays.
The logger creates a fresh table for each call and writes it with `commit=False`,
sharing the step with metrics. It does not accumulate rows across batches or
validation passes. In Python the same API is available as
`logger.log_table(name, columns, data_dict=batch, sample_index=[0, 2])`.

## Histograms over a complete validation pass

Use this when every evaluated example must contribute one numeric observation,
without retaining all observations or reducing them to batch averages.
This feature is single-process only. It consumes ordinary model inputs/outputs;
it does not compute a metric or depend on a consumer's training routine.

```yaml
logging:
  - fn: log_histogram
    name: val/score_distribution
    key: score_per_sample
    bins:
      range: [0.0, 2.0]
      count: 100
```

The entry belongs under an operation's `logging`, just like media and tables.
`name` must be unique across pass-level histograms in all operations and cannot
be a progress coordinate. `key` selects an exact key from merged model inputs
and outputs after all operations have run. It must contain a real numeric tensor,
array or sequence of shape `[batch]` or `[batch, 1]`: one observation per input
example. Scalars, sequence-valued metrics, boolean/complex values and mismatched
batch lengths fail. Produce unreduced per-example values as model outputs;
scalar assessments and their existing logging remain independent.

`bins` accepts exactly one of two forms. For evenly spaced bins, provide
`range: [lower, upper]` and an integer `count` between 1 and 512. Endpoints must
be finite real numbers with `lower < upper`. For custom spacing:

```yaml
bins:
  edges: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
```

`edges` contains 2–513 finite, strictly increasing real numbers (1–512 bins).
Mixed forms, missing fields, unknown keys, boolean values, and noninteger counts
are rejected. No range is inferred from the observations. Both forms resolve to
fixed float64 edges before validation starts; ranges too narrow to produce
distinct edges at that precision are rejected. The old top-level `bin_edges`
configuration is not accepted. The backend sink still receives resolved edges.

Intervals include their left edge and exclude their right edge, except the final
bin includes its right edge. Fixed edges make passes comparable.
The entry requires exactly `fn`, `name`, `key` and `bins`.
No sample selection, implicit range, padding removal, or subsampling is applied.
`log_histogram` inherently covers the full validation pass; there is no
`aggregate` option. Supplying the obsolete option raises a configuration error.

The collector holds integer counts on CPU and releases each batch's values;
persistent memory depends on bin count, not validation dataset size. It does not
retain computation graphs. It is recreated for every validation pass.

### Coverage contract

- Set `trainer.max_val_samples: null` (negative values also normalize to no cap).
  Any active cap is rejected, even if larger than the dataset.
- Distributed execution is rejected before entering training/validation.
- Validation loaders must use `drop_last: false` and standard automatic batching.
  For map-style datasets, only the standard sequential sampler or a standard
  non-replacement random sampler covering every dataset position is accepted.
  Custom batch samplers, subset/replacement samplers and unbatched loaders fail.
- The map-style evaluated sample count must equal the dataset length. For
  iterable datasets, full coverage means **natural exhaustion of the supplied
  stream**. Hypatorch cannot detect samples hidden by upstream filters, truncation,
  repetition, or missing data. It does not infer an external dataset's cardinality
  or deduplicate IDs. Collation must preserve one output example per input example.
- An interrupt, stop request, missing key, bad shape or loader exception aborts
  the pass: no pass-level histogram is published, and its local state is discarded.
  A graceful-stop request during collection therefore raises an error rather than
  publishing a partial full-pass result. Earlier ordinary media logs may remain.
- An empty, naturally exhausted validation dataset emits zero counts and
  `total_count=0`, not a fabricated observation.

### Reporting

`WandbLogger` receives the completed counts via its `log_histogram` sink, in one
`commit=False` call before the scalar epoch report, using the final validation
step and existing progress coordinates. For an empty pass it uses the current
trainer step. Reporting is not throttled by `log_every_n_steps`.
It logs the histogram under `name` and these companion metrics:

| Key suffix (appended to `name/`) | Meaning |
| --- | --- |
| `total_count` | All observed examples, including invalid values and outliers |
| `invalid_count` | NaN and positive/negative infinity; excluded from bins |
| `underflow_count` | Finite values below the first edge |
| `overflow_count` | Finite values above the final edge |

The invariant is `sum(bin_counts) + invalid_count + underflow_count + overflow_count
== total_count`. Outliers are **not clipped into the end bins**; they are reported
separately. The W&B plot alone shows only in-range values, so inspect the counters.
The counts are not epoch-averaged scalar metrics. Other current backends inherit
a no-op sink; the coverage checks still apply, but they do not render histograms.

This feature does not change first-batch audio/image/table selection. Collection
of selected media across validation batches is a separate extension.
