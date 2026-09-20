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

An operation may declare `logging` entries for what a scalar cannot carry. They
are called once per validation pass, on its **first batch only** — never during
training — and they render the **first sample** of that batch.

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

`fn` names a method on the active logger and every other key is forwarded to it,
together with `data_dict` and `global_step`. The set of usable methods is fixed
by `DataLogger`, and only two fit this call shape:

| `fn` | Takes | Renders |
| --- | --- | --- |
| `log_images` | `log_image_keys: [{key, len_key?}]` | One subplot per key: a line for 1-D, an image for 2-D, truncated to `len_key` |
| `log_text` | `log_text_keys: [{key}]` | One line per key |

`log_value` is not usable here — assessments call it themselves, and its
signature is `(name, value)`, so routing it through this path raises. `log_artifact`
is driven by the trainer for checkpoints.

Only `WandbLogger` and `MLflowLogger` implement the two. The base class defines
them as no-ops, so the same config under `ConsoleLogger` logs nothing at all,
without complaint. Omitting the keys argument is the opposite failure: the
implementations read it directly and raise a `KeyError`.
