# Running a model

Use this when you set the `trainer` block of a config, or need to know what the
loop does between your mappings and a checkpoint.

## What a step does

For each operation, in declaration order:

1. Run its mappings, merging the batch with everything produced so far.
2. Compute its losses, sum them, and backward through them.
3. Step its optimizer, once the accumulation boundary is reached.
4. Compute its metrics.

Both losses and metrics see the full dict, so a metric may read another
operation's output. Optimizer steps are per operation, which is what keeps two
operations from updating each other's parameters.

## What an epoch does

Validation runs **before** training within each epoch iteration, so epoch 0
reports the untrained baseline, and it runs on rank zero only. Cadence is
`check_val_every_n_epoch`; a run without a validation dataset skips it.

Training then runs the epoch, and the loop repeats until `max_epochs`, or until a
stop is requested — `max_samples`, `max_val_samples` and a SIGINT handler all
request a graceful stop that still finalizes the checkpoint.

## The trainer block

| Key | Effect |
| --- | --- |
| `max_epochs` | Epoch limit; unbounded when unset |
| `max_samples`, `max_val_samples` | Sample limits that request a stop |
| `devices`, `accelerator`, `strategy`, `precision` | Execution placement; single-node DDP is the supported distributed shape |
| `log_every_n_steps` | Metric logger cadence, default 25 |
| `check_val_every_n_epoch` | Validation cadence, default 1 |
| `grad_accum_steps` | Batches per optimizer step |
| `gradient_clip_val`, `gradient_clip_algorithm` | Trainer-wide clipping |
| `seed`, `float32_matmul_precision`, `compile_model`, `autocast_dtype` | Determinism and performance |
| `checkpoint_interval_seconds`, `checkpoint_artifact_path`, `save_last` | Checkpointing, usually set by the caller rather than the config |

Unknown keys are rejected with a `TypeError` naming them, so the trainer block is
validated by this package and not silently ignored.

## Checkpoints

Periodic checkpoints are time-based, written at most every
`checkpoint_interval_seconds` and named by epoch and step. A final `last.ckpt` is
written when the loop ends if `save_last` is on. Both go to the caller's
`checkpoint_path` and are handed to the logger's `log_artifact`, whose return
value — for W&B, the artifact handle — the trainer keeps as
`last_checkpoint_artifact`.

A checkpoint carries the model state (minus `exclude_from_checkpoint`), the
optimizers, the schedulers with their counters, and the RNG state. Resuming
restores all of it.

## Inference

`Trainer.predict(model, dataset)` runs every operation per batch under `no_grad`
in `predict` mode, with no optimizers, losses, metrics or logging, and emits
Lightning-shaped callback events: `on_predict_start`,
`on_predict_batch_end(output, batch, batch_idx)`, `on_predict_end`. Each callback
receives the merged input and output dict, so pass-through keys such as sample
ids are available alongside predictions.

## Distributed

Single-node DDP, entered when `devices > 1`. Rank zero does validation, logging
and checkpoint writing; other ranks participate in the loop and synchronize at
barriers. Loggers are wrapped so their calls forward only on rank zero.

Launching the processes is the caller's job — the trainer expects to already be
inside them.

## Gotchas

- A logger is constructed before it is passed in, and a backend-bound logger
  expects its run to exist. Build it after opening the run, not before.
- `max_samples` counts training samples, so a small value combined with a large
  validation set still pays for validation first.
- `save_last` writes on rank zero only; a distributed run where rank zero exits
  early leaves no final checkpoint.
