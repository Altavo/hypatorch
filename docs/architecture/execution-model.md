# Execution model

How a declared model becomes a running training loop. The stages are ordinary;
they are written down because the boundary between what the config declares and
what the loop does is not visible from either side alone.

Writing a model is documented in
[`hypatorch/agent_docs/`](../../hypatorch/agent_docs/). This page is for changing
the package.

## From config to graph

```
config
  submodules  -> setattr(model, name, module) ......... nodes become attributes
  operations  -> self.operations ...................... one optimizer scope each
      mappings -> self.mappings[op] .................... the edges
      losses   -> self.losses[op]
      metrics  -> self.metrics[op]
      logging  -> self.logging[op]
  union of optimize_submodules
              -> everything else is frozen ............ requires_grad = False
```

`_get_content` pivots the operation dicts into per-key views, which is why a
missing key in one operation is a default rather than an error: the model asks
for `losses` across all operations and gets an empty list where none was declared.

## One step

```
input_dict (the batch)
  for each operation, in declaration order:
      forward(input_dict + output_dict, operation_name, mode)
          for each mapping:
              validate key maps against the real signature
              gather inputs, run under no_grad if frozen / not calculate_grad / not train
              map returned values to output keys, refusing to overwrite
      compute_loss   -> sum of weighted assessments -> backward
      optimizer step, on accumulation boundaries
      compute_metrics
  logger.step_done()
```

The shared dict is assembled per call with `shared_dict(input_dict,
output_dict)`, so a mapping always sees the batch plus everything produced so
far, and `update_output` enforces write-once across operations.

## Signature introspection

`validate_io_keys` compares a mapping's key maps against
`get_input_variable_names` (from `inspect.getfullargspec`) and
`get_output_variable_names`, which parses the function source with `ast` and reads
the names in its single `return`. Bare names are reported by name; expression
returns become positional placeholders that `_run_submodule` maps in declaration
order.

This is the package's most unusual dependency: **the source of a submodule's
called function is part of its interface**. A module whose forward is generated,
decorated in a way that hides its source, or otherwise not introspectable cannot
be wired by name, and a second `return` statement is rejected outright.

## The loop

`_training_loop` alternates validation and training per epoch, validation first,
rank zero only, on the `check_val_every_n_epoch` cadence. Stop requests —
`max_samples`, a signal handler — set a flag checked between phases, so a stop
still reaches `_finalize_last_checkpoint`.

Checkpointing is time-based rather than epoch-based: `_maybe_save_periodic_checkpoint`
compares a monotonic clock against `checkpoint_interval_seconds`. Every saved
file is offered to `logger.log_artifact`, and the handle that call returns is kept
as `last_checkpoint_artifact` for a caller to reference.

## Loggers

`DataLogger` accumulates values per step and reports on `step_done` /
`epoch_done` at the configured cadence. Metrics enter it from the model, not the
trainer: `_handle_assessments` calls `log_value(f"{mode}/{name}", ...)` for every
computed assessment, and `Model.log_data` forwards configured `logging` entries
to the named logger method. The trainer only decides when a step and an epoch
are over, and wraps the logger so a distributed run forwards on rank zero only.

`WandbLogger` and `MLflowLogger` bind to an already-open run of their backend;
`ConsoleLogger` binds to nothing. Opening and closing that run belongs to the
caller.
