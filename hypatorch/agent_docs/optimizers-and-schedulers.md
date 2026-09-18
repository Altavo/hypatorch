# Optimizers, schedulers and freezing

Use this when you decide what an operation updates and how its learning rate
moves.

## One optimizer per operation

```yaml
operations:
  update_encoder:
    optimizer:
      _partial_: true
      _target_: torch.optim.Adam
      lr: 0.01
    optimize_submodules: [image_encoder]
```

The optimizer is declared `_partial_` and called with the parameters collected
from `optimize_submodules`. An operation always has exactly one; more than one
optimizer means more than one operation.

## Freezing is by omission

At construction the model computes the union of every operation's
`optimize_submodules`, and everything else gets `requires_grad = False`. There is
no `freeze:` key, and adding a submodule to the config without naming it in an
operation silently makes it frozen — which is usually what you want for a
pretrained encoder and never what you want for a head you forgot to wire.

Frozen is about parameters, not about gradients. Whether a call builds a graph is
decided by its mapping's `calculate_grad` and the mode, never by frozen-ness:

- `calculate_grad: false` — runs under `no_grad`, costs no activation memory, and
  nothing upstream of it can be trained through this call.
- `calculate_grad: true` — builds the graph, so gradients pass **through** a
  frozen submodule to trainable modules upstream while the frozen one still
  updates nothing. This is how a frozen encoder or a pretrained loss network
  trains what feeds it, and it costs the activation memory of that branch.

## Schedulers

```yaml
lr_scheduler:
  scheduler:
    _partial_: true
    _target_: torch.optim.lr_scheduler.StepLR
    step_size: 10
  interval: epoch
  frequency: 1
```

The scheduler is built from its partial with the operation's optimizer, then
wrapped so that `interval` (`epoch` or `step`) and `frequency` decide when
`.step()` is actually called: the wrapper counts and fires every `frequency`
units. A scheduler with `interval: step` ignores epoch boundaries entirely, and
vice versa — a mismatch here shows up as a learning rate that never moves.

Scheduler state, including its counters, is saved and restored with the
checkpoint.

## Gradient clipping

`gradient_clipping` on an operation is a partial called with that operation's
parameters, applied before the optimizer step. The trainer also has its own
`gradient_clip_val` and `gradient_clip_algorithm`; the per-operation form is the
one to use when two operations need different limits.

## Accumulation

`grad_accum_steps` on the trainer divides each loss before backward and steps the
optimizer only on complete accumulation boundaries. It applies to every
operation; it is not configurable per operation.
