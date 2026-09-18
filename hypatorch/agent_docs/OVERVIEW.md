# hypatorch

Use this when you write or change a model configuration, a submodule that a
configuration wires up, or the trainer settings a run executes with.

## Mental model

A hypatorch model is a **declarative dataflow graph over a shared dict**. You do
not write a `forward`; you declare which submodule reads which keys and writes
which keys, and the graph runs in declaration order.

```
submodules   named torch.nn.Modules — the nodes
operations   named training scopes — each owns one optimizer and its own graph
mappings     the edges: submodule, its input keys, its output keys
```

Everything flows through one dict per batch. A mapping's `inputs` renames dict
keys to the submodule's argument names, and its `outputs` renames the submodule's
returned values to dict keys. Nothing is positional at the config level and
nothing is global: a key exists because some mapping produced it.

Two consequences worth holding onto:

- **Keys are write-once.** Neither a submodule nor an operation may overwrite a
  key already in the dict. Rerunning a submodule under a different name means
  giving its outputs different keys.
- **The graph is validated against real signatures.** Input and output key maps
  are checked against the submodule's actual arguments and its actual `return`
  statement before anything runs, so a wiring mistake fails at the first batch
  rather than as a shape error later.

An operation is the unit of optimization, not just of grouping: one optimizer,
one set of submodules to update, its own losses and its own metrics. A model with
two operations trains two graphs per step, in order.

## Boundary

This package owns:

- The model abstraction: submodules, operations, mappings, and the dict they
  share.
- Losses and metrics as configured assessments, with masking and length
  harmonization.
- Optimizers, schedulers and gradient clipping, per operation.
- The training loop, validation cadence, checkpoint files, distributed setup,
  and the metric loggers.

It does not own:

- Where data comes from or what a batch contains. It consumes an iterable of
  dicts.
- The tracking run a logger writes into, and the lifecycle around it. A logger
  binds to a run someone else opened.
- Recipes, dispatch, or how a run reaches a container.
- What a submodule computes. A submodule is any `torch.nn.Module` that satisfies
  the rules in [writing-a-submodule.md](writing-a-submodule.md).

## Entry points

| Doing | Use |
| --- | --- |
| Declare a model | `_target_: hypatorch.Model` with `submodules` and `operations` |
| Declare a loss or metric | `_target_: hypatorch.HypaAssessment`, or a ready-made one from `hypatorch.losses` |
| Run training | `hypatorch.Trainer(...).train(model, ...)` |
| Run inference | `hypatorch.Trainer(...).predict(model, dataset)` |
| Emit metrics | `hypatorch.logger.ConsoleLogger` / `WandbLogger` / `MLflowLogger` |

## Always

- Give every submodule that must be trained to an operation's
  `optimize_submodules`. Anything no operation optimizes is **frozen** —
  `requires_grad = False` — at construction. Freezing is the default, not an
  opt-in.
- Name output keys for what they are, not for what produced them. The next
  mapping reads them by name and nothing namespaces them.
- Set `calculate_grad: false` on a mapping whose submodule only needs to run, not
  to train. It wraps that call in `no_grad`.
- `calculate_grad: true` on a mapping whose submodule does not appear in
  `optimize_submodules` passes gradients without getting trained itself.

## Never

- Never let two mappings write the same output key, in one operation or across
  operations in a step. That is an error, not a last-write-wins.
- Never give a submodule more than one `return` statement; the output names are
  read from it. See [writing-a-submodule.md](writing-a-submodule.md).
- Never assume a metric's name is unique on its own — assessments are logged as
  `<mode>/<name>`, so the same name in train and validation is distinguishable,
  but two assessments sharing a name within one mode are not.

## Where to go next

| Deciding | Read |
| --- | --- |
| Submodules, operations, mappings, the graph | [model-config.md](model-config.md) |
| Making a module usable from a config | [writing-a-submodule.md](writing-a-submodule.md) |
| Losses, metrics, masking, custom logging | [losses-and-metrics.md](losses-and-metrics.md) |
| Optimizers, schedulers, clipping, freezing | [optimizers-and-schedulers.md](optimizers-and-schedulers.md) |
| Running it: epochs, validation, checkpoints, devices | [trainer.md](trainer.md) |
