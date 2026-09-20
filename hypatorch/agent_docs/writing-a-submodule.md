# Writing a submodule

Use this when a model config must call a module you wrote, and the wiring has to
validate.

## The rules

A submodule is a plain `torch.nn.Module`, with three constraints that come from
how the graph is validated:

1. **Every return must agree in arity.** The output names are read from the
   source of the called function, so several returns are fine as long as each
   provides the same number of values in the same order — the names come from the
   last one, the values from whichever branch ran. Returns of differing length
   are rejected when the names are read. A return inside a nested function is
   ignored.
2. **Return bare names, or nothing but expressions.** `return logits, features`
   maps by name. A single expression — `return torch.cat(x, dim=-1)` — has no
   name and maps positionally. Mixing the two in one tuple is rejected, because
   the two mappings cannot be combined; see below.
3. **Name arguments for what they are.** A mapping's `inputs` keys are the
   argument names, and every required argument must appear. Arguments with
   defaults are optional in the config.
4. Use `*_len` for the key that carries a length tensor. Convention only.
```python
class Encoder(torch.nn.Module):
    def forward(self, x, lengths=None):
        features = self.net(x)
        logits = self.head(features)
        return logits, features
```

```yaml
- encoder:
    inputs:
        x: time_series
        x_len: time_series_len
    outputs:
        logits: class_logits
        features: embedding
```

## Never mix named and expression returns

```python
def forward(self, x):
    return torch.cat([x, x], dim=-1), x     # one expression, one name
```

Name matching takes the order from the return and positional fallback takes it
from the configured key order, so a mapping that half-matches would assign values
to the wrong keys. That is refused with a message naming which keys matched which
way. Assign the expression to a variable and return the names:

```python
def forward(self, x):
    wide = torch.cat([x, x], dim=-1)
    return wide, x
```

## What is checked, and when

Before each call, the configured key maps are validated against the introspected
signature: every required input must be present, and output keys must either
match the returned names or be no more numerous than the returned values. A
mismatch raises a message naming the submodule, its class, and both key sets.

The number of returned values is also checked against the number the source
declares, so returning a tuple of the wrong length is caught immediately.

## Calling something other than forward

By default a mapping calls the module itself — `module(**inputs)` — so
`nn.Module.__call__` runs, and with it any registered forward hooks, before
`forward`. The signature and the return that are validated are `forward`'s.

`fn: encode` names a different method, and that method is then **called
directly**. It is validated by the same rules — matching arity across returns,
named values, required arguments covered — but it bypasses `__call__`, so forward
hooks registered on that submodule do not fire.

That matters for a module whose behaviour lives partly in a hook: normalization
bookkeeping, activation capture, or anything registered by an outside tool.
Reach for `fn` for a genuinely separate entry point, not to select between two
code paths a hook is expected to observe.

## Gradients and modes

`calculate_grad: false` wraps the call in `no_grad`, and so does any mode other
than training. Frozen-ness does not: a submodule no operation optimizes still
builds a graph when its mapping says `calculate_grad: true`, which is what lets
gradients reach trainable modules upstream of it. Set it to `false` only when
nothing before the call needs to be trained through it.

The module's own `training` flag follows the model's, except for names listed in
`submodules_eval`, which stay in `eval()`.

## Gotchas

- Source introspection means the function must be introspectable: a module whose
  `forward` is generated, wrapped by a decorator that hides the source, or
  compiled away cannot be wired this way.
- A returned value that is a single tensor still goes through the same path; the
  config maps one output key to it.
- Returning a dict is not a supported shape. Return values positionally, or split
  the module.
