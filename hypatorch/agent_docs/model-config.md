# Model configuration

Use this when you declare a model: which modules exist, how data flows between
them, and what each training scope optimizes.

## The shape

```yaml
_target_: hypatorch.Model

defaults:
  - /submodules/image_encoder: lenet
  - /submodules/resize: bilinear

submodules_eval:
exclude_from_checkpoint:

operations:
  update_encoder:
    optimizer:
      _partial_: true
      _target_: torch.optim.Adam
      lr: 0.01

    optimize_submodules:
      - image_encoder

    mappings:
      - resize:
          inputs:
            x: image
          outputs:
            x: scaled_image
          calculate_grad: false
      - image_encoder:
          inputs:
            x: scaled_image
          outputs:
            output: logits
          calculate_grad: true

    losses:
      - _target_: hypatorch.HypaAssessment
        assessment:
          _target_: torch.nn.CrossEntropyLoss
          reduction: mean
        name: mean_cross_entropy
        inputs:
          input: logits
          target: class
```

`submodules` maps a name to a `torch.nn.Module`; each becomes an attribute of the
model under that name. Composing them from a config group, as above, is the usual
way — the name on the left is what mappings refer to.

## Mappings are the graph

A mapping is one entry per submodule invocation:

| Key | Meaning |
| --- | --- |
| `inputs` | `{argument_name: dict_key}` — which dict key feeds which argument |
| `outputs` | `{returned_name: dict_key}` — which returned value becomes which dict key |
| `calculate_grad` | `false` runs the call under `no_grad` |
| `apply` | Modes this mapping runs in; omitted means always. Canonical names are `train` / `val` / `predict` |
| `fn` | Call this method instead of the module's `__call__` |

Mappings run in list order. Each one sees the batch plus everything earlier
mappings produced, and its outputs are added to that dict. A submodule may appear
in several mappings — under a different set of output keys each time.

An output key map value may be a list, which unpacks one returned sequence into
several keys.

`inputs` must cover every required argument of the target function; optional ones
may be omitted. Both key maps are validated against the real signature and the
real `return` statement before the first batch runs.

## Input entries

An input is a dict key by default, and a value only when you say so:

```yaml
      - quantizer_kmeans:
          inputs:
            x: wavlm_large_-1
            x_len: wavlm_large_-1_len
            kmeans_update:
              value: false
              key_map: false
          outputs:
            encoding_indices: indices_su
            quantized: su
          calculate_grad: false
```

| Form | Passes |
| --- | --- |
| `arg: some_key` | The dict value at `some_key` |
| `arg: {value: X, key_map: false}` | `X` itself, as a literal |
| `arg: {value: some_key, key_map: true}` | The dict value at `some_key`, spelled out |
| `arg: [entry, entry]` | A list, each entry resolved by the rules above |

The literal form is how a flag, a threshold or any other constant reaches a
submodule without occupying a key in the shared dict. Without `key_map: false` a
bare string is looked up, and a miss raises an error that spells out the literal
form — so the common mistake is self-correcting.

## Operations are optimizer scopes

Each entry under `operations` owns:

- `optimizer` — a partial, called with the parameters it should update
- `optimize_submodules` — which submodules those parameters come from
- `mappings` — that operation's graph
- `losses`, `metrics` — assessments computed after the graph
- optionally `lr_scheduler`, `gradient_clipping`, `logging`

Every operation runs on every step, in declaration order, and the output of one
is visible to the next. That is how a two-stage setup — say a frozen encoder
scored by one operation and a head trained by another — shares intermediates
without recomputing them.

A submodule named by **no** operation's `optimize_submodules` is frozen at
construction: its parameters get `requires_grad = False`. Freezing is therefore
something you get by omission.

## Write-once keys

Neither a mapping nor an operation may write a key that already exists in the
dict, and both raise naming the offending keys. Sequence matters instead of
precedence: to transform a value in place, give the result a new key.

## The other model-level keys

| Key | Effect |
| --- | --- |
| `submodules_eval` | Submodules kept in `eval()` even while the model trains — for a frozen batch-norm or dropout-bearing submodule/model |
| `exclude_from_checkpoint` | Submodules dropped from the saved state dict; on load, they are skipped rather than demanded |
| `checkpoints` | A list of `{path, prefix_rm, prefix_add}` entries loaded into the model, with prefix rewriting for weights trained under a different attribute name |

`checkpoints` entries are attempted strictly and retried with `strict=False` on
failure, with a warning. A partial load therefore succeeds quietly — check the
warning if a finetune starts from a suspiciously high loss.

## Gotchas

- A submodule named in a mapping but absent from `submodules` fails with "Forgot
  to define it?" at the first forward, not at construction.
- `apply` is checked at construction: a value that could never match a mode is a
  `ValueError` naming the mapping, not a mapping that silently never runs.
- A list matches by membership and a bare string by containment, so
  `apply: [train]` means exactly training while `apply: evaluation` runs in
  validation — it contains `val`. Prefer the list form.
- The batch dict and the output dict are merged for each call, so a mapping can
  read a raw batch key and a computed key in the same `inputs`.
