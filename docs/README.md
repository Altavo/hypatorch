# Documentation

How to use this package is documented inside it, in
[`hypatorch/agent_docs/`](../hypatorch/agent_docs/). That bundle ships in the
wheel, so a project pinned to a version reads the documentation of that version.

| Read | For |
| --- | --- |
| [agent_docs/OVERVIEW.md](../hypatorch/agent_docs/OVERVIEW.md) | The dataflow-graph model, and where each decision belongs |
| [agent_docs/model-config.md](../hypatorch/agent_docs/model-config.md) | Submodules, operations, mappings |
| [agent_docs/writing-a-submodule.md](../hypatorch/agent_docs/writing-a-submodule.md) | Making a module callable from a config |
| [agent_docs/losses-and-metrics.md](../hypatorch/agent_docs/losses-and-metrics.md) | Assessments, masking, logging |
| [agent_docs/optimizers-and-schedulers.md](../hypatorch/agent_docs/optimizers-and-schedulers.md) | What an operation updates, and how |
| [agent_docs/trainer.md](../hypatorch/agent_docs/trainer.md) | Epochs, validation, checkpoints, devices |

This directory keeps what does not ship.

| Document | For |
| --- | --- |
| [architecture/execution-model.md](architecture/execution-model.md) | How a declared model becomes a loop, and why submodule source is part of its interface |

Rule of thumb for where a new page belongs: if someone writing a model config
would write different YAML after reading it, it belongs in `agent_docs/`. If it
explains how the machinery works underneath, it belongs here.
