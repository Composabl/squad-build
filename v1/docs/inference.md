# Inference (Running a Trained Policy)

After training, checkpoints are saved under `checkpoint_uri/{skill-name}/` (default: `/tmp/amesa/{skill-name}/`). This page covers how to load a checkpoint and run the policy in a standalone script — outside of the training loop.

## Do not use `PPO.from_checkpoint`

`PPO.from_checkpoint(path)` (and the equivalent `Algorithm.from_checkpoint`) restores the full training stack, including rollout workers. Each worker calls `gym.make(<env_id>)` where `<env_id>` is the AMESA-internal gymnasium registration (`'composabl'`). That registration only exists while `ServerAmesa` and the AMESA training loop are running, so the call raises:

```
gymnasium.error.NameNotFound: Environment 'composabl' doesn't exist
```

## Use `Policy.from_checkpoint` instead

`Policy.from_checkpoint` loads only the saved network weights with no worker creation. It is safe to call from any plain Python process with no AMESA server running:

```python
from ray.rllib.policy.policy import Policy

policies = Policy.from_checkpoint("/tmp/amesa/my-skill/")
policy   = policies["default_policy"]

action, _, _ = policy.compute_single_action(obs, explore=False)
```

`compute_single_action` returns a `(action, state, info)` triple — unpack accordingly.

## Full standalone inference pattern

```python
import os
import numpy as np
from ray.rllib.policy.policy import Policy

CHECKPOINT_PATH = "/tmp/amesa/my-skill/"

policies = Policy.from_checkpoint(CHECKPOINT_PATH)
policy   = policies["default_policy"]

# obs must match the observation space the skill was trained on
obs = np.array([0.5, 1.0, -0.3], dtype=np.float32)

action, _, _ = policy.compute_single_action(obs, explore=False)
print("Action:", action)
```

## Finding the checkpoint path

The default save location is `/tmp/amesa/<skill-name>/`. To use a custom path, set `checkpoint_uri` on the `Skill`:

```python
skill = Skill("my-skill", MyTeacher, training_cycles=100)
skill.set_checkpoint_uri("/path/to/checkpoints")
```

Or via the training job JSON:

```json
{
  "learning": {
    "checkpoint_uri": "/path/to/checkpoints"
  }
}
```

The trainer appends the skill name as a subdirectory, so the final checkpoint lives at `{checkpoint_uri}/{skill-name}/`.

---

## ⚠️ Quirks

**`Policy.from_checkpoint` requires the same Ray version** — The checkpoint format is tied to the Ray/RLlib version used during training. Load it with the same version.

**`explore=False` for deterministic evaluation** — Pass `explore=False` to `compute_single_action` to disable stochastic sampling and get the greedy action from the policy.

**obs dtype must match training** — RLlib is strict about observation dtypes. If your sensor space is `Box(dtype=float32)`, pass `numpy.float32` arrays, not `float64`.
