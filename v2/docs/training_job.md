# Training Job

A training job is a single JSON file that fully describes an agent, its environment, compute target, and training configuration. It is the serialized form produced by `Agent.export()` / `Agent.to_json()` and consumed by `Agent.from_json()` at job startup.

---

## Top-level structure

| Field                      | Type                     | Default                | Description                                        |
| -------------------------- | ------------------------ | ---------------------- | -------------------------------------------------- |
| `license`                  | string                   | —                      | AMESA license key                                  |
| `active_working_directory` | string                   | `/tmp/amesa`           | Working dir for checkpoints and artifacts          |
| `agent`                    | **AgentSchema**          | required               | Serialized agent (see [Agent](#agent-agentschema)) |
| `env`                      | **EnvConfig**            | —                      | Simulation environment config                      |
| `target`                   | **TargetConfig**         | `local:localhost:1337` | Compute target (see [Target](#target))             |
| `trainer`                  | **TrainerConfigBase**    | —                      | Training hyperparameters                           |
| `meta`                     | **Meta**                 | —                      | Project/org identifiers                            |
| `flags`                    | **FlagsConfig**          | —                      | Feature flags                                      |
| `post_processing`          | **PostProcessingConfig** | —                      | Record/benchmark config                            |

---

## `env`

```json
{ "name": "amesa", "init": {} }
```

`init` is merged into the sim's `env_init` payload at startup.

---

## `target`

The `target` key must contain exactly one child key: `v2`. v2 uses an event-based Redis Streams architecture.

### `v2`

See [trainer.md](trainer.md) for the full field reference.

```json
{
  "target": {
    "v2": {
      "redis_url": "redis://localhost:6379",
      "sim_image": "my-sim:latest",
      "initial_replicas": 4,
      "num_episode_managers": 2,
      "enable_ppo_training": true,
      "ppo_training_samples": 4000,
      "enable_perceptors": true,
      "sim_node_local": true,
      "perceptor_node_local": true,
      "skill_node_local": true,
      "episode_manager_local": true
    }
  }
}
```

---

## `trainer`

```json
{
  "ray": { "address": "auto" },
  "training_cycles": 100,
  "dependencies": [
    { "url": "https://example.com/pkg.whl", "version": "1.0.0", "type": "pip" }
  ]
}
```

| Field             | Default  | Description                                                                   |
| ----------------- | -------- | ----------------------------------------------------------------------------- |
| `ray.address`     | `"auto"` | Ray cluster address; `"auto"` connects to an existing cluster or starts one   |
| `training_cycles` | —        | Default PPO iterations per outer training cycle (can be overridden per skill) |
| `dependencies`    | `[]`     | Python packages to install before training starts                             |

---

## `meta`

```json
{ "project_id": "my-project", "org_id": "my-org" }
```

---

## `post_processing`

Optional post-training actions. Both sub-keys are optional.

```json
{
  "post_processing": {
    "record": {
      "avi_file_name": "record.avi",
      "max_frames": 120
    },
    "benchmark": {
      "file_name": "benchmark.json",
      "num_episodes_per_scenario": 1,
      "num_steps_per_episode": 3600
    }
  }
}
```

### `record`

| Field           | Default        | Description              |
| --------------- | -------------- | ------------------------ |
| `avi_file_name` | `"record.avi"` | Output AVI filename      |
| `max_frames`    | `120`          | Maximum frames to record |

### `benchmark`

| Field                       | Default            | Description                         |
| --------------------------- | ------------------ | ----------------------------------- |
| `file_name`                 | `"benchmark.json"` | Output benchmark results filename   |
| `num_episodes_per_scenario` | `1`                | Episodes to run per scenario        |
| `num_steps_per_episode`     | `3600`             | Maximum steps per benchmark episode |

---

## `flags`

```json
{
  "dry_run_enabled": false,
  "dry_run_output_dir": "/tmp/amesa-dry-run",
  "dry_run_output_file": "output.json",
  "print_debug_info": false,
  "is_eula_agreed": false,
  "seed": 1706,
  "deterministic_training": true
}
```

| Field                    | Default                | Description                                                               |
| ------------------------ | ---------------------- | ------------------------------------------------------------------------- |
| `dry_run_enabled`        | `false`                | Run without training; useful for validating job JSON                      |
| `dry_run_output_dir`     | `"/tmp/amesa-dry-run"` | Directory for dry-run output                                              |
| `dry_run_output_file`    | `"output.json"`        | Filename for dry-run output                                               |
| `print_debug_info`       | `false`                | Emit verbose debug logs                                                   |
| `is_eula_agreed`         | `false`                | Must be `true` (or set `AMESA_EULA_AGREED=1`) to proceed                  |
| `seed`                   | `1706`                 | RNG seed for reproducible training; falls back to `1706` if `null`        |
| `deterministic_training` | `true`                 | Fix the RNG seed for reproducible results (uses `seed` value when `true`) |

---

## `agent` (AgentSchema)

The serialized form of an `Agent`. Produced by `Agent.export()` / `Agent.to_json()` and loaded by `Agent.from_json()`.

```json
{
  "id": "<uuid>",
  "version": "local",
  "sensors": [],
  "perceptors": [],
  "skills": [],
  "skill_selectors": [],
  "skills_coordinated": [],
  "skill_groups": [],
  "environment_name": null,
  "skill_to_train": null,
  "graph": {}
}
```

> **v2 note:** `skills_coordinated` is not supported in v2. Include it as an empty list.

### `sensors[]`

```json
{
  "name": "my-sensor",
  "description": "...",
  "lambda_str": "lambda sensors: sensors['x']",
  "normalize": false,
  "normalizer": null
}
```

See [sensors.md](sensors.md) for normalization options.

### `perceptors[]`

```json
{ "name": "my-perceptor", "description": "...", "impl": {}, "config": "..." }
```

See [perceptors.md](perceptors.md). When using v2 target, set `enable_perceptors: true` in the `v2` target config or perceptor outputs will be absent from all observations.

### `skills[]` — controller (`type: "SkillController"`)

```json
{
  "name": "my-controller",
  "type": "SkillController",
  "config": {
    "remote_address": null,
    "impl_cls": {
      "cls_name": "MyController",
      "cls_module": "my_agent.controller",
      "cls_src": "<base64-pickle>",
      "cls_deps": []
    },
    "impl_cls_data": {
      "guidance": null,
      "goals": [],
      "constraints": null
    },
    "model_io": {},
    "scenarios": [],
    "scenarios_current_idx": 0
  }
}
```

See [skills/controller.md](skills/controller.md).

### `skills[]` — DRL skill (`type: "SkillTeacher"`)

Adds `learning`, `resources`, and `model` blocks to the config:

```json
{
  "name": "my-skill",
  "type": "SkillTeacher",
  "config": {
    "remote_address": null,
    "impl_cls": {
      "cls_name": "...",
      "cls_module": "...",
      "cls_src": "<base64-pickle>",
      "cls_deps": []
    },
    "impl_cls_data": { "guidance": null, "goals": [], "constraints": null },
    "model_io": {},
    "scenarios": [],
    "scenarios_current_idx": 0,
    "learning": {
      "training_cycles": null,
      "train_batch_size": 4000,
      "replay_buffer_size": 50000,
      "rl_algo": "PPO",
      "gamma": null,
      "learning_rate": null,
      "grad_clip": null,
      "grad_clip_by": null,
      "train_batch_size_per_learner": null,
      "needs_warmup": false
    },
    "resources": {
      "workers": 1,
      "learner_workers": 0,
      "envs_per_worker": 1,
      "num_cpus_per_worker": 1.0,
      "num_gpus_per_worker": 0.0,
      "num_cpus_per_learner_worker": 1.0,
      "num_gpus_per_learner_worker": 0.0,
      "remote_worker_envs": false,
      "rollout_fragment_length": "auto",
      "sample_timeout_s": 500.0,
      "sample_timeout_s_max": 1000.0
    },
    "model": {
      "checkpoint_uri": "/tmp/amesa",
      "fc_layers": [256, 256]
    },
    "use_image_nn": false
  }
}
```

#### `learning` field reference

| Field                          | Default                           | Description                                                   |
| ------------------------------ | --------------------------------- | ------------------------------------------------------------- |
| `training_cycles`              | `null` (inherits trainer default) | PPO update iterations per outer training cycle for this skill |
| `train_batch_size`             | `4000`                            | Samples collected before each PPO update                      |
| `replay_buffer_size`           | `50000`                           | Replay buffer capacity                                        |
| `rl_algo`                      | `"PPO"`                           | RL algorithm                                                  |
| `gamma`                        | `null`                            | Discount factor (uses algo default if null)                   |
| `learning_rate`                | `null`                            | Learning rate (uses algo default if null)                     |
| `grad_clip`                    | `null`                            | Gradient clipping value                                       |
| `grad_clip_by`                 | `null`                            | Gradient clipping mode (`"global_norm"`, etc.)                |
| `train_batch_size_per_learner` | `null`                            | Per-learner batch size (multi-learner setups)                 |
| `needs_warmup`                 | `false`                           | Collect warmup episodes before training begins                |

#### `resources` field reference

| Field                         | Default  | Description                                                    |
| ----------------------------- | -------- | -------------------------------------------------------------- |
| `workers`                     | `1`      | Rollout workers for this skill                                 |
| `learner_workers`             | `0`      | Dedicated learner workers (0 = in-process)                     |
| `envs_per_worker`             | `1`      | Environments per rollout worker                                |
| `num_cpus_per_worker`         | `1.0`    | CPUs allocated per rollout worker                              |
| `num_gpus_per_worker`         | `0.0`    | GPUs allocated per rollout worker                              |
| `num_cpus_per_learner_worker` | `1.0`    | CPUs allocated per learner worker                              |
| `num_gpus_per_learner_worker` | `0.0`    | GPUs allocated per learner worker                              |
| `remote_worker_envs`          | `false`  | Run environments in separate remote subprocesses per worker    |
| `rollout_fragment_length`     | `"auto"` | Steps per rollout fragment (`"auto"` or an integer)            |
| `sample_timeout_s`            | `500.0`  | Seconds before a rollout sample times out                      |
| `sample_timeout_s_max`        | `1000.0` | Maximum timeout ceiling; grows by 10% increments from the base |

#### `model` field reference

| Field            | Default        | Description                                        |
| ---------------- | -------------- | -------------------------------------------------- |
| `checkpoint_uri` | `"/tmp/amesa"` | Path to save/load model checkpoints                |
| `fc_layers`      | `[256, 256]`   | Fully-connected layer sizes for the policy network |

### `skill_selectors[]` (`type: "SkillSelector"`)

Same structure as a DRL skill config, plus ordering and children:

```json
{
  "name": "my-selector",
  "type": "SkillSelector",
  "config": {
    "remote_address": null,
    "impl_cls": { ... },
    "impl_cls_data": { "guidance": null, "goals": [], "constraints": null },
    "model_io": {},
    "scenarios": [],
    "scenarios_current_idx": 0,
    "learning": { ... },
    "resources": { ... },
    "model": { ... },
    "order_type": "fixed",
    "repeat_type": "once",
    "children": ["skill-name-a", "skill-name-b"]
  }
}
```

| Field            | Values                    | Description                                                                    |
| ---------------- | ------------------------- | ------------------------------------------------------------------------------ |
| `remote_address` | string \| `null`          | URL of a remotely-hosted skill; when set, `impl_cls` is ignored                |
| `order_type`     | `"fixed"` \| `"variable"` | `"fixed"` runs children in declaration order; `"variable"` selects dynamically |
| `repeat_type`    | `"once"` \| `"repeat"`    | `"once"` exits after all children complete; `"repeat"` loops                   |
| `children`       | `string[]`                | Ordered list of child skill names (must match names in `skills[]`)             |

See [skills/selector.md](skills/selector.md).

### `model_io` (shared by skills and selectors)

Describes the sensor and action spaces for a skill.

```json
{
  "sensor_space": {
    "type": "Dict",
    "spaces": {
      "x": {
        "type": "Box",
        "low": -1.0,
        "high": 1.0,
        "shape": [1],
        "dtype": "float32"
      }
    },
    "order": ["x"]
  },
  "action_space": { "type": "Discrete", "n": 3 },
  "custom_sensor_space": false,
  "custom_action_space": false,
  "filtered_sensor_space": []
}
```

#### Space types

| Type       | Fields                                  | Description                                              |
| ---------- | --------------------------------------- | -------------------------------------------------------- |
| `Box`      | `low`, `high`, `shape` (int[]), `dtype` | Continuous bounded space                                 |
| `Discrete` | `n`                                     | Integer action space `[0, n)`                            |
| `Dict`     | `spaces` (object), `order` (string[])   | Named composite space; `order` fixes key iteration order |

See [spaces.md](spaces.md) for MultiBinary, MultiDiscrete, and Tuple.

### `graph`

NetworkX adjacency format. Selectors appear as nodes with directed edges to their child skill names.

```json
{
  "directed": true,
  "multigraph": false,
  "graph": [],
  "nodes": [{ "id": "my-selector" }, { "id": "skill-a" }, { "id": "skill-b" }],
  "adjacency": [[{ "id": "skill-a" }, { "id": "skill-b" }], [], []]
}
```

The graph is auto-generated by `Agent.export()`. You do not need to construct it manually.

---

## Producing and loading job JSON

```python
# Serialize
agent = build_agent()
agent.export("./output_dir")          # writes agent.json to a directory
json_str = agent.to_json()            # returns JSON string

# Load
from amesa_core.agent.agent import Agent
agent = Agent.from_json("path/to/agent.json")
```

---

## ⚠️ Quirks

**`skills_coordinated` must be an empty list in v2** — The coordinated skill type (`SkillCoach` / `SkillCoordinatedSet`) is not available in v2. Serialized agents from v1 with coordinated skills cannot be used directly in v2 jobs. See [skills/coordinated.md](skills/coordinated.md).

**`children` names must match exactly** — In `skill_selectors[].config.children`, each name must match the `name` field of a skill in `skills[]`. A mismatch causes a silent failure at graph construction time.

**`cls_src` is a base64-pickled class** — The `impl_cls.cls_src` field is produced automatically by `Agent.export()`. Do not hand-author it. The class must be importable (or self-contained) on the machine running the job.

**`is_eula_agreed` or `AMESA_EULA_AGREED=1`** — The job will not start unless either `flags.is_eula_agreed` is `true` in the JSON or the environment variable `AMESA_EULA_AGREED` is set to `"1"`.

**`enable_perceptors` for v2 target** — If the agent has perceptors and the target is `v2`, set `target.v2.enable_perceptors: true`. Without it, perceptor outputs are silently absent from observations. See [perceptors.md](perceptors.md#v2-training-you-must-opt-in-to-perceptors).
