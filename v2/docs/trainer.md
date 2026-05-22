# Trainer

`Trainer` orchestrates the training loop. For v2 development, it uses a Redis-stream event-based architecture.

## Import

```python
from amesa_train.trainer import Trainer
```

## Basic usage

```python
trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=50)
finally:
    trainer.close()
```

## v2 Config

Pass a dict to `Trainer(config)`. The `"v2"` target key activates the event-based (Redis Streams) training stack.

```python
config = {
    "target": {
        "v2": {
            # --- Connection ---
            "redis_url": "redis://localhost:6379",
            "sim_image": "my-sim-image:latest",

            # --- Parallelism ---
            "initial_replicas": 4,          # sim+skill replica pairs
            "num_episode_managers": 2,

            # --- Training ---
            "enable_ppo_training": True,
            "ppo_training_samples": 4000,

            # --- Components (opt-in) ---
            "enable_perceptors": True,      # required if agent.add_perceptor() is used
            "enable_controllers": False,
            "enable_evaluation": False,
            "enable_historian": False,
            "enable_queue_metrics": False,  # set True to log Redis queue depths

            # --- Locality (True = in-process, False = expect remote Docker/K8s) ---
            "sim_node_local": True,
            "perceptor_node_local": True,
            "skill_node_local": True,
            "episode_manager_local": True,

            # --- Auto-scale ---
            "enable_auto_scale": False,
            "auto_scale_target_steps_per_second": 100.0,
            "auto_scale_min_improvement_ratio": 0.1,
            "auto_scale_max_replicas": 8,

            # --- Advanced ---
            "enable_sim_group": True,
            "enable_remote_skill": False,
            "enable_inference_node": True,
            "batch_size": 4000,
            "max_episodes_in_memory": 50,
            "episode_timeout": 300.0,
        }
    }
}
```

### v2 config field reference

#### Connection

| Field           | Default                    | Description                                                        |
| --------------- | -------------------------- | ------------------------------------------------------------------ |
| `redis_url`     | `"redis://localhost:6379"` | Redis Streams connection URL                                       |
| `sim_image`     | `None`                     | Docker image for sim containers (used when `sim_node_local=False`) |
| `sim_address`   | `None`                     | Pre-existing sim gRPC address; skips spawning a new container      |
| `sim_addresses` | `None`                     | Dict of `{"sim-id": "host:port"}` for multiple pre-running sims    |
| `sim_env_init`  | `None`                     | Extra payload merged into sim `env_init` at startup                |

#### Parallelism

| Field                  | Default | Description                                                 |
| ---------------------- | ------- | ----------------------------------------------------------- |
| `initial_replicas`     | `1`     | Number of sim+skill (+ perceptor if enabled) replica groups |
| `num_episode_managers` | `1`     | Episode managers that shard sim replicas                    |

#### Training

| Field                    | Default | Description                                                 |
| ------------------------ | ------- | ----------------------------------------------------------- |
| `enable_ppo_training`    | `True`  | Run PPO weight updates                                      |
| `ppo_training_samples`   | `4000`  | Steps to collect before each PPO update                     |
| `batch_size`             | `4000`  | Alias for `ppo_training_samples` inside the episode manager |
| `max_episodes_in_memory` | `50`    | Max completed episodes buffered before training             |
| `episode_timeout`        | `300.0` | Seconds before an incomplete episode is discarded           |

#### Components (opt-in)

| Field                   | Default     | Description                                                                                                                                                                                                   |
| ----------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enable_perceptors`     | **`False`** | **Set `True` when `agent.add_perceptor(...)` is used**, or perceptor outputs will be absent from every observation. See [Perceptors → v2 opt-in](../perceptors.md#v2-training-you-must-opt-in-to-perceptors). |
| `enable_controllers`    | `False`     | Enable controller nodes (selector skills with sub-controllers)                                                                                                                                                |
| `enable_evaluation`     | `False`     | Run evaluation episodes alongside training                                                                                                                                                                    |
| `enable_historian`      | `False`     | Emit telemetry to the historian service                                                                                                                                                                       |
| `enable_queue_metrics`  | `True`      | Log Redis queue depth metrics each cycle; set `False` to silence                                                                                                                                              |
| `enable_sim_group`      | `True`      | Use the sim group coordinator                                                                                                                                                                                 |
| `enable_remote_skill`   | `True`      | Allow skill nodes to be remote; set `False` for fully local dev                                                                                                                                               |
| `enable_inference_node` | `True`      | Run a dedicated inference node for policy queries                                                                                                                                                             |

#### Locality

All `*_local` flags: `True` = start the node in-process (default for local dev). `False` = expect the node to already be running remotely (Docker / K8s).

| Field                   | Default                                         |
| ----------------------- | ----------------------------------------------- |
| `sim_node_local`        | `True`                                          |
| `perceptor_node_local`  | `True`                                          |
| `skill_node_local`      | `None` (derives from `not enable_remote_skill`) |
| `episode_manager_local` | `True`                                          |

#### Auto-scale

| Field                                | Default | Description                                           |
| ------------------------------------ | ------- | ----------------------------------------------------- |
| `enable_auto_scale`                  | `False` | Ramp replicas up when steps/second falls below target |
| `auto_scale_target_steps_per_second` | `100.0` | Target throughput before scaling out                  |
| `auto_scale_min_improvement_ratio`   | `0.1`   | Require ≥10% throughput gain to scale again           |
| `auto_scale_max_replicas`            | `8`     | Hard cap on replicas per node type                    |

#### Kubernetes / distributed

| Field                    | Default | Description                                                    |
| ------------------------ | ------- | -------------------------------------------------------------- |
| `cluster_container_mode` | `False` | Set `True` when running inside a K8s job                       |
| `skill_to_train`         | `None`  | Name of the specific skill to train (multi-skill agents)       |
| `environment_name`       | `None`  | K8s environment name                                           |
| `environment_namespace`  | `None`  | K8s namespace                                                  |
| `connection_urls`        | `None`  | Dict of remote endpoint URLs for skills/controllers/perceptors |
| `kubernetes_job_id`      | `None`  | K8s job ID for auto-scale resource requests                    |

#### Redis Stream topics (rarely need changing)

| Field                    | Default                      |
| ------------------------ | ---------------------------- |
| `sim_observation_topic`  | `"amesa.sim.observations"`   |
| `skill_action_topic`     | `"amesa.skill.actions"`      |
| `sim_input_topic`        | `"amesa.sim.input"`          |
| `perceptor_input_topic`  | `"amesa.perceptor.input"`    |
| `perceptor_output_topic` | `"amesa.perceptor.output"`   |
| `inference_topic`        | `"amesa.inference.requests"` |
| `episode_topic`          | `"amesa.episodes"`           |
| `consumer_group`         | `"amesa-training-group"`     |

## Environment variables

| Variable            | Required           | Description |
| ------------------- | ------------------ | ----------- |
| `AMESA_LICENSE`     | Yes                | License key. In remote controller and perceptor Docker containers, `COMPOSABL_LICENSE` is accepted as a fallback if `AMESA_LICENSE` is not set. |
| `AMESA_EULA_AGREED` | Yes (set to `"1"`) | Accept EULA |

Set these before calling `trainer.train()`:

```python
import os
os.environ.setdefault("AMESA_LICENSE", "YOUR_LICENSE_KEY")
os.environ.setdefault("AMESA_EULA_AGREED", "1")
```

## Full training script pattern

```python
import os
from amesa_train.trainer import Trainer
from my_agent.build_agent import build_agent

os.environ.setdefault("AMESA_LICENSE", "YOUR_LICENSE_KEY")
os.environ.setdefault("AMESA_EULA_AGREED", "1")

config = {
    "target": {
        "v2": {
            "redis_url": "redis://localhost:6379",
            "sim_image": "my-sim:latest",
            "initial_replicas": 2,
            "num_episode_managers": 1,
            "enable_ppo_training": True,
            "ppo_training_samples": 4000,
            "sim_node_local": True,
            "perceptor_node_local": True,
            "skill_node_local": True,
            "episode_manager_local": True,
        }
    }
}

agent = build_agent()
trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=100)
finally:
    trainer.close()
```

## Local target (non-v2, for reference)

```python
config = {
    "target": {
        "local": {
            "address": "localhost:1337"  # address of a running sim server
        }
    }
}
```

---

## ⚠️ Quirks

**Redis must be running** — v2 requires a Redis instance. For local dev, start one with Docker:

```bash
docker run -d -p 6379:6379 redis:7-alpine redis-server --save "" --appendonly no
```

**`train_cycles` vs `training_cycles`** — `train_cycles` is the argument to `trainer.train(agent, train_cycles=N)` — it controls the outer loop count. `training_cycles` is a **required** kwarg on `Skill(...)` that sets PPO iterations per skill per cycle. In v2, the runtime reads `training_cycles` exclusively from the skill config and raises a `ValueError` at training start if it is missing — the `train_cycles` argument to `trainer.train()` is ignored for this purpose. Always set it:

```python
skill = Skill("my-skill", MyTeacher, training_cycles=100)
```

**`trainer.close()` in a finally block** — Always call `trainer.close()` to tear down Redis consumers and any spawned processes. Omitting it can leave stale consumer groups in Redis.
