# Trainer

`Trainer` orchestrates the training loop. In v1 it uses Ray/RLlib internally — no Redis or external message broker is required.

## Import

```python
from amesa_train import Trainer
```

## Basic usage

```python
trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=50)
finally:
    trainer.close()
```

## v1 Config

Pass a dict to `Trainer(config)`. The top-level `"target"` key selects the execution target. v1 supports four targets: `local`, `docker`, `kubernetes`, and `amesa`.

---

### `local` target

The sim is already running on a known address. No containers are spawned. Best for development — start your sim manually in one terminal, run training in another.

```python
config = {
    "target": {
        "local": {
            "address": "localhost:1337",
            "protocol": "grpc",  # optional — defaults to "grpc"
        }
    }
}
```

| Field | Default | Description |
|---|---|---|
| `address` | required | `host:port` of the running sim gRPC server |
| `protocol` | `"grpc"` | Transport protocol (`"grpc"` or `"http"`) |

---

### `docker` target

The trainer spawns the sim as a Docker container before training begins, then tears it down when `trainer.close()` is called.

```python
config = {
    "target": {
        "docker": {
            "image": "my-sim:latest",
            "protocol": "grpc",
        }
    }
}
```

| Field | Default | Description |
|---|---|---|
| `image` | required | Docker image tag for the sim container |
| `protocol` | `"grpc"` | Transport protocol |

The container must expose a gRPC server on port `1337`. See [simulator.md](simulator.md#packaging-as-a-docker-image) for the Dockerfile pattern.

---

### `kubernetes` target

The trainer submits sim pods to a Kubernetes cluster. Useful for scaling out training with many parallel sim replicas.

```python
config = {
    "target": {
        "kubernetes": {
            "image": "my-sim:latest",
            "sim_cpu": "500m",
            "sim_memory": "256Mi",
            "namespace": "amesa-train",
            "namespace_sims": "amesa-sims",
        }
    }
}
```

| Field | Default | Description |
|---|---|---|
| `image` | required | Docker image for sim pods |
| `sim_cpu` | `"500m"` | CPU request per sim pod |
| `sim_memory` | `"256Mi"` | Memory request per sim pod |
| `namespace` | `"amesa-train"` | Namespace for the training job |
| `namespace_sims` | `"amesa-sims"` | Namespace for sim pods |

---

### `amesa` target

Runs training on the AMESA managed cloud. The AMESA platform handles cluster provisioning, scaling, and teardown.

```python
config = {
    "target": {
        "amesa": {
            "image": "my-sim:latest",
            "plan": "small",  # small / medium / large
        }
    }
}
```

| Field | Default | Description |
|---|---|---|
| `image` | required | Docker image for the sim |
| `plan` | `"small"` | Compute plan: `"small"`, `"medium"`, or `"large"` |

---

## Skill training hyperparameters

In v1, per-skill training hyperparameters are set as kwargs on `Skill(...)`, not in the trainer config:

```python
from amesa_core.agent.skill.skill import Skill

skill = Skill(
    "my-skill",
    MyTeacher,
    training_cycles=100,      # PPO update iterations per outer training cycle
    train_batch_size=4000,    # samples collected per PPO batch
    workers=2,                # Ray rollout workers for this skill
    envs_per_worker=1,        # environments per worker
)
```

### Hyperparameter reference

| kwarg | Type | Default | Description |
|---|---|---|---|
| `training_cycles` | int | trainer default | PPO update iterations to run for this skill each outer cycle |
| `train_batch_size` | int | `4000` | Total samples collected before each PPO update |
| `workers` | int | `1` | Number of Ray rollout workers |
| `envs_per_worker` | int | `1` | Gymnasium environments per rollout worker |
| `fc_layers` | list[int] | `[256, 256]` | Fully-connected layer sizes for the policy network |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `AMESA_LICENSE` | Yes | License key issued by AMESA |
| `AMESA_EULA_AGREED` | Yes (set to `"1"`) | Accept the AMESA EULA |

Set these before calling `trainer.train()`:

```python
import os
os.environ.setdefault("AMESA_LICENSE", "YOUR_LICENSE_KEY")
os.environ.setdefault("AMESA_EULA_AGREED", "1")
```

---

## Full training script pattern

```python
from __future__ import annotations

import os

from amesa_train import Trainer
from my_agent.build_agent import build_agent

os.environ.setdefault("AMESA_LICENSE", "YOUR_LICENSE_KEY")
os.environ.setdefault("AMESA_EULA_AGREED", "1")

config = {
    "target": {
        "local": {
            "address": "localhost:1337",
        }
    }
}

agent = build_agent()
trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=10)
    print("Training complete!")
finally:
    trainer.close()
```

## Ray initialization

The v1 trainer initializes Ray internally when `trainer.train()` is called. You do not need to call `ray.init()` yourself. The trainer also shuts Ray down when `trainer.close()` is called.

If Ray is already initialized in your process (e.g., you started it with `ray.init()` for other purposes), the trainer will detect the running instance and join it rather than starting a new one.

---

## ⚠️ Quirks

**`training_cycles` (on Skill) vs `train_cycles` (on Trainer.train)** — `training_cycles` is the kwarg on `Skill(...)` that controls the number of PPO update iterations per skill per outer loop. `train_cycles` is the argument to `trainer.train(agent, train_cycles=N)` and controls how many outer loops run. Both affect total training time; they are independent knobs.

**Must call `trainer.close()`** — Always wrap `trainer.train()` in a try/finally and call `trainer.close()` in the finally block. This tears down Ray workers and any spawned sim processes. Omitting it can leave orphaned Ray actors or Docker containers running.

```python
trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=10)
finally:
    trainer.close()
```

**`local` target requires a running sim** — The trainer does not start the sim for you when using the `local` target. Start your sim server separately before calling `trainer.train()`. See [simulator.md](simulator.md) for the server bootstrap pattern.

**`docker` target requires Docker** — The Docker daemon must be running and the image must be available locally (or pullable). The trainer pulls the image if needed before spawning the container.

**Ray version compatibility** — v1 requires a specific Ray/RLlib version pinned in `amesa-train-dev`. Do not upgrade Ray independently; always upgrade through the `amesa-train-dev` package.
