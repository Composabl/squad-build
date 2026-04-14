## Zero-to-Run

This appendix is the missing glue for a **clean-room repo**: package layout, server entrypoint, and
the commands needed to run a local gRPC sim or a V2 `sim_image`. It is intended to be sufficient
context for a coding agent to build **sim + server + server_impl + full agent stack** from scratch.

### Minimal repo layout

```
my_sim_project/
├── requirements.txt             # quick local runs (pip)
├── pyproject.toml               # required if you load the sim by path/plugin
├── sim/
│   ├── __init__.py
│   ├── sim.py                   # Gymnasium Env (pure simulation logic)
│   ├── server_impl.py           # ServerAmesa adapter (SimImpl)
│   └── server.py                # gRPC/HTTP runner
├── agent/
│   ├── perceptor.py             # optional
│   ├── teacher.py               # learned skill
│   ├── controller.py            # programmed skill (optional)
│   ├── build_agent.py
│   └── run_training.py
└── README.md
```

### Dependencies (local runs)

Use the full SDK if you want training + core in a single install:

```bash
pip install amesa
```

Minimal sim-only installs typically need:

```bash
pip install amesa-core gymnasium numpy
```

> **License**: set `AMESA_LICENSE` and `AMESA_EULA_AGREED=1` in your environment before training.

### `pyproject.toml` for sim plugins (required for path loading)

When using `server_entrypoint` or any path-based sim loading, you must provide a `[amesa]` section
with an explicit entrypoint. This mirrors the CLI sim template:

```toml
[project]
name = "my-sim"
version = "0.1.0"
description = "My custom sim"
dependencies = [
  "amesa-core"
]

[amesa]
type = "sim"
entrypoint = "sim.server_impl:SimImpl"
dependencies_system = []
```

### `server_impl.py` (Sim → ServerAmesa adapter)

Use a Gymnasium `Env` for the simulation logic, then wrap it in a `ServerAmesa` implementation
that forwards `make`, `reset`, `step`, etc.

```python
# sim/server_impl.py
import gymnasium as gym
from typing import Any, Dict, SupportsFloat, Tuple
from amesa_core.networking.sim.server_amesa import ServerAmesa
from .sim import Env

class SimImpl(ServerAmesa):
    def __init__(self, *args, **kwargs):
        self.env_init = kwargs.get("env_init", {})
        self.env = Env(self.env_init)

    async def make(self, env_id: str, env_init: dict):
        self.env_init = env_init if env_init else self.env_init
        return {"id": "my_sim", "max_episode_steps": 1000}

    async def sensor_space_info(self) -> gym.Space:
        return self.env.sensor_space

    async def action_space_info(self) -> gym.Space:
        return self.env.action_space

    async def action_space_sample(self) -> Any:
        return self.env.action_space.sample()

    async def reset(self) -> Tuple[Any, Dict[str, Any]]:
        return self.env.reset()

    async def step(
        self, action
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        return self.env.step(action)

    async def close(self):
        self.env.close()
```

### `server.py` (gRPC/HTTP runner)

This is the actual server process that your trainer or V2 runner connects to:

```python
# sim/server.py
import asyncio
import os
from argparse import ArgumentParser
from amesa_core.networking.sim import server as server_make
from sim.server_impl import SimImpl

async def start(host, port, protocol, env_init):
    server = server_make.make(
        server_impl=SimImpl,
        host=host,
        port=port,
        protocol=protocol,
        env_init=env_init,
    )
    await server.start()
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    parser = ArgumentParser(description="Start the sim server")
    parser.add_argument("--host", default=os.environ.get("HOST") or "[::]")
    parser.add_argument("--port", type=int, default=os.environ.get("PORT") or 1337)
    parser.add_argument("--protocol", default="grpc")
    parser.add_argument("--env_init", type=str, default="{}")
    args = parser.parse_args()

    # env_init is parsed from string to dict
    args.env_init = eval(args.env_init)
    asyncio.run(start(args.host, args.port, args.protocol, args.env_init))
```

### Running a local gRPC sim (for `target.local`)

**Terminal A — start the sim server:**

```bash
python sim/server.py --host 0.0.0.0 --port 1337 --protocol grpc --env_init "{}"
```

**Terminal B — run training with a local target:**

```python
# agent/run_training.py (excerpt)
config = {
  "target": {
    "local": {"address": "localhost:1337"}  # gRPC sim endpoint
  }
}
trainer = Trainer(config)
trainer.train(agent, train_cycles=10)
```

### V2 `sim_image` (Dockerized sim server)

If you want V2 to **spawn the sim container per replica**, build a container that starts the sim
server on port `1337`.

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY sim ./sim
CMD ["python", "sim/server.py", "--host", "0.0.0.0", "--port", "1337", "--protocol", "grpc", "--env_init", "{}"]
```

Build + use in V2 config:

```bash
docker build -t my-sim:latest .
```

```python
config = {
  "target": {
    "v2": {
      "redis_url": "redis://localhost:6379",
      "sim_image": "my-sim:latest",
      "initial_replicas": 4,
      "num_episode_managers": 2,
    }
  }
}
```

### Full agent stack checklist (what a coding agent should create)

- **Sim**: Gymnasium `Env` + `ServerAmesa` adapter (`server_impl.py`)
- **Server**: `server.py` runner (gRPC/HTTP)
- **Perceptors**: `PerceptorImpl` classes + registration in `Agent`
- **Skills**:
  - **Teacher** (DRL): `SkillTeacher` with reward/termination/criteria
  - **Controller** (programmed): `SkillController` with `compute_action`
- **Scenarios**: dict-based scenarios added via `skill.add_scenario(...)`
- **Selectors/Groups** (optional):
  - **SkillSelector** to route between skills
  - **SkillGroup** for plan/execute pairings
- **Trainer config**: local or V2 target, plus `ppo_training_samples`, etc.

These pieces, plus the examples in `end_to_end_example.md`, are the **minimum complete context** needed to
build a sim + server + full agent stack in a standalone repo.
