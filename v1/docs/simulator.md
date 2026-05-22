# Simulator

The simulator provides the training environment. It must implement `ServerAmesa` and be exposed via the sim server bootstrap.

## Import

```python
from amesa_core.networking.sim.server_amesa import ServerAmesa
from amesa_core.networking.sim import server as server_make
```

## Implement `ServerAmesa`

All methods are `async`.

### Required methods

```python
async def make(self, env_id: str, env_init: dict) -> EnvSpec
async def sensor_space_info(self) -> Space
async def action_space_info(self) -> Space
async def action_space_sample(self) -> any
async def reset(self) -> tuple[any, dict]
async def step(self, action) -> tuple[any, float, bool, bool, dict]
async def close(self) -> None
async def set_scenario(self, scenario) -> None
async def get_scenario(self) -> any
async def get_render(self, render_mode) -> any
```

### Method semantics

| Method                | Notes                                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `make`                | Called once at startup. Apply `env_init` configuration here. Return an `EnvSpec`-compatible dict with `{"id": ..., "max_episode_steps": ...}` or a real `EnvSpec`. |
| `sensor_space_info`   | Return the gymnasium (or amesa) `Space` describing observations.                                                                                                   |
| `action_space_info`   | Return the gymnasium `Space` describing actions. **This is the authoritative action space source for the trainer.**                                                |
| `action_space_sample` | Return one random action sample from the action space.                                                                                                             |
| `reset`               | Reset the environment. Return `(obs_dict, info_dict)`. Gymnasium 0.26+ signature.                                                                                  |
| `step`                | Apply `action`. Return `(obs, reward, terminated, truncated, info)`.                                                                                               |
| `set_scenario`        | Apply the scenario to the environment state. Called before each episode. Receives a `Scenario` object — call `.sample()` to get a plain `dict` of scalar values.   |
| `get_scenario`        | Return the current scenario dict (used for logging/debugging).                                                                                                     |
| `get_render`          | Return a rendered frame or `None`. Takes a `render_mode` argument.                                                                                                 |

## Full example

```python
import gymnasium as gym
import numpy as np
from amesa_core.networking.sim.server_amesa import ServerAmesa

class MySimImpl(ServerAmesa):

    def __init__(self, *args, **kwargs):
        self.env_init = kwargs.get("env_init", {})
        self.env = MyGymEnv(self.env_init)

    async def make(self, env_id: str, env_init: dict):
        self.env_init = env_init or self.env_init
        return {"id": "my-sim", "max_episode_steps": 200}

    async def sensor_space_info(self):
        return self.env.observation_space

    async def action_space_info(self):
        return self.env.action_space

    async def action_space_sample(self):
        return self.env.action_space.sample()

    async def reset(self):
        return self.env.reset()

    async def step(self, action):
        return self.env.step(action)

    async def close(self):
        self.env.close()

    async def set_scenario(self, scenario):
        self.env.set_scenario(scenario.sample() if hasattr(scenario, "sample") else scenario)

    async def get_scenario(self):
        return self.env.get_scenario()

    async def get_render(self, render_mode):
        return None
```

## Gym environment pattern

Wrap a standard `gymnasium.Env` inside `ServerAmesa` (delegation pattern):

```python
class MySimImpl(ServerAmesa):
    def __init__(self, *args, **kwargs):
        self.env = MyGymEnv(kwargs.get("env_init", {}))

    async def reset(self):
        return self.env.reset()      # returns (obs, info)

    async def step(self, action):
        return self.env.step(action) # returns (obs, reward, terminated, truncated, info)
```

## Server bootstrap (server.py)

```python
import argparse
import asyncio
import json
import os
from amesa_core.networking.sim import server as server_make
from my_sim.server_impl import MySimImpl

async def start(host: str, port: int, protocol: str, env_init: dict) -> None:
    sim_server = server_make.make(
        server_impl=MySimImpl,
        host=host,
        port=port,
        protocol=protocol,
        env_init=env_init,
    )
    await sim_server.start()
    while True:
        await asyncio.sleep(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Start the sim server")
    parser.add_argument("--host", default=os.environ.get("HOST") or "0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 1337))
    parser.add_argument("--protocol", default=os.environ.get("PROTOCOL") or "grpc")
    parser.add_argument("--env_init", type=str, default=os.environ.get("ENV_INIT") or "{}")
    args = parser.parse_args()
    asyncio.run(start(args.host, args.port, args.protocol, json.loads(args.env_init)))

if __name__ == "__main__":
    main()
```

### `server_make.make()` signature

```python
server_make.make(
    server_impl: ServerAmesa,  # your implementation class (not an instance)
    host: str,                 # bind address
    port: int,                 # bind port
    config: dict = {},         # passed as env_init kwarg to your __init__
    protocol: str = "grpc",    # "grpc" (default) or "http"
)
```

Returns a `ServerGRPC` or `ServerHTTP` instance; call `.start()` to begin serving.

## Packaging as a Docker image

When using the `docker` or `kubernetes` trainer targets, the trainer spawns the sim as a Docker container. The container must expose a gRPC server on port `1337`.

### Dockerfile pattern

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=1337
ENV PROTOCOL=grpc

EXPOSE 1337
CMD ["python", "server.py"]
```

### Docker environment variables

| Variable   | Default   | Description                                                               |
| ---------- | --------- | ------------------------------------------------------------------------- |
| `HOST`     | `0.0.0.0` | Bind address inside the container (should always be `0.0.0.0` for Docker) |
| `PORT`     | `1337`    | gRPC port inside the container                                            |
| `PROTOCOL` | `grpc`    | Transport protocol (`grpc` or `http`)                                     |
| `ENV_INIT` | `{}`      | JSON string merged into `env_init` at startup (passed to `make()`)        |
| `AMESA_ENV` | host value | AMESA backend environment (`STAGING`, `TEST`, `PROD`). Automatically forwarded from the host process — do not set this on the container directly. |

The trainer config references the image by name:

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

### Building and running locally

```bash
# Build
docker build -t my-sim:latest ./sim

# Run standalone (for testing)
docker run -p 1337:1337 my-sim:latest

# Run with env_init override
docker run -p 1337:1337 -e ENV_INIT='{"max_steps": 500}' my-sim:latest
```

### requirements.txt minimum

```
amesa-core-dev==<version>
gymnasium==0.28.1
numpy==1.26.4
```

---

## ⚠️ Quirks

**`get_render` takes `render_mode`** — Unlike `gymnasium`'s parameterless `render()`, this method receives a `render_mode` argument. Return `None` if rendering is unsupported.

**`reset` must return a tuple** — `(obs, info_dict)` matching gymnasium 0.26+ API. Older gymnasium style (returning obs only) will cause protocol errors.

**`set_scenario` receives a `Scenario` object** — Call `.sample()` on it to get a plain `dict` of resolved scalar values. Use `hasattr(scenario, "sample")` to guard against plain dicts if you need the sim to work in both contexts.

**`env_init` vs `set_scenario`** — `env_init` is passed once at server startup via `make()`. `set_scenario` is called before each episode. Use `env_init` for static configuration (e.g., max steps), scenarios for per-episode variation.

**Docker host is always `0.0.0.0`** — Inside a Docker container the server must bind to `0.0.0.0`, not `127.0.0.1`. The trainer connects to the published port on the host machine.
