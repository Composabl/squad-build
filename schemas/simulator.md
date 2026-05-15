# Simulator Schema — `ServerAmesa`

The simulator is the environment the agent trains in. AMESA extends the
`gymnasium.Env` standard. You implement a `ServerAmesa` subclass that bridges your
sim logic to the AMESA SDK's gRPC layer.

## Import

```python
from amesa_core.networking.sim.server_amesa import ServerAmesa
```

## Interface Contract

All methods are `async`. Implement all of them.

```python
class ServerAmesa:

    # ── Setup ──────────────────────────────────────────────────────────

    async def make(self, env_id: str, env_init: dict) -> dict:
        """
        Create/configure the environment instance.
        env_id:   string identifier for the environment type
        env_init: dict of initial config params (from trainer config)
        Returns:  dict with at least {"id": "<sim_name>"}
        """

    async def sensor_space_info(self) -> gym.Space:
        """
        Return the gymnasium Space describing all sensor (observation) variables.
        Called once at startup; result defines the agent's observation space.
        """

    async def action_space_info(self) -> gym.Space:
        """
        Return the gymnasium Space describing valid actions.
        """

    async def action_space_sample(self):
        """
        Return one sample action from the action space.
        Used by the SDK for space validation and random exploration.
        """

    # ── Episode control ────────────────────────────────────────────────

    async def reset(self) -> tuple[dict, dict]:
        """
        Reset to initial state; return (observation, info).
        observation: dict keyed by sensor names
        info:        auxiliary dict (may be empty)
        """

    async def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """
        Advance one timestep.
        action:      action produced by the agent (matches action_space_info())
        Returns:     (observation, reward, terminated, truncated, info)
          observation: dict keyed by sensor names
          reward:      float from sim (teacher may override this)
          terminated:  True = natural episode end
          truncated:   True = episode cut short (time limit, etc.)
          info:        auxiliary dict
        """

    async def close(self):
        """Cleanup. Called when the SDK is done with this environment instance."""

    # ── Scenario control ───────────────────────────────────────────────

    async def set_scenario(self, scenario: dict):
        """
        Receive the current scenario dict from the agent.
        Store it and use it to configure your sim's initial conditions on next reset().
        scenario keys match the variable names defined in your Scenario objects.
        """

    async def get_scenario(self) -> dict | None:
        """Return the currently active scenario dict, or None if none set."""

    # ── Rendering (optional) ───────────────────────────────────────────

    async def get_render(self) -> np.ndarray | str:
        """Return the current rendered frame (numpy array or base64 string)."""
```

## Minimal Implementation Pattern

```python
from amesa_core.networking.sim.server_amesa import ServerAmesa
from my_sim import MyGymEnv   # your gymnasium.Env subclass

class SimImpl(ServerAmesa):
    def __init__(self, *args, **kwargs):
        self.env_init = kwargs.get("env_init", {})
        self.env = MyGymEnv(self.env_init)

    async def make(self, env_id: str, env_init: dict):
        self.env = MyGymEnv(env_init or self.env_init)
        return {"id": "my-sim"}

    async def sensor_space_info(self):
        return self.env.observation_space   # gym.Space

    async def action_space_info(self):
        return self.env.action_space        # gym.Space

    async def action_space_sample(self):
        return self.env.action_space.sample()

    async def reset(self):
        return self.env.reset()             # (obs, info)

    async def step(self, action):
        return self.env.step(action)        # (obs, reward, terminated, truncated, info)

    async def close(self):
        self.env.close()

    async def set_scenario(self, scenario):
        self.env.scenario = scenario

    async def get_scenario(self):
        return getattr(self.env, "scenario", None)

    async def get_render(self):
        return self.env.render()
```

## `main.py` — Exposing the Sim via gRPC

```python
import asyncio
from amesa_core.networking.sim.server import SimServer
from my_sim.server_impl import SimImpl

async def main():
    server = SimServer(SimImpl)
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
```

## File Structure for a Packaged Sim

```
your-simulator/
├── Dockerfile
├── requirements.txt
├── docker/
│   └── entrypoint.sh
└── src/
    ├── __init__.py
    ├── main.py          # gRPC server entry point
    ├── server_impl.py   # ServerAmesa subclass
    └── sim.py           # gymnasium.Env subclass
```

## Observation Dict Convention

Sensor names in the observation dict returned by `reset()` and `step()` must be
stable string keys. These keys are the names you pass to `Sensor("name", ...)` in
the agent builder. They must match exactly — the SDK maps them via your Sensor
lambdas.

## Publishing

```bash
# From inside your sim folder:
composabl sim publish

# Or build and push a Docker image:
docker build -t <dockerhub-user>/<sim-name> .
docker push <dockerhub-user>/<sim-name>
# Then register via the AMESA UI (Simulators → New Simulator → External)
```

## gRPC Methods Exposed

The AMESA SDK auto-generates gRPC bindings. Your `ServerAmesa` methods map 1:1 to
these RPC endpoints:

| gRPC method | ServerAmesa method |
|---|---|
| `make` | `make()` |
| `step` | `step()` |
| `reset` | `reset()` |
| `close` | `close()` |
| `action_space_sample` | `action_space_sample()` |
| `action_space_info` | `action_space_info()` |
| `observation_space_info` | `sensor_space_info()` |
| `set_scenario` | `set_scenario()` |
| `get_scenario` | `get_scenario()` |
| `get_render` | `get_render()` |

## Behavioral Notes

- `make()` may be called multiple times (e.g., for different `env_id` values). Handle
  re-initialization gracefully.
- `set_scenario()` is called before `reset()` at each episode start. Your sim should
  read the scenario dict inside `reset()` to set initial conditions accordingly.
- `step()` reward is passed as `sim_reward` to `teacher.compute_reward()`. The
  teacher's return value is what the RL algorithm sees — not the raw sim reward.
- `terminated` vs `truncated`: AMESA follows gymnasium convention. Use `terminated`
  for natural episode endings (goal reached, failure state). Use `truncated` for
  time limits.
