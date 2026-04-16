### Setting Up the Environment Instance

#### `make(env_id, env_init)`

Creates a new instance of the environment.

- `env_id` (string) — identifier for the environment type.
- `env_init` (dict) — initial configuration parameters.

```python
async def make(self, env_id: str, env_init: dict):
    self.env_id = env_id if env_id else self.env_id
    self.env_init = env_init if env_init else self.env_init
    self.env = Sim(self.env_init)
    return {
        "id": "my-simulator",
        "max_episode_steps": int(self.env.max_steps) if self.env.max_steps is not None else 0,
        "order_enforce": False,
    }
```

#### `sensor_space_info()`

Returns details about the environment's sensor (observation) space as a `gym.Space`.

```python
async def sensor_space_info(self) -> gym.Space:
    return self.env.sensor_space
```

#### `action_space_info()`

Returns the agent system's action space as a `gym.Space`.

```python
async def action_space_info(self) -> gym.Space:
    return self.env.action_space
```

#### `action_space_sample()`

Returns a random sample from the action space. Useful for exploration.

```python
async def action_space_sample(self):
    return self.env.action_space.sample()
```

### Running the Environment

#### `reset()`

Resets the environment and returns the first observation plus an info dict.

```python
async def reset(self):
    return self.env.reset()
```

#### `step(action)`

Applies an action and returns a tuple:

| Return Value  | Type  | Description                         |
| ------------- | ----- | ----------------------------------- |
| `observation` | —     | The observation after the action    |
| `reward`      | float | Reward received                     |
| `terminated`  | bool  | Whether the episode ended naturally |
| `truncated`   | bool  | Whether the episode was cut short   |
| `info`        | dict  | Additional metadata                 |

```python
async def step(self, action):
    return self.env.step(action)
```

#### `close()`

Signals the simulator is done; perform cleanup.

```python
async def close(self):
    self.env.close()
```

#### `set_scenario(scenario)` / `get_scenario()`

Sets or retrieves the current scenario the agent system is training on.

```python
from amesa_core.agent.scenario import Scenario

async def set_scenario(self, scenario):
    self.env.set_scenario(scenario)

async def get_scenario(self):
    if self.env.scenario is None:
        return Scenario({"dummy": 0})
    return self.env.scenario
```

#### Implementation notes

- **`env_init` is your sim contract.** The SDK forwards `env_init` from training → sim `make(...)`. Use it for model paths, reward config, initial state, max steps, or any custom knobs your sim needs.
- **V2 event-based sims require a transport target.** `env_init` must include either `sim_address` (connect to an existing gRPC sim) or `sim_image` (spawn a container per sim).
- **Named sensors need dict observations.** If your agent expects named sensors, define the observation space as a `gym.spaces.Dict` of scalar `Box` entries and return dict observations from `reset()`/`step()` so downstream sensor extraction stays consistent.
- **Sim methods stay synchronous.** Keep your `gym.Env` `reset()`/`step()` synchronous; the ServerAmesa wrapper provides async boundaries.
- **Scenario handling:** `set_scenario()` receives a `Scenario` object created from a dict. Use `scenario.sample()` to materialize concrete values for reset-time initialization.
- **Scenario serialization:** For V2 stacks that expect a `Scenario` object, return a minimal `Scenario` when unset instead of `None`.
- **Reward/termination responsibility:** the sim can supply reward/termination, but a skill’s teacher can override. If your sim reward is meaningful, pass it through in the teacher.

### Visualization

#### `get_render()`

Returns the current rendered image of the environment (numpy array or string).

```python
async def get_render(self):
    return self.env.render()
```
