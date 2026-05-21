# Perceptors

A perceptor transforms raw sim observations before they reach skills. Use them for computed features, derived signals, or any preprocessing that multiple skills share.

A perceptor is two objects:

- `PerceptorImpl` — your implementation (the logic)
- `Perceptor` — a wrapper that registers the impl with the agent

## Implement `PerceptorImpl`

```python
from amesa_core.agent.perceptor.perceptor_impl import PerceptorImpl

class MyPerceptor(PerceptorImpl):

    async def compute(self, obs_spec, obs) -> dict:
        """
        Transform the observation.

        Args:
            obs_spec: The observation space specification (rarely needed).
            obs: The raw observation dict from the sim.

        Returns:
            A dict of new/derived keys that get merged into the observation
            seen by skills.
        """
        return {
            "derived_value": obs["a"] - obs["b"],
        }

    def filtered_sensor_space(self, obs) -> list[str]:
        """
        Return the list of raw sensor names this perceptor needs as input.
        """
        return ["a", "b"]
```

## Register with the Agent

```python
from amesa_core.agent.perceptor.perceptor import Perceptor

MY_PERCEPTOR = Perceptor(
    "my-perceptor",        # unique name
    MyPerceptor,           # the class (not an instance)
    "Computes derived_value from a and b",  # optional description
)

agent.add_perceptor(MY_PERCEPTOR)
```

Multiple perceptors can be added; their outputs are merged into the observation dict.

## Observing perceptor outputs in skills

Keys returned by `compute()` become available in `transformed_sensors` inside the skill. Include them in `filtered_sensor_space()` of the skill to receive them.

```python
# In your SkillTeacher
async def filtered_sensor_space(self):
    return ["a", "b", "derived_value"]  # derived_value comes from the perceptor
```

---

## v2 training: you must opt-in to perceptors

In the v2 training stack, **perceptors are disabled by default**. If your agent has perceptors registered via `agent.add_perceptor(...)`, you must explicitly enable them in the trainer config or the perceptor node will not start and perceptor-derived keys will be absent from every observation.

```python
config = {
    "target": {
        "v2": {
            # ... other fields ...
            "enable_perceptors": True,   # ← required when agent has perceptors
            "perceptor_node_local": True, # run perceptor node in-process (dev default)
        }
    }
}
```

Without `"enable_perceptors": True`:

- `agent.perceptors` is ignored at runtime
- The skill node receives only the raw sim sensor keys
- Any teacher method that uses `.get("perceptor_key", default)` will silently use the default value, which can trigger incorrect rewards, false success, or immediate episode termination

---

## Perceptor state is not reset between episodes

`Perceptor.reset()` returns the existing `PerceptorImpl` instance rather than creating a new one. Stateful impl attributes (e.g. `self.last_value = None`) persist across episode boundaries.

**Workaround** — explicitly clear state in `compute()` at the start of a new episode, or detect a reset by checking for a sentinel value:

```python
async def compute(self, obs_spec, obs):
    if self.last_temp is None or obs.get("step_count", 1) == 0:
        self.last_temp = float(obs["air_temp"])
    ...
```

---

## ⚠️ Quirks

**Class, not instance** — Pass the **class** to `Perceptor(...)`, not an instance. The SDK instantiates it internally.

**`obs` may be a dict or list** — If the sim returns an array observation, convert it manually inside `compute()`:

```python
async def compute(self, obs_spec, obs):
    if not isinstance(obs, dict):
        obs = dict(zip(EXPECTED_KEYS, obs))
    ...
```
