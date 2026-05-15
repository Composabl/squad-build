# Perceptor Schema — `PerceptorImpl`

Perceptors are the feature engineering / perception layer. They run after sensor
mapping but before any skill teacher or controller. Each perceptor receives the
current full sensor dict and returns new derived keys to add to it.

## Import

```python
from amesa_core import PerceptorImpl, Perceptor
# or:
from composabl import PerceptorImpl, Perceptor
```

## Interface Contract

```python
class PerceptorImpl:

    def __init__(self, *args, **kwargs):
        """Initialize per-episode state here (filters, accumulators, etc.)."""

    async def compute(self, obs_spec, obs: dict) -> dict:
        """
        Compute derived features from the current observation.

        obs_spec:  gymnasium Space spec — may be None in some call paths.
                   Do not rely on it being a valid Space.
        obs:       Full AMESA sensor dict including all registered sensors
                   plus outputs of any perceptors that ran before this one.

        Returns:   dict of new {key: value} pairs to add to obs.
                   Keys MUST NOT already exist in obs — collision raises Exception.
                   Return only new keys.
        """

    def filtered_sensor_space(self, obs: dict) -> list[str]:
        """
        Declare which sensor keys this perceptor reads.
        DOCUMENTATION ONLY — not enforced by the training pipeline.
        """
```

## Wrapper Registration

Wrap your `PerceptorImpl` in a `Perceptor` object and attach it to the agent:

```python
perceptor = Perceptor(
    perceptor_name="my-perceptor",    # unique name
    impl_cls=MyPerceptorImpl,         # class (not instance) OR instance
    description="What this computes", # optional
)
agent.add_perceptor(perceptor)
```

## Execution Order

Perceptors run in registration order. Each receives the full enriched dict from all
prior perceptors. Register dependencies before the perceptors that depend on them.

```python
agent.add_perceptor(Perceptor("velocity",       VelocityPerceptor))      # runs first
agent.add_perceptor(Perceptor("kinetic-energy", KineticEnergyPerceptor)) # reads "velocity"
```

## Episode Reset Behavior

At the start of each episode, the SDK re-instantiates the perceptor via `impl_cls()`.
All `self.*` state is wiped. Expensive initialization (loading ML models) in
`__init__` runs on every episode reset — cache at the class or module level if needed.

## Deployment Modes

### In-Process (default)

```python
agent.add_perceptor(Perceptor("ema", ExponentialMovingAverage(alpha=0.1)))
```

### Portable (URL tarball)

```python
from amesa_core import PerceptorOptions

perceptor = Perceptor(
    perceptor_name="derivative",
    impl_cls="",
    config=PerceptorOptions(remote_address="http://host/derivative-0.0.1.tar.gz")
)
await agent.init()  # Downloads and extracts
```

### Remote (gRPC Docker container)

```python
perceptor = Perceptor(
    perceptor_name="cv-model",
    impl_cls="",
    config=PerceptorOptions(remote_address="http://registry.example.com/cv-perceptor:latest")
)
```

When `remote_address` starts with `http`, the SDK pulls the Docker image and
communicates via gRPC. A filesystem path loads the module locally.

## Common Patterns

### Simple Derivative (velocity from position)

```python
import time

class DerivativePerceptor(PerceptorImpl):
    def __init__(self):
        self.prev_value = None
        self.prev_time = None

    async def compute(self, obs_spec, obs):
        current = obs["position"]
        now = time.time()
        derivative = 0.0
        if self.prev_value is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0:
                derivative = (current - self.prev_value) / dt
        self.prev_value = current
        self.prev_time = now
        return {"velocity": derivative}

    def filtered_sensor_space(self, obs):
        return ["position"]
```

### Exponential Moving Average (noise filter)

```python
class EMAPerceptor(PerceptorImpl):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.filtered = None

    async def compute(self, obs_spec, obs):
        raw = obs["noisy_sensor"]
        self.filtered = raw if self.filtered is None else (
            self.alpha * raw + (1 - self.alpha) * self.filtered
        )
        return {"filtered_sensor": self.filtered}
```

### ML Model as Perceptor

```python
import pickle

class MLPerceptor(PerceptorImpl):
    def __init__(self):
        self.model = pickle.load(open("model.pkl", "rb"))  # expensive — runs each episode

    async def compute(self, obs_spec, obs):
        X = [[obs["Ca"], obs["T"], obs["Tc"]]]
        prediction = self.model.predict(X)[0]
        return {"ml_prediction": float(prediction)}

    def filtered_sensor_space(self, obs):
        return ["Ca", "T", "Tc"]
```

### LLM Analyst (display-only, no agent input)

```python
class AnalystPerceptor(PerceptorImpl):
    async def compute(self, obs_spec, obs):
        response = llm_client.ask(f"Current state: {obs}. Provide analysis.")
        console_client.post(response)
        return {"llm_analysis": 0}  # must return something; 0 = no agent input
```

### LLM Executive (returns recommendation to agent)

```python
class AdvisorPerceptor(PerceptorImpl):
    async def compute(self, obs_spec, obs):
        response = llm_client.ask(f"State: {obs}. Recommended action?")
        action_value = parse_action(response)  # convert text to numeric
        return {"llm_recommendation": action_value}
```

## Critical Rules

| Rule | Detail |
|---|---|
| No key collisions | Returning a key that exists in obs raises Exception. |
| New keys only | Return only new derived keys, never modify existing sensor values. |
| `obs_spec` may be None | Handle `obs_spec is None` gracefully in `compute()`. |
| State resets every episode | All `self.*` is wiped at episode boundaries. |
| `filtered_sensor_space` unenforced | Reads outside the declared list are allowed but undocumented. |
| Outputs not auto-normalized | Perceptor outputs reach skills at raw scale. Normalize in `compute()` or teacher's `transform_sensors()`. |
| Shared across Ray workers | In distributed training, the `Perceptor` wrapper returns `self` from `__deepcopy__`. Stateful perceptors may corrupt state across workers. |

## `await` Compatibility Note

The AMESA runtime may `await` a perceptor instance directly during reset. To prevent
`can't be used in 'await' expression` errors, add a no-op `__await__` to your impl:

```python
class MyPerceptor(PerceptorImpl):
    def __await__(self):
        yield self
        return self
```
