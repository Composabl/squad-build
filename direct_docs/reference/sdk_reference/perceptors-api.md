# Perceptors API

## Overview

A **perceptor** is a transformation stage in the observation pipeline. It takes the mapped sensor dictionary from your simulator, computes derived or enriched features, and adds them to the observation space that skills consume. Perceptors are the "feature engineering" or "perception" layer of your agent.

Use perceptors for:
- Computing derivatives or integrals from sensor timeseries
- Filtering noisy signals
- Extracting features (e.g., object detection from camera frames)
- Normalizing and preprocessing observations
- Implementing state estimators (Kalman filters, moving averages)

## How Perceptors Fit in the Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          SIMULATOR                                  │
│  raw_obs = OrderedDict / gym.Space sample                           │
└────────────────────────┬────────────────────────────────────────────┘
                         │ gRPC / Redis stream
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SENSOR MAPPING LAYER                             │
│  map_sim_sensors_to_amesa_sensors()                                 │
│  Output: amesa_sensors = { "sensor_a": val, "sensor_b": val, ... }  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PERCEPTOR PIPELINE                             │
│  Runs in registration order                                         │
│  Each perceptor receives full amesa_sensors, adds new keys          │
└────────────────────────┬────────────────────────────────────────────┘
                         │ enriched dict
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│               TEACHER / CONTROLLER  (per Skill)                     │
│  Receives enriched observation space                                │
│  Filters, normalizes, and feeds to RL policy                        │
└─────────────────────────────────────────────────────────────────────┘
```

**Key point:** Perceptors run *after* sensor mapping but *before* skill teachers/controllers. All skills in an agent see the same enriched observation space.

## The Perceptor Interface

To create a perceptor, subclass `PerceptorImpl` and implement two methods:

```python
from composabl import PerceptorImpl

class MyPerceptor(PerceptorImpl):
    def __init__(self):
        # Initialize state here (filters, accumulators, models, etc.)
        pass

    async def compute(self, obs_spec, obs):
        """
        Compute derived features from the sensor observation.
        
        Args:
            obs_spec: Gymnasium Space spec (may be None in some paths).
            obs:      The named sensor dict, e.g. {"position": 1.0, "velocity": 0.5, ...}
        
        Returns:
            dict: New {key: value} pairs to add to observations.
                  Keys must not overlap with existing obs keys (raises Exception if they do).
        """
        # Compute features
        result = {}
        result["my_feature"] = obs["sensor_a"] * 2
        return result

    def filtered_sensor_space(self, obs):
        """
        Return list of sensor keys this perceptor needs.
        
        Note: This is for documentation/introspection only.
              It is NOT enforced by the training pipeline.
        """
        return ["sensor_a"]
```

Register the perceptor on your agent:

```python
from composabl import Agent, Perceptor

agent = Agent("my-agent")

perceptor = Perceptor(
    perceptor_name="my-perceptor",
    impl_cls=MyPerceptor,
    description="Computes derived features"
)
agent.add_perceptor(perceptor)
```

## Chaining Perceptors

Multiple perceptors run sequentially in registration order. Each receives the full observation dictionary — including outputs from all prior perceptors.

```python
agent.add_perceptor(Perceptor("velocity", VelocityPerceptor))      # Runs first
agent.add_perceptor(Perceptor("kinetic-energy", EnergyPerceptor))  # Runs second
```

The velocity perceptor computes `velocity` from position. The energy perceptor then reads `velocity` (plus other sensors) to compute kinetic energy.

**Order matters:** Perceptors that depend on outputs from other perceptors must be registered *after* those dependencies.

## Deployment Modes

Perceptors support three deployment modes:

### In-Process Perceptor (Default)

Runs in the same Python process as your agent. Fastest, simplest.

```python
perceptor = Perceptor("my-filter", NoiseFilter())
agent.add_perceptor(perceptor)
```

### Portable Perceptor (URL-based)

Download and load a packaged perceptor from a URL:

```python
from composabl import PerceptorOptions

perceptor = Perceptor(
    perceptor_name="derivative",
    impl_cls="",
    config=PerceptorOptions(
        remote_address="http://0.0.0.0:8000/derivative-0.0.1.tar.gz"
    )
)
agent.add_perceptor(perceptor)
await agent.init()  # Downloads and extracts the tar.gz
```

### Remote Perceptor (gRPC + Docker)

Deploy a perceptor in a separate gRPC container. Set `remote_address` to an HTTP(S) URL:

```python
perceptor = Perceptor(
    perceptor_name="cv-model",
    impl_cls="",
    config=PerceptorOptions(
        remote_address="http://registry.example.com/cv-perceptor:latest"
    )
)
agent.add_perceptor(perceptor)
```

The SDK will pull the Docker image, start it, and communicate with the perceptor via gRPC. State is isolated per container.

## Output Normalization

**Important:** Perceptor outputs are **not automatically normalized**. The SDK passes them through to skills as-is.

Normalization applies only to sensors (via `Sensor.normalize_sample()`), not to perceptor-derived features. If your perceptor outputs values on a different scale than your other sensors, skills will see mixed scales, which can hurt learning.

If normalization is needed, do it in your perceptor's `compute()` method, or let the skill's `transform_sensors()` method handle it:

```python
class MyTeacher(SkillTeacher):
    async def transform_sensors(self, sensors, action):
        # Normalize perceptor output
        sensors["my_feature"] = (sensors["my_feature"] - 0.5) / 2.0
        return sensors
```

## Gotchas

**1. Key collisions raise exceptions.** If your perceptor returns a key that already exists in the observation dictionary (from sensors or prior perceptors), the SDK raises an exception. Return only new keys.

```python
async def compute(self, obs_spec, obs):
    # WRONG: "position" already exists
    return {"position": 0}
    
    # RIGHT: return a new derived key
    return {"position_rate": obs["position"] - self.prev_position}
```

**2. State resets on episode boundaries.** Each time an episode resets (in V2 Redis training), the SDK calls `Perceptor.reset()`, which creates a new instance of your perceptor impl via `impl_cls()`. All accumulated state (previous values, filter buffers, running statistics) is wiped. If you initialize expensive resources in `__init__` (loading ML models, etc.), you pay that cost on every episode reset.

```python
class ExpensivePerceptor(PerceptorImpl):
    def __init__(self):
        # This runs on EVERY episode reset — can be slow
        self.model = load_large_model()
        self.history = []
```

**3. `obs_spec` may be None.** In some code paths (unit tests, certain V2 call sites), `obs_spec` is passed as `None`. Your `compute()` method must handle this gracefully:

```python
async def compute(self, obs_spec, obs):
    # Don't assume obs_spec is a valid Space object
    if obs_spec is None:
        # Safe default
        pass
    return {...}
```

**4. `filtered_sensor_space()` is not enforced.** The training pipeline does not call this method or validate that you only read the sensors you declared. It's purely for documentation. If your perceptor reads sensors not listed in `filtered_sensor_space()`, no error is raised — but downstream tools may not know about those dependencies.

**5. Perceptors are shared across Ray actors.** In distributed training (Ray), the `Perceptor` wrapper returns `self` from `__deepcopy__()`, meaning the same wrapper instance is shared across multiple Ray workers. For stateful perceptors, this can cause state corruption if not carefully managed. Prefer immutable computation or explicitly synchronize state.

**6. Remote address ambiguity.** If `remote_address` starts with `http`, the SDK treats it as a gRPC container. If it's a filesystem path, the SDK loads the module locally from that directory. This distinction is not advertised in error messages — if a path lookup fails, you may see a cryptic gRPC error instead.

## Code Examples

### Example 1: Simple Noise Filter

```python
class ExponentialMovingAverage(PerceptorImpl):
    """Smooth noisy sensor readings with exponential moving average."""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.filtered_value = None
    
    async def compute(self, obs_spec, obs):
        raw_value = obs["noisy_sensor"]
        
        if self.filtered_value is None:
            self.filtered_value = raw_value
        else:
            self.filtered_value = (
                self.alpha * raw_value + 
                (1 - self.alpha) * self.filtered_value
            )
        
        return {"filtered_sensor": self.filtered_value}
    
    def filtered_sensor_space(self, obs):
        return ["noisy_sensor"]

# Register
perceptor = Perceptor("ema-filter", ExponentialMovingAverage(alpha=0.2))
agent.add_perceptor(perceptor)
```

### Example 2: Derivative Perceptor (Velocity from Position)

```python
import time

class DerivativePerceptor(PerceptorImpl):
    """Compute time-derivative of a sensor value."""
    
    def __init__(self):
        self.previous_value = None
        self.previous_time = None
    
    async def compute(self, obs_spec, obs):
        current_value = obs["position"]
        current_time = time.time()
        derivative = 0.0
        
        if self.previous_value is not None and self.previous_time is not None:
            dt = current_time - self.previous_time
            if dt > 0:
                derivative = (current_value - self.previous_value) / dt
        
        self.previous_value = current_value
        self.previous_time = current_time
        return {"velocity": derivative}
    
    def filtered_sensor_space(self, obs):
        return ["position"]

# Register
agent.add_perceptor(Perceptor("velocity", DerivativePerceptor()))
```

### Example 3: Chained Perceptors (Velocity, then Kinetic Energy)

```python
class KineticEnergyPerceptor(PerceptorImpl):
    """Compute kinetic energy from velocity and mass."""
    
    def __init__(self, mass=1.0):
        self.mass = mass
    
    async def compute(self, obs_spec, obs):
        # This runs AFTER velocity perceptor, so "velocity" is available
        velocity = obs.get("velocity", 0.0)
        ke = 0.5 * self.mass * (velocity ** 2)
        return {"kinetic_energy": ke}
    
    def filtered_sensor_space(self, obs):
        # Depends on both the original sensor AND perceptor output
        return ["mass", "velocity"]

# Register in order
agent.add_perceptor(Perceptor("velocity", DerivativePerceptor()))
agent.add_perceptor(Perceptor("ke", KineticEnergyPerceptor(mass=5.0)))
```

### Example 4: Stateful Feature Engineering

```python
import numpy as np

class FeatureEngineer(PerceptorImpl):
    """Extract sliding-window statistics from a timeseries."""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = []
    
    async def compute(self, obs_spec, obs):
        value = obs["signal"]
        self.history.append(value)
        
        # Keep only the last window_size values
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Compute statistics
        if len(self.history) >= 2:
            return {
                "signal_mean": float(np.mean(self.history)),
                "signal_std": float(np.std(self.history)),
                "signal_trend": float(self.history[-1] - self.history[0]),
            }
        else:
            return {
                "signal_mean": float(value),
                "signal_std": 0.0,
                "signal_trend": 0.0,
            }
    
    def filtered_sensor_space(self, obs):
        return ["signal"]

# Register
agent.add_perceptor(Perceptor("features", FeatureEngineer(window_size=20)))
```
