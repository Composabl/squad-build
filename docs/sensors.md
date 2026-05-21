# Sensors

`Sensor` maps a raw sim observation (dict or array) to a named scalar or array value that skills and perceptors can reference by name.

## Constructor

```python
Sensor(
    sensor_name: str,                              # required — unique name
    description: str = "",
    sensor_mapping: str | Callable | None = None,  # optional lambda
    normalize: bool = False,
    normalizer: NormalizeSensor | None = None,
)
```

## Usage

```python
from amesa_core.agent.sensors.sensor import Sensor

# Simple named sensor — no mapping, value passed through by key
s1 = Sensor("temperature")

# Lambda mapping — extracts a value from the observation dict
s2 = Sensor("temp_error", sensor_mapping=lambda s: s["air_temp"] - s["target_temp"])

# String form of lambda (preferred for safety — see quirk below)
s3 = Sensor("speed", sensor_mapping="lambda s: s['speed']")
```

## Normalization

```python
s = Sensor("position", normalize=True)
# Normalizer is lazily created on first sample using the data shape.
# You can also provide one explicitly:
from amesa_core.agent.sensors.normalize_sensor import NormalizeSensor
s = Sensor("position", normalize=True, normalizer=NormalizeSensor(...))
```

## Sensor lists

Pass a list to `agent.add_sensors()`:

```python
agent.add_sensors([
    Sensor("x", sensor_mapping="lambda s: s['x']"),
    Sensor("y", sensor_mapping="lambda s: s['y']"),
])
```

## `filtered_sensor_space` interaction

In `SkillTeacher` and `SkillController`, `filtered_sensor_space()` returns a list of sensor **names** (strings). Only sensors with matching names are forwarded to `transformed_sensors` in the skill.

---

## ⚠️ Quirks

**Lambda serialization** — Sensors are serialized to JSON when building the agent graph. If you pass a `Callable`, the SDK attempts to extract its source string via `inspect`. This can fail for lambdas defined inside loops, closures, or dynamically. **Always use the string form** `"lambda s: s['key']"` to guarantee round-trip serialization.

```python
# Risky — may not serialize
Sensor("v", sensor_mapping=lambda s: s["v"])

# Safe
Sensor("v", sensor_mapping="lambda s: s['v']")
```

**Dict-style access in lambdas** — Inside the lambda, the observation is wrapped in an `AttrDict`, so both `s["key"]` and `s.key` work.
