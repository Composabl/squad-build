# Sensor Schema

Sensors define the named observation variables the agent can see. They map raw sim
output into named slots in the AMESA sensor dict, with optional normalization.

## Import

```python
from amesa_core.agent.sensors.sensor import Sensor
# or:
from amesa_core import Sensor
```

## Constructor

```python
Sensor(
    name: str,                              # unique key in the sensor dict
    description: str,                       # human-readable description
    mapping: Callable[[dict], Any] = None,  # lambda to extract from raw obs
    normalize: bool = False,                # whether to normalize output
    low: float = None,                      # normalization lower bound
    high: float = None,                     # normalization upper bound
)
```

| Parameter | Required | Description |
|---|---|---|
| `name` | Yes | The key agents use to read this sensor (`transformed_sensors["name"]`) |
| `description` | Yes | Text description (documentation only, not enforced) |
| `mapping` | Recommended | Lambda that extracts this sensor's value from the raw sim obs dict |
| `normalize` | No | If True, scale output to [-1, 1] using `low`/`high` |
| `low` | Conditional | Required when `normalize=True` |
| `high` | Conditional | Required when `normalize=True` |

## Registration

```python
agent = Agent()
agent.add_sensors([
    Sensor("temperature", "Process temperature in Celsius", lambda obs: obs["temperature"]),
    Sensor("pressure",    "Tank pressure in PSI",           lambda obs: obs["pressure"]),
    Sensor("setpoint",    "Target temperature",             lambda obs: obs["setpoint"]),
])
```

`add_sensors()` accepts a list. Call it once before adding skills.

## Mapping Lambda

The mapping lambda receives the full raw observation dict from the simulator and
returns the value for this specific sensor:

```python
# Simple passthrough (key name matches sim key)
Sensor("air_temp", "Air temperature", lambda obs: obs["air_temp"])

# Transformation
Sensor("air_temp_celsius", "Temp in C", lambda obs: (obs["air_temp_f"] - 32) * 5/9)

# Computed from multiple sim outputs
Sensor("speed", "Speed", lambda obs: (obs["vx"]**2 + obs["vy"]**2)**0.5)

# Index into array obs
Sensor("position_x", "X position", lambda obs: obs["position"][0])
```

If no mapping lambda is provided, the SDK looks for a key in the raw obs that
matches the sensor name exactly.

## Normalization

```python
Sensor(
    "temperature",
    "Process temperature",
    lambda obs: obs["temperature"],
    normalize=True,
    low=0.0,
    high=200.0,
)
```

Normalized sensors are scaled to the range expected by RL algorithms. If your
sensors span very different scales, normalization improves learning stability.

Perceptor outputs are NOT automatically normalized — normalize them in the
perceptor's `compute()` method or in the teacher's `transform_sensors()`.

## Relationship to Other Components

- **Perceptors** receive the full AMESA sensor dict (all named sensors) and add new
  derived keys. They do not replace existing sensor values.
- **Teacher.filtered_sensor_space()** returns a list of sensor names (including
  perceptor-derived names) that the RL policy observes.
- **Teacher.transform_sensors()** receives the full AMESA sensor dict and can
  further modify values before filtering.

## Convention

Name sensors after what they represent, not where they come from. Prefer:
- `"temperature"` over `"sensor_0"`
- `"air_temp"`, `"target_temp"`, `"ambient_temp"` over `"temp_1"`, `"temp_2"`, `"temp_3"`

Sensor names become keys in every downstream dict — teachers, perceptors, scenarios
all reference them by name.
