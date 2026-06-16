# Perceptor Interface

`PerceptorImpl` transforms raw observations into derived features consumed by skills.

## Full scaffold

```python
from amesa_core.agent.perceptor.perceptor_impl import PerceptorImpl

class MyPerceptor(PerceptorImpl):
    async def compute(self, obs_spec, obs) -> dict:
        """Produce derived variables from raw observations.

        :param obs_spec: Observation space specification.
        :param obs: Raw observation dict from the simulator.
        :returns: Dict of derived feature key/value pairs.
        :rtype: dict
        :raises NotImplementedError: if not overridden in the subclass.
        :raises Exception: if a returned key collides with a key already present
            in the sensor space (``"Perceptor X has a key Y that already exists
            in the sensor space dictionary"``).
        :raises ValueError: if a returned value is ``None``; the framework calls
            ``flatten_object_numpy`` on each value when building the observation
            space and ``None`` is not a valid leaf (``"flatten_object_numpy: None
            is not a valid leaf value"``).
        """
        return {
            "derived_value": float(obs["a"]) - float(obs["b"]),
        }

    def filtered_sensor_space(self, obs) -> list[str]:
        """Declare which raw sensor keys this perceptor depends on.

        :param obs: Raw observation dict (used for dynamic inspection if needed).
        :returns: List of raw sensor key strings.
        :rtype: list[str]
        :raises NotImplementedError: if not overridden in the subclass.
        """
        return ["a", "b"]
```

## Methods and intended use

### `compute(self, obs_spec, obs) -> dict` (required)

Use this to calculate perceptor outputs from simulator observations. Capabilities include combining sensors, applying normalization, feature engineering, and exposing stable keys that downstream teachers/controllers consume.

### `filtered_sensor_space(self, obs) -> list[str]` (required)

Use this to declare the raw sensor keys the perceptor reads. This acts as the perceptor's dependency contract and helps keep observation usage explicit and minimal.

## Optional methods

`PerceptorImpl` has no optional lifecycle methods you must implement.

## Method contracts

- `compute(...)` returns a dict of derived keys/values.
- `filtered_sensor_space(...)` returns the raw input sensor keys this perceptor reads.
- Returned keys should be stable and documented in component metadata (`variables` in publish config).
