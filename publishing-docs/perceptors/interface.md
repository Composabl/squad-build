# Perceptor Interface

## Core implementation

```python
class MyPerceptor(PerceptorImpl):
    async def compute(self, obs_spec, obs) -> dict:
        ...

    async def filtered_sensor_space(self) -> list[str]:
        ...
```

## Notes

- `compute(...)` returns a dict of derived keys/values.
- `filtered_sensor_space()` must include exactly the keys returned by `compute`.
- Perceptors can be stateful (e.g., temporal derivatives).
