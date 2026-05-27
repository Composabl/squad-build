# SkillController Interface

## Required methods

```python
async def compute_action(self, transformed_sensors: dict, action_mask) -> any
async def compute_success_criteria(self, transformed_sensors: dict, action) -> bool
async def compute_termination(self, transformed_sensors: dict, action) -> bool
async def filtered_sensor_space(self) -> list[str]
```

## Optional methods

```python
async def transform_sensors(self, sensors) -> dict
async def compute_action_mask(self, transformed_sensors: dict, action)
```

## Notes

- `transform_sensors` takes only `sensors` (no `action` parameter).
- Action-mask conventions follow the teacher mask conventions (`Discrete`, `Box`, `Tuple`, `Dict`).
- `compute_termination` is required for controllers.
