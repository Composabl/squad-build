# SkillTeacher Interface

## Required methods

```python
async def compute_reward(self, transformed_sensors: dict, action, sim_reward: float) -> float
async def compute_success_criteria(self, transformed_sensors: dict, action) -> bool
async def transform_action(self, transformed_sensors: dict, action)
async def filtered_sensor_space(self) -> list[str]
```

## Optional methods

```python
async def compute_termination(self, transformed_sensors: dict, action) -> bool
async def transform_sensors(self, sensors, action) -> dict
async def compute_action_mask(self, transformed_sensors: dict, action)
async def get_custom_action_space(self)
```

## Parameter/return notes

- `transformed_sensors` is the post-`transform_sensors` view.
- `action` shape depends on action space (`Discrete`, `Box`, `Dict`, etc.).
- `sim_reward` is the simulator reward; blend or ignore as needed.
- `compute_reward` must return a Python `float`.
- `filtered_sensor_space()` returns `list[str]` sensor names.

## Action mask shape reminders

- `Discrete(n)`: length `n`
- `Box(shape=(N,))`: length `N*2` (mean/std pairs)
- `Tuple`: tuple/list per sub-space
- `Dict`: dict keyed by action-space keys
