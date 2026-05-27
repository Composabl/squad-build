# Coordinated Interface

## SkillCoach required methods

```python
async def compute_reward(self, transformed_sensors: dict, action: dict, sim_reward: float) -> dict[str, float]
async def compute_success_criteria(self, transformed_sensors: dict, action: dict) -> bool
async def compute_termination(self, transformed_sensors: dict, action: dict) -> bool
```

## SkillCoach optional methods

```python
async def transform_sensors(self, sensors, action) -> dict
async def transform_action(self, transformed_sensors: dict, action: dict) -> dict
async def compute_action_mask(self, transformed_sensors: dict, action: dict)
```

## Notes

- Reward dict keys must exactly match child skill names.
- Success/termination apply to the coordinated unit as a whole.
- Per-child action-mask values follow teacher mask conventions.
