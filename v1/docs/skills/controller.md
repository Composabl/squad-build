# SkillController

`SkillController` is the base class for deterministic / rule-based skills. Subclass it when the action can be computed directly from the observation (PID, MPC, lookup table, etc.).

## Required methods

All four must be implemented:

```python
async def compute_action(self, transformed_sensors: dict, action) -> any
async def filtered_sensor_space(self) -> list[str] | any
async def compute_success_criteria(self, transformed_sensors: dict, action) -> bool
async def compute_termination(self, transformed_sensors: dict, action) -> bool
```

## Optional methods (have defaults)

```python
async def transform_sensors(self, sensors) -> dict
# Default: return sensors unchanged
# Note: signature takes only sensors (no action), unlike SkillTeacher
```

## Full interface

```python
from amesa_core.agent.skill.skill_controller import SkillController
from amesa_core.spaces import Box
from typing import Dict

class MyController(SkillController):

    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))

    async def compute_action(self, transformed_sensors: Dict, action):
        """
        Compute and return the action based on current sensors.
        This is called every step instead of a learned policy.
        """
        error = transformed_sensors.get("error", 0.0)
        control = -0.5 * error   # simple P-controller
        return [float(control)]

    async def filtered_sensor_space(self):
        """Return sensor names needed for compute_action."""
        return ["value", "target", "error"]

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        """Return True when the controller has achieved its goal."""
        return abs(transformed_sensors.get("error", 1.0)) < 0.05

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        """Return True to end the episode (e.g., unsafe state or goal reached)."""
        return transformed_sensors.get("value", 0.0) > 100.0

    async def transform_sensors(self, sensors) -> Dict:
        """Optional: preprocess sensors before compute_action is called."""
        return sensors
```

## Instantiating

```python
from amesa_core.agent.skill.skill import Skill

skill = Skill("my-controller-skill", MyController)
```

Controllers do not need `training_cycles` (no ML training).

---

## Uploading as a portable component

To upload a controller as a standalone portable component, include a `pyproject.toml` alongside your implementation:

```toml
[project]
name = "my-controller"
version = "0.1.0"
description = "A rule-based controller for my skill."
authors = [{ name = "Your Name", email = "you@example.com" }]
dependencies = [
    "amesa-core",
]

[amesa]
type = "skill-controller"
entrypoint = "my_module.controller:MyController"
```

- `type` must be `"skill-controller"`
- `entrypoint` is `"module.path:ClassName"`

---

## ⚠️ Quirks

**`compute_termination` is required** — Unlike `SkillTeacher` (where it defaults to `False`), you must implement it in a controller.

**`transform_sensors` takes only `sensors`** — The controller version does not receive `action` as a second argument (the teacher version does).

**Controllers can be used as `SkillSelector` implementations** — Passing `MyController` to `SkillSelector(name, MyController)` creates a rule-based selector.
