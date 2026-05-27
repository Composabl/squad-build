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

### Method parameter types

| Parameter | Type | Description |
|---|---|---|
| `transformed_sensors` | `dict[str, Any]` | Sensor readings after `transform_sensors` has run. Keys are the sensor names declared in `filtered_sensor_space`. Values are typically `numpy.float32` scalars or arrays. |
| `action` | `numpy.ndarray \| int \| dict` | The **previous** action sent to the sim. Available for feedback or stateful controllers. On the first step of each episode, this reflects the initial action from the sim's reset. |

### Return types

| Method | Return type | Notes |
|---|---|---|
| `compute_action` | Same type/shape as the sim's action space | Must be directly consumable by the sim without further processing. Return a list or ndarray for `Box` spaces, an int for `Discrete`. |
| `filtered_sensor_space` | `list[str]` | Sensor names the controller needs. Only these keys will appear in `transformed_sensors`. |
| `compute_success_criteria` | `bool` | Return `True` to signal the controller has achieved its goal. Ends the episode. |
| `compute_termination` | `bool` | Return `True` to abort the episode (e.g. unsafe state reached). **Required** — unlike `SkillTeacher`, there is no default. |

## Optional methods (have defaults)

```python
async def transform_sensors(self, sensors) -> dict
# Default: return sensors unchanged
# Note: signature takes only sensors (no action), unlike SkillTeacher which receives (sensors, action).
# Pre-processes the raw sim observation before compute_action is called.
# sensors is the raw observation dict from the sim, before any filtering.

async def compute_action_mask(self, transformed_sensors: dict, action) -> list | tuple | dict | None
# Default: return None  ← no masking
# Same mask semantics as SkillTeacher.compute_action_mask. Return a mask shaped to the action space
# to constrain which actions the runtime may select.
# Discrete(n):    list/array of length n  (1 = allowed, 0 = masked)
# Box(shape=(N,)): list/array of length N*2  (mean0, std0, mean1, std1, … per dimension)
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

## Adding a scenario

```python
from amesa_core.agent.skill.skill_controller import SkillController
from amesa_core.agent.scenario.scenario import Scenario

skill = Skill("my-controller-skill", MyController)
skill.add_scenario({"initial_pos": 0.0, "target": 3.0})
```

`add_scenario` accepts a `dict` or `Scenario` object. The scenario state is passed to the sim at the start of each episode. Controllers support scenarios in the same way as teachers — the scenario dict is serialized into `config.scenarios[]` in the job JSON.

---

## Job JSON schema

In the serialized agent JSON, each controller skill entry follows `SkillControllerOptions`.

```json
{
  "name": "my-controller-skill",
  "type": "SkillController",
  "config": {
    "remote_address": null,
    "impl_cls": {
      "cls_name": "MyController",
      "cls_module": "my_agent.controller",
      "cls_src": "<base64-pickle>",
      "cls_deps": []
    },
    "impl_cls_data": {
      "guidance": null,
      "goals": [],
      "constraints": null
    },
    "model_io": {},
    "scenarios": [],
    "scenarios_current_idx": 0
  }
}
```

| Field | Default | Description |
|---|---|---|
| `remote_address` | `null` | URL of a remotely-hosted controller. When set, `impl_cls` is ignored and requests are forwarded. |
| `impl_cls` | — | Serialized controller class (produced by `Agent.export()`). |
| `impl_cls_data` | — | Goals and guidance attached to the controller. `goals` is unused for controllers (no reward signal needed), but the field is present for schema consistency. |
| `model_io` | `{}` | Sensor and action space overrides. |
| `scenarios` | `[]` | Scenario configurations cycled during execution. |

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
