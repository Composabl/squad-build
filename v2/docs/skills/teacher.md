# SkillTeacher

`SkillTeacher` is the base class for ML-trained skills. Subclass it to define the reward signal, success condition, and how the model's output is mapped to an action.

## Required methods

These must be implemented:

```python
async def compute_reward(self, transformed_sensors: dict, action, sim_reward: float) -> float
async def compute_success_criteria(self, transformed_sensors: dict, action) -> bool
async def transform_action(self, transformed_sensors: dict, action) -> any
async def filtered_sensor_space(self) -> list[str]
```

## Optional methods (have defaults)

```python
async def compute_termination(self, transformed_sensors: dict, action) -> bool
# Default: return False  ← episode never terminates early from teacher side

async def transform_sensors(self, sensors, action) -> dict
# Default: return sensors unchanged

async def compute_action_mask(self, transformed_sensors: dict, action) -> list | tuple | dict | None
# Default: return None  ← no masking
# Return type depends on the action space — the trainer calls space.fit_mask() on the result.
# Discrete(n):       list/array of length n  (1 = allowed, 0 = masked)
# Box(shape=(N,)):   list/array of length N*2  (mean0, std0, mean1, std1, … per dimension)
# MultiDiscrete:     flat array, tuple, or list of per-subspace arrays
# Tuple(spaces):     tuple/list with one mask element per subspace
# Dict(spaces):      dict with one mask value per key (recursively fitted)
# See docs/spaces.md → "Action masks" for full shape requirements.

async def get_custom_action_space(self) -> Space | None
# Default: return None  ← use the sim's native action space
# Override to return a custom action space (e.g. Box, Discrete, Dict) that the
# policy trains with instead of the space reported by the sim's action_space_info().
# Works for both local and remote teachers.
```

## Full interface

```python
from amesa_core.agent.skill.skill_teacher import SkillTeacher
from amesa_core.spaces import Box
from typing import Dict, List

class MyTeacher(SkillTeacher):

    def __init__(self):
        super().__init__()
        # self.action_space is used by teacher methods (e.g. sampling a fallback
        # action in transform_action). The trainer itself gets the action space
        # from the sim's action_space_info(), not from the teacher.
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))

    async def compute_reward(self, transformed_sensors: Dict, action, sim_reward: float) -> float:
        """
        Return a scalar reward for this step.
        Use transformed_sensors (after transform_sensors runs) for your signal.
        sim_reward is whatever the sim returned — you may use or ignore it.
        """
        error = abs(transformed_sensors.get("error", 0.0))
        return -error

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        """Return True when the skill has achieved its goal."""
        return abs(transformed_sensors.get("error", 1.0)) < 0.1

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        """Return True to terminate the episode early (e.g. unsafe state)."""
        return transformed_sensors.get("value", 0.0) > 100.0

    async def transform_action(self, transformed_sensors: Dict, action):
        """
        Post-process the model's raw action before it is sent to the sim.
        Return action unchanged if no transformation is needed.
        """
        return action

    async def filtered_sensor_space(self) -> List[str]:
        """
        Return the sensor names this skill cares about.
        Only these keys will appear in transformed_sensors.
        Include perceptor output keys here too.
        """
        return ["value", "target", "error"]

    async def transform_sensors(self, sensors, action) -> Dict:
        """Optional: pre-process sensors before they reach compute_reward etc."""
        return sensors
```

## Custom action space

By default the trainer uses the action space reported by the sim's `action_space_info()`. To override this, implement `get_custom_action_space()` on your teacher:

```python
from amesa_core.spaces import Box

class MyTeacher(SkillTeacher):
    async def get_custom_action_space(self):
        return Box(low=-1.0, high=1.0, shape=(1,))
```

This is the preferred pattern. It works for both local and remote teachers. The custom space is preserved across rollout worker sync and takes priority over the sim's reported space.

`self.action_space` set in `__init__` is available for use **within your teacher methods** (e.g. `self.action_space.sample()` in `transform_action`), but the trainer does **not** read it — use `get_custom_action_space()` to influence training.

As a fallback, you can also pass `custom_action_space=` to the `Skill` constructor:

```python
Skill("my-skill", MyTeacher, custom_action_space=Box(low=-1.0, high=1.0, shape=(1,)))
```

> **Note:** `custom_action_space=` is not supported as a `SkillSelector` constructor kwarg. Use `get_custom_action_space()` on the selector's teacher instead.

## Adding a scenario

```python
from amesa_core.agent.skill.skill import Skill

skill = Skill("my-skill", MyTeacher, training_cycles=100)
skill.add_scenario({"initial_pos": [0.0, 5.0], "target": 3.0})
```

---

## ⚠️ Quirks

**`is_compute_done` uses AND but the runtime uses OR** — The `SkillTeacher.is_compute_done()` helper returns `compute_success_criteria AND compute_termination`. The runtime, however, ends the episode when **either** returns `True` (`truncated = teacher_success or teacher_terminated`). Do not rely on `is_compute_done` to reason about when episodes end — use the individual methods directly.

**`compute_success_criteria` ends episodes** — When `compute_success_criteria` returns `True`, the runtime sets `truncated=True` and the episode ends immediately. This is independent of `compute_termination`. Design your success thresholds carefully.

**Missing sensor keys default to 0 — a silent failure mode** — If a key used in `compute_success_criteria` or `compute_termination` is absent from `transformed_sensors` (e.g. because a perceptor was not enabled), `dict.get("key", 0.0)` returns `0.0`. A threshold like `abs(0.0) < 0.3` is True, so the episode will succeed (and terminate) on every step. If you see near-instant episode termination, check that all perceptor outputs your teacher depends on are actually present.

**`compute_termination` is optional in Teacher** — it defaults to `False`. `SkillController.compute_termination` is **required**. This asymmetry is a common source of confusion.

**`self.action_space` is not the trainer's action space** — `self.action_space` set in `__init__` is only available within your teacher methods (e.g. for sampling in `transform_action`). The trainer uses the sim's `action_space_info()` by default. To override the trainer's action space, implement `get_custom_action_space()` — see the "Custom action space" section above.

**`filtered_sensor_space` returns names, not spaces** — Return a `list[str]`, not `list[Sensor]` objects.
