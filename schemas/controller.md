# Controller Schema — `SkillController`

A `SkillController` is a deterministic (or rule-based) policy that runs in the same
simulation loop as RL skills but is **never trained**. Use it for PID controllers,
rule-based policies, mathematical expressions, or expert baselines.

## Import

```python
from amesa_core import SkillController
# or explicit:
from amesa_core.agent.skill.skill_controller import SkillController
```

## Interface Contract

All required methods are `async`. Controllers are **peer concepts** to Teachers — they
do not inherit from `SkillTeacher`.

```python
class SkillController(ABC):

    def __init__(self, *args, **kwargs):
        """
        Permissive constructor — accepts any args/kwargs.
        Re-called on every episode reset for local controllers.
        All self.* state is wiped at episode boundaries.
        """
        self.action_space = None    # never set by framework
        self.sensor_space = None    # never set by framework
        self.scenario = None        # set via add_scenario() if needed

    # ── Required ────────────────────────────────────────────────────────

    async def compute_action(
        self,
        transformed_sensors: dict,  # full amesa_sensors dict after perceptor pipeline
        action                      # action from previous step; None on first step
    ):
        """
        Core control logic. Return the action to send to the simulator.
        Return type must match the sim's action space:
          Box          → np.array([...], dtype=np.float32)
          Discrete     → int
          MultiDiscrete → list[int]
          Dict         → dict keyed by sub-space names
          Tuple        → tuple of sub-space values
        Return value is passed through sim_action_space.coerce_sample() — type
        coercion and clamping are done for you (unlike SkillTeacher).
        """

    async def filtered_sensor_space(self) -> list[str]:
        """
        Declare which sensors this controller reads.
        Called at init; result is cached — do not return dynamic values.
        """

    async def compute_success_criteria(
        self,
        transformed_sensors: dict,
        action
    ) -> bool:
        """
        Return True if the episode should be considered a success.
        ⚠️ V1 WARNING: Never called in V1 (processor stubs with pass → returns None).
        Called correctly in V2 (EventControllerProcessor).
        Implement for V2 compatibility but don't rely on it in V1.
        """

    async def compute_termination(
        self,
        transformed_sensors: dict,
        action
    ) -> bool:
        """
        Return True to end the episode early (failure).
        ⚠️ V1 WARNING: Never called in V1 (processor stubs with pass → returns None).
        Called correctly in V2. Same caveat as compute_success_criteria.
        """

    # ── Optional / Dead code warnings ───────────────────────────────────

    async def transform_sensors(self, sensors) -> dict:
        """
        ⚠️ NEVER CALLED in V1 or V2. Dead code.
        Sensor preprocessing must be done inside compute_action() instead.
        """
        return sensors
```

## Registration

Pass the **class** (not an instance) to `Skill()`. Type inference via
`issubclass(impl_cls, SkillController)` assigns `SkillDirector.CONTROLLER`
automatically.

```python
from amesa_core import Agent
from amesa_core.agent.skill import Skill

skill = Skill("my-controller", MyController)
agent.add_skill(skill)
```

No special registration method for controllers. `agent.add_skill()` handles everything.

## Training kwargs are silently ignored

```python
# These are silently dropped for controllers — no error, no warning:
Skill("ctrl", MyController, training_cycles=500, workers=4, fc_layers=[64, 64])
```

## Selector Controller

If the controller routes to child skills (returns an integer index), use `SkillSelector`:

```python
from amesa_core.agent.skill.skill_selector import SkillSelector

selector = SkillSelector(
    "my-selector",
    MySelectorController,           # SkillController subclass returning int index
    children=["child-a", "child-b"]
)
agent.add_selector_skill(selector, ...)
```

`compute_action()` must return an integer (or int-castable value) that is the
0-based index of the child skill to run.

## Per-Step Execution (V1)

```
sim step
  → SkillControllerProcessor._execute(sim_sensors)
    → compute_single_action(sim_sensors, prev_action)
      → convert_sim_sensors_to_amesa_sensors()  [sensor mapping + perceptors]
      → controller.compute_action(amesa_obs, action)   ← YOUR CODE
      → sim_action_space.coerce_sample(action)         [type coercion]
      → return action → sim
```

**Not called in V1:** `transform_sensors()`, `compute_success_criteria()`,
`compute_termination()`.

## Episode Reset

Local controllers are fully re-instantiated every episode:

```python
self.controller = skill.get_impl_cls_instance()  # new __init__()
```

All per-episode state is wiped. For cross-episode state, use class-level variables
or external storage.

## V1 vs V2 Differences

| Aspect | V1 (Ray/RLlib) | V2 (Redis Streams) |
|---|---|---|
| `compute_success_criteria` | **Never called** — stub returns None | Called correctly |
| `compute_termination` | **Never called** — stub returns None | Called correctly |
| `transform_sensors` | Never called | Never called (dead code in both) |
| Episode termination control | Must be handled by the simulator | Works via `compute_termination` |

## Minimal Working Example

```python
from amesa_core import SkillController
from typing import Dict, List
import numpy as np

class ProportionalController(SkillController):
    """Proportional setpoint controller. Baseline for RL benchmarking."""

    def __init__(self, *args, **kwargs):
        self.kp = 0.8
        self.setpoint = 0.0
        self.success_threshold = 0.05
        self.out_of_bounds = 10.0

    async def compute_action(self, transformed_sensors: Dict, action) -> np.ndarray:
        position = transformed_sensors["position"][0]
        error = self.setpoint - position
        output = float(np.clip(self.kp * error, -1.0, 1.0))
        return np.array([output], dtype=np.float32)

    async def filtered_sensor_space(self) -> List[str]:
        return ["position", "velocity"]

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        return abs(transformed_sensors["position"][0] - self.setpoint) < self.success_threshold

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        return abs(transformed_sensors["position"][0]) > self.out_of_bounds
```

## Controller vs. Teacher Decision

| Situation | Use |
|---|---|
| Policy is fully specified (PID, rule-based, expert system) | `SkillController` |
| Need to benchmark against a hand-coded baseline | `SkillController` |
| Need ONNX export, checkpointing, or RL training | `SkillTeacher` |
| Policy must adapt from experience | `SkillTeacher` |

## Critical Notes

- `SkillControllerJson` is a dead stub. Do not use it.
- `transform_action` is not in the base class ABC. Any `transform_action` you define
  on a local controller is dead code in both V1 and V2.
- Sensor space is inherited from the first trained RL skill in a mixed agent. If the
  agent has no RL skills, the controller's sensor space is never set.
- Training kwargs (`fc_layers`, `workers`, `training_cycles`) on `Skill(...)` are
  silently dropped when `impl_cls` is a `SkillController`.
