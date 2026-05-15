# Teacher API

## Overview

A **Teacher** defines the learning contract for a reinforcement learning skill. It tells the training loop what the reward signal is, when the skill has succeeded or failed, which sensors to observe, and how to pre-process observations and post-process actions.

Every RL skill in AMESA requires a Teacher. The Teacher is where you encode domain expertise — the "what good looks like" for a skill. Use a Teacher when you need custom reward shaping, success criteria, action transformation, or observation filtering for a skill being trained with RL.

## Interface

`SkillTeacher` is an abstract base class. All user-defined teachers extend it directly.

```
SkillTeacher(ABC)
    └── Goal(SkillTeacher)          ← structured objective base (maintain, maximize, etc.)
         └── CoordinatedGoal(Goal) ← combines multiple Goals into one teacher
    └── SkillTeacherJson(SkillTeacher) ← framework-internal; not for user subclassing
```

**Module path:** `amesa_core.agent.skill.skill_teacher`

**Import:**

```python
from amesa_core import SkillTeacher
# or explicit path:
from amesa_core.agent.skill.skill_teacher import SkillTeacher
```

**Base class definition:**

```python
class SkillTeacher(ABC):
    def __init__(self, *args, **kwargs):
        self.action_space: "Space" = None   # never populated by the framework
        self.sensor_space: "Space" = None   # never populated by the framework
        self.scenario: Scenario = None      # set via add_scenario() before training
```

## Required Methods

All four methods are abstract and must be implemented. All are `async`.

---

### `compute_reward`

```python
@abstractmethod
async def compute_reward(
    self,
    transformed_sensors: Dict,
    action,
    sim_reward: float
) -> float:
```

Compute the reward signal for one training step.

| Parameter | Type | Description |
|---|---|---|
| `transformed_sensors` | `Dict` | Filtered and normalized sensor dict produced by `transform_sensors()` → `filtered_sensor_space()` pipeline. Keys match sensor names. |
| `action` | any | The action that was sent to the simulator on the **previous** step. May be `None` on the first step. |
| `sim_reward` | `float` | Raw reward returned by the simulator. |

**Returns:** `float` — the reward value passed to the RL training algorithm.

**Called by:** `SkillProcessor.step()` once per training step, after the sim returns.

**Constraints:** Must return a scalar float. Returning `None` or a non-numeric crashes the training loop.

---

### `compute_success_criteria`

```python
@abstractmethod
async def compute_success_criteria(
    self,
    transformed_sensors: Dict,
    action
) -> bool:
```

Determine whether the skill has achieved its goal this step.

| Parameter | Type | Description |
|---|---|---|
| `transformed_sensors` | `Dict` | Filtered, normalized sensor dict. |
| `action` | any | Previous step's action. |

**Returns:** `bool` — `True` ends the episode as a **success**. The `success_counter` increments by 1, which drives curriculum advancement (scenario progression).

**Called by:** `SkillProcessor.step()` once per step.

> **Note:** If `compute_termination` also returns `True` on the same step, termination takes precedence — the episode ends but the success counter is **decremented**, not incremented.

---

### `transform_action`

```python
@abstractmethod
async def transform_action(
    self,
    transformed_sensors: Dict,
    action
) -> any:
```

Post-process the raw action produced by the RL model before it is sent to the simulator.

| Parameter | Type | Description |
|---|---|---|
| `transformed_sensors` | `Dict` | Filtered, normalized sensor dict. |
| `action` | any | Raw action from the RL model (already unsquashed from internal representation). |

**Returns:** the final action passed to the simulator. Can be the same object or a modified version.

**Called by:** `SkillProcessor.process_action()` after each model inference.

**Use cases:** clipping to physical limits, unit conversion from normalized to raw, remapping action dimensions.

---

### `filtered_sensor_space`

```python
@abstractmethod
async def filtered_sensor_space(self) -> List[str]:
```

Declare which sensors this skill observes.

**Returns:** `List[str]` — sensor names matching keys in `Agent.sensors`. These define the **observation space shape** of the RL policy.

**Called by:** `SkillProcessor.init()` — **once at startup**, not per step. The result is cached for the life of the skill processor.

**Fallback:** If the returned list is `None` or empty, the framework uses all agent sensors:

```python
if self.filter_keys is None or len(self.filter_keys) == 0:
    self.filter_keys = [sensor.name for sensor in self.context.agent.sensors]
```

---

## Optional Methods / Hooks

These methods have default implementations. Override them when you need the behavior.

---

### `compute_termination`

```python
async def compute_termination(
    self,
    transformed_sensors: Dict,
    action
) -> bool:
    return False  # default
```

Signal an **early failure** termination. `True` ends the episode as a failure; the `success_counter` decrements by 1 (floor 0).

**Called by:** `SkillProcessor.step()` once per step.

---

### `compute_action_mask`

```python
async def compute_action_mask(
    self,
    transformed_sensors: Dict,
    action
) -> List[bool]:
    return None  # default — no masking
```

Mask discrete actions for the current step. Returns a list of booleans, one per action slot — `True` means the action is allowed. `None` disables masking entirely.

The teacher mask is AND-combined with any sim-provided mask. If the skill has a custom action space, the sim mask is ignored.

**Called by:** `SkillProcessor.compute_action_mask()` inside `process_sim_sensors()`.

---

### `transform_sensors`

```python
async def transform_sensors(self, sensors, action):
    return sensors  # default — pass through
```

Pre-process observations before they reach `filtered_sensor_space` filtering and normalization.

| Parameter | Type | Description |
|---|---|---|
| `sensors` | `Dict` | Full `amesa_sensors` dict — all sensors after the perceptor pipeline. |
| `action` | any | **Always `None`** in all current call paths. Do not write logic that depends on this value. |

**Returns:** a modified sensors dict (or the same dict unchanged).

**Called by:** `SkillProcessor.process_sim_sensors()` once per step, before filtering.

**Use cases:** feature engineering, derived signals, unit conversion.

---

### `add_scenario`

```python
def add_scenario(self, scenario: Scenario):
    self.scenario = scenario
```

Called by the framework before training starts to inject the current `Scenario`. Override to react to scenario changes.

---

### `is_compute_done`

```python
async def is_compute_done(self, transformed_sensors, action) -> bool:
    return (
        await self.compute_success_criteria(transformed_sensors, action)
        and await self.compute_termination(transformed_sensors, action)
    )
```

Returns `True` only when both success criteria AND termination are simultaneously true. **Not called in the standard training loop** — only invoked by `env_validator.py`. Overriding this has no effect on training behavior.

---

## Data Flow

How Teacher fits into the per-step training loop:

```
Simulator
    │ raw obs
    ▼
Sensor mapping (convert_sim_sensors_to_amesa_sensors)
    │ amesa_sensors: Dict[sensor_name → value]
    ▼
Perceptor pipeline (ordered; each merges new keys)
    │ amesa_sensors: Dict (possibly augmented)
    ▼
teacher.transform_sensors(amesa_sensors, None)
    │ transformed_sensors: Dict
    ▼
filter_sample(transformed_sensors, filter_keys)     ← from filtered_sensor_space()
    │ filtered_sensors: Dict (subset of keys)
    ▼
sensor.normalize_sample() per key
    │ normalized_filtered_sensors: Dict
    ▼
flatten_sample()
    │ flat obs array → RL model (Ray RLlib PPO)
    ▼
RL model → raw_action
    ▼
teacher.transform_action(transformed_sensors, raw_action)
    │ processed_action
    ▼
Simulator (step)
    │ sim_reward, sim_terminated, new_obs
    ▼
teacher.compute_reward(transformed_sensors, processed_action, sim_reward)
    │ teacher_reward → RL training
teacher.compute_success_criteria(transformed_sensors, processed_action)
    │ → episode end (success) if True and not terminated
teacher.compute_termination(transformed_sensors, processed_action)
    │ → episode end (failure) if True
```

**Instance lifecycle:**

```
SkillProcessor.init()
    ├── teacher = impl_cls()                      ← __init__() called with NO ARGS
    └── await teacher.filtered_sensor_space()     ← observation space defined once

Per episode reset:
    └── teacher = impl_cls()                      ← NEW INSTANCE; all self.* state wiped

Per step:
    ├── process_sim_sensors():
    │   ├── await teacher.transform_sensors(amesa_sensors, None)
    │   └── await teacher.compute_action_mask(transformed, prev_action)
    ├── step():
    │   ├── await teacher.compute_reward(transformed, action, sim_reward)
    │   ├── await teacher.compute_success_criteria(transformed, action)
    │   └── await teacher.compute_termination(transformed, action)
    └── process_action():
        └── await teacher.transform_action(transformed, raw_model_action)
```

---

## Registration

Pass the class (not an instance) to `Skill`. The framework detects `issubclass(impl_cls, SkillTeacher)` and configures a DRL skill automatically.

```python
from amesa_core import Agent, Skill, Sensor, Scenario
from amesa_core import SkillTeacher

class MyTeacher(SkillTeacher):
    ...

skill = Skill(
    "my-skill",
    MyTeacher,
    train_batch_size=4000,
    training_cycles=200,
    fc_layers=[256, 256],
    workers=2,
)
skill.add_scenario(Scenario({"temperature": {"data": 80, "type": "is_equal"}}))

agent = Agent()
agent.add_sensors([Sensor("temperature", "Process temperature")])
agent.add_skill(skill)
```

**`Skill` keyword arguments for Teacher skills:**

| Kwarg | Configures | Default |
|---|---|---|
| `train_batch_size` | `learning.train_batch_size` | `4000` |
| `training_cycles` | `learning.training_cycles` | `None` (train until stopped) |
| `fc_layers` | `model.fc_layers` (hidden layer sizes) | `[256, 256]` |
| `workers` | `resources.workers` | `1` |
| `learner_workers` | `resources.learner_workers` | `0` |
| `envs_per_worker` | `resources.envs_per_worker` | `1` |
| `num_cpus_per_worker` | `resources.num_cpus_per_worker` | `1.0` |
| `num_gpus_per_worker` | `resources.num_gpus_per_worker` | `0.0` |
| `custom_action_space` | `model_io.action_space` | `None` |

---

## Relationship to Other Skills

| Skill Type | Uses Teacher? | Notes |
|---|---|---|
| `Skill("name", MyTeacher)` | ✅ Yes | Standard DRL skill; Teacher defines the learning signal |
| `SkillSelector("name", MyTeacher)` | ✅ Yes | Selector also accepts a Teacher impl for routing logic |
| `Skill("name", MyController)` | ❌ No | Controller — deterministic, no RL policy |
| `SkillCoach` | Related | Coach is the selector-level equivalent; uses the `SkillCoach` ABC |
| `Goal` / `CoordinatedGoal` | Extends Teacher | Declarative alternative; still satisfies the Teacher contract |

**When to use Teacher vs Controller vs Goal:**

- **Teacher** — you need custom reward shaping or domain-specific success logic that doesn't fit a pre-built pattern.
- **Goal / CoordinatedGoal** — your objective maps to maintain, maximize, minimize, approach, or avoid a sensor value. Less code, same RL training.
- **Controller** — no RL needed; the action logic is deterministic and fully specified.

---

## Complete Example

A minimal Teacher for a process control scenario where the skill must maintain a sensor near a setpoint:

```python
import math
from typing import Dict, List
from amesa_core import SkillTeacher, Agent, Skill, Sensor, Scenario


class SetpointTeacher(SkillTeacher):
    """
    Teach an RL policy to maintain `temperature` near a target setpoint.
    Reward decays exponentially with distance from target.
    Episode succeeds when within tolerance; fails when error exceeds limit.
    """

    TARGET = 80.0
    TOLERANCE = 2.0
    FAIL_DISTANCE = 20.0

    def __init__(self):
        # __init__ must accept zero arguments — called with no args on every episode reset
        self.steps = 0

    async def compute_reward(self, transformed_sensors: Dict, action, sim_reward: float) -> float:
        error = abs(transformed_sensors["temperature"] - self.TARGET)
        return -error  # negative error → maximize by minimizing distance

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        error = abs(transformed_sensors["temperature"] - self.TARGET)
        return error <= self.TOLERANCE

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        error = abs(transformed_sensors["temperature"] - self.TARGET)
        return error >= self.FAIL_DISTANCE

    async def transform_action(self, transformed_sensors: Dict, action):
        # Clamp action to valid heater range [0.0, 1.0]
        return max(0.0, min(1.0, action))

    async def filtered_sensor_space(self) -> List[str]:
        # Only expose the temperature sensor to the RL policy
        return ["temperature"]

    async def transform_sensors(self, sensors, action):
        # action is always None here — do not use it
        return sensors


# --- Registration ---
agent = Agent()
agent.add_sensors([Sensor("temperature", "Process temperature in Celsius")])

skill = Skill(
    "temperature-control",
    SetpointTeacher,
    train_batch_size=4000,
    fc_layers=[128, 128],
)
skill.add_scenario(Scenario({"temperature": {"data": 60.0, "type": "set_value"}}))

agent.add_skill(skill)
```

### Goal-Based Alternative (CoordinatedGoal)

For structured objectives, use `CoordinatedGoal` to skip writing `compute_reward`, `compute_success_criteria`, and `compute_termination` manually:

```python
from typing import Dict
from amesa_core.agent.skill.goals.coordinated_goal import CoordinatedGoal
from amesa_core.agent.skill.goals.maintain_goal import MaintainGoal


class SetpointGoalTeacher(CoordinatedGoal):
    def __init__(self):
        temperature_goal = MaintainGoal(
            "temperature",
            "Maintain process temperature near setpoint",
            target=80.0,
            stop_distance=2.0,
        )
        super().__init__([temperature_goal])

    async def transform_sensors(self, sensors, action):
        return sensors

    async def transform_action(self, transformed_sensors: Dict, action):
        return max(0.0, min(1.0, action))
```

---

## Gotchas

- **Teacher is re-instantiated every episode.** `SkillProcessor.reset()` calls `impl_cls()` with no arguments at the start of each episode. All `self.*` state is destroyed at the episode boundary. Your `__init__` must support zero-argument construction. Never use `self.*` for multi-episode statistics.

- **`transform_sensors` `action` parameter is always `None`.** The call site is `await self.teacher.transform_sensors(amesa_sensors, None)` in every current code path. The parameter exists in the signature but nothing passes it. Do not write logic that depends on `action` in `transform_sensors`.

- **`action_space` and `sensor_space` are never populated.** `SkillTeacher.__init__` sets both to `None`. The training pipeline never assigns these on the teacher — they live on the `Skill` object. Calling `self.action_space.sample()` inside any teacher method will raise `AttributeError`.

- **`filtered_sensor_space()` is called once, at init.** Its return value is cached as `self.filter_keys` for the life of the skill processor. Returning different values conditionally per step has no effect.

- **Termination beats success.** If both `compute_success_criteria` and `compute_termination` return `True` on the same step, the episode ends as a failure. The `success_counter` is decremented, not incremented. Check: `if success and not terminated: counter += 1 elif terminated: counter -= 1`.

- **`is_compute_done` is not called during training.** This method is only invoked from `env_validator.py`. Overriding it has no effect on training behavior.

- **`__deepcopy__` returns `self`.** Teachers are not truly deep-copyable. This can cause unexpected state sharing in distributed or config-copy scenarios.

- **Test file has a wrong `transform_sensors` signature.** `test_teacher.py` contains a demo teacher with `async def transform_sensors(self, sensors)` — missing the `action` arg. Copying this pattern can cause failures if the teacher is called with two positional arguments.

- **`SkillTeacher` powers both skills and selectors.** `SkillSelector` also accepts a `SkillTeacher` subclass. The same Teacher class can serve as an RL skill's learning logic or a selector's routing logic.

---

## API Quick Reference

| Method | Required | Signature | Called From | Default |
|---|---|---|---|---|
| `compute_reward` | ✅ | `async (transformed_sensors, action, sim_reward) → float` | `SkillProcessor.step()` | — |
| `compute_success_criteria` | ✅ | `async (transformed_sensors, action) → bool` | `SkillProcessor.step()` | — |
| `transform_action` | ✅ | `async (transformed_sensors, action) → action` | `SkillProcessor.process_action()` | — |
| `filtered_sensor_space` | ✅ | `async () → List[str]` | `SkillProcessor.init()` (once) | — |
| `compute_termination` | ☐ | `async (transformed_sensors, action) → bool` | `SkillProcessor.step()` | `return False` |
| `compute_action_mask` | ☐ | `async (transformed_sensors, action) → List[bool] \| None` | `SkillProcessor.compute_action_mask()` | `return None` |
| `transform_sensors` | ☐ | `async (sensors, action) → Dict` | `SkillProcessor.process_sim_sensors()` | `return sensors` |
| `add_scenario` | ☐ | `(scenario: Scenario) → None` | Framework, before training | sets `self.scenario` |
| `is_compute_done` | ☐ | `async (transformed_sensors, action) → bool` | `env_validator.py` only | `success AND termination` |
