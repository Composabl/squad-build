# Teacher Schema — `SkillTeacher`

A `SkillTeacher` defines the learning contract for one RL skill. It provides the reward
signal, success/failure criteria, sensor filtering, and action/sensor transforms. Every
RL skill requires a teacher.

## Import

```python
from amesa_core import SkillTeacher
# or explicit:
from amesa_core.agent.skill.skill_teacher import SkillTeacher
```

## Interface Contract

All required methods are `async`. The class is abstract — you must implement all
four required methods.

```python
class SkillTeacher(ABC):

    def __init__(self, *args, **kwargs):
        """
        MUST support zero-argument construction.
        Called with no args at startup AND on every episode reset.
        Do not initialize resources that depend on constructor arguments.
        """
        self.action_space = None    # never set by framework — do not read
        self.sensor_space = None    # never set by framework — do not read
        self.scenario = None        # set by add_scenario() before training

    # ── Required ────────────────────────────────────────────────────────

    async def compute_reward(
        self,
        transformed_sensors: dict,  # filtered + normalized sensor dict from transform_sensors()
        action,                     # action sent to sim on previous step; None on step 0
        sim_reward: float           # raw reward returned by the simulator
    ) -> float:
        """
        Return the reward passed to the RL algorithm.
        Must return a scalar float. Returning None or non-numeric crashes training.
        """

    async def compute_success_criteria(
        self,
        transformed_sensors: dict,
        action                      # previous step's action
    ) -> bool:
        """
        Return True to end the episode as a SUCCESS.
        Increments success_counter, which drives curriculum progression.
        If compute_termination() also returns True on the same step,
        termination takes precedence and success_counter is DECREMENTED.
        """

    async def transform_action(
        self,
        transformed_sensors: dict,
        action                      # raw action from the RL model (already unsquashed)
    ):
        """
        Post-process the RL model's action before it is sent to the simulator.
        Use for: clipping to physical limits, unit conversion, remapping dimensions.
        Must return the final action value.
        """

    async def filtered_sensor_space(self) -> list[str]:
        """
        Return list of sensor names this skill's RL policy observes.
        Names must match keys registered via agent.add_sensors() or perceptor outputs.
        Called ONCE at startup; result is cached for the skill's lifetime.
        Returning None or [] causes the framework to use ALL agent sensors.
        """

    # ── Optional / Hooks ────────────────────────────────────────────────

    async def compute_termination(
        self,
        transformed_sensors: dict,
        action
    ) -> bool:
        """
        Return True to end the episode as a FAILURE.
        Decrements success_counter (floor 0).
        Default: return False
        """
        return False

    async def compute_action_mask(
        self,
        transformed_sensors: dict,
        action
    ) -> list[bool] | None:
        """
        Return list of booleans masking discrete actions (True = allowed).
        Return None to disable masking (default).
        AND-combined with any sim-provided mask.
        Only effective for discrete action spaces.
        """
        return None

    async def transform_sensors(
        self,
        sensors: dict,  # full amesa_sensors dict after perceptor pipeline
        action          # ALWAYS None in all current call paths — do not use
    ) -> dict:
        """
        Pre-process observations before filtered_sensor_space() filtering.
        Use for: feature engineering, derived signals, unit conversion.
        Default: return sensors unchanged.
        """
        return sensors

    def add_scenario(self, scenario):
        """Called by framework before training. Override to react to scenario changes."""
        self.scenario = scenario
```

## Required Methods Summary

| Method | Returns | Called When | Default |
|---|---|---|---|
| `compute_reward` | `float` | Every step, after sim returns | — (required) |
| `compute_success_criteria` | `bool` | Every step | — (required) |
| `transform_action` | action | After RL model inference | — (required) |
| `filtered_sensor_space` | `list[str]` | Once at init | — (required) |
| `compute_termination` | `bool` | Every step | `return False` |
| `compute_action_mask` | `list[bool] \| None` | Every step | `return None` |
| `transform_sensors` | `dict` | Every step, before filtering | `return sensors` |

## Registration

Pass the **class** (not an instance) to `Skill()`. The framework detects
`issubclass(impl_cls, SkillTeacher)` automatically.

```python
from amesa_core import Agent, Skill, Sensor, Scenario

class MyTeacher(SkillTeacher):
    ...

skill = Skill(
    "my-skill",
    MyTeacher,                # class, not instance
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

## `Skill()` Kwargs for Teacher Skills

| Kwarg | Effect | Default |
|---|---|---|
| `train_batch_size` | PPO batch size | `4000` |
| `training_cycles` | Stop after N cycles (None = run until stopped) | `None` |
| `fc_layers` | Hidden layer sizes, e.g. `[256, 256]` | `[256, 256]` |
| `workers` | Number of Ray rollout workers | `1` |
| `learner_workers` | Learner worker count | `0` |
| `envs_per_worker` | Environments per worker | `1` |
| `num_cpus_per_worker` | CPUs per worker | `1.0` |
| `num_gpus_per_worker` | GPUs per worker | `0.0` |
| `custom_action_space` | Override action space | `None` |

## Minimal Working Example

```python
from amesa_core import SkillTeacher

class SetpointTeacher(SkillTeacher):
    TARGET = 80.0
    TOLERANCE = 2.0
    FAIL_DISTANCE = 20.0

    def __init__(self):
        # Zero-argument constructor — also called on every episode reset
        self.steps = 0

    async def compute_reward(self, transformed_sensors, action, sim_reward):
        error = abs(transformed_sensors["temperature"] - self.TARGET)
        return -error

    async def compute_success_criteria(self, transformed_sensors, action):
        return abs(transformed_sensors["temperature"] - self.TARGET) <= self.TOLERANCE

    async def compute_termination(self, transformed_sensors, action):
        return abs(transformed_sensors["temperature"] - self.TARGET) >= self.FAIL_DISTANCE

    async def transform_action(self, transformed_sensors, action):
        return max(0.0, min(1.0, action))   # clamp to [0, 1]

    async def filtered_sensor_space(self):
        return ["temperature"]

    async def transform_sensors(self, sensors, action):
        return sensors  # pass through
```

## Critical Behavioral Rules

| Rule | Detail |
|---|---|
| Re-instantiated every episode | `__init__()` with no args is called at each episode reset. All `self.*` state is wiped. |
| `transform_sensors` action is always None | The call site always passes `None` as the second arg. Never write logic depending on it. |
| `action_space` and `sensor_space` are never set | Both remain `None` throughout. Do not call `self.action_space.sample()`. |
| `filtered_sensor_space` called once | Return value is cached at init — conditional returns per step have no effect. |
| Termination beats success | If both `compute_termination` and `compute_success_criteria` return True on the same step, the episode ends as failure (counter decremented). |
| `is_compute_done` not called during training | Only invoked from `env_validator.py`. Overriding it has no training effect. |
| SkillTeacher also powers SkillSelectors | The same Teacher class can serve as routing logic inside a `SkillSelector`. |
