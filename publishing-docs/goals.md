# Goals

Goals are pre-built `SkillTeacher` subclasses that provide standardized reward functions, success criteria, and termination logic. Instead of implementing `compute_reward`, `compute_success_criteria`, and `compute_termination` by hand, you subclass a goal type and supply configuration through constructor arguments.

Every goal type inherits from `Goal`, which itself inherits from `SkillTeacher`. A goal is a fully valid `impl_cls` for a `Skill`.

---

## Base `Goal` constructor parameters

All goal types share these constructor parameters, inherited from the `Goal` base class:

| Parameter              | Type                                    | Default            | Description                                                                                                                                                                                                                    |
| ---------------------- | --------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `sensor`               | `str \| Sensor`                         | **required**       | The sensor whose value this goal tracks. Pass a string name or a `Sensor` object.                                                                                                                                              |
| `name`                 | `str`                                   | `""`               | Optional human-readable label for this goal instance.                                                                                                                                                                          |
| `target`               | `float \| int \| str \| Sensor \| None` | varies by subclass | The reference value. Can be a literal number, the name of another sensor (dynamic target), or a `Sensor` object. If `None`, `boundary_left` and `boundary_right` must both be set, and `target` is computed as their midpoint. |
| `tolerance`            | `float \| int`                          | `0.1`              | The half-width of the success band around `target`. A sensor reading is considered "on target" when `abs(sensor_value - target) <= tolerance`.                                                                                 |
| `stop_value`           | `float \| int \| None`                  | `None`             | Terminate the episode if the sensor reaches this value (direction depends on goal type).                                                                                                                                       |
| `stop_steps`           | `int \| None`                           | `None`             | Terminate the episode after this many steps regardless of sensor values. Counted per goal instance, not per episode.                                                                                                           |
| `boundary_left`        | `float \| int \| None`                  | `None`             | Left (lower) edge of the valid sensor range used for normalization and boundary checks. Required when `target=None`.                                                                                                           |
| `boundary_right`       | `float \| int \| None`                  | `None`             | Right (upper) edge of the valid sensor range. Required when `target=None`.                                                                                                                                                     |
| `boundary_is_relative` | `bool`                                  | `False`            | When `True`, the boundary check is applied to `(sensor_value - target_value)` rather than `sensor_value` directly. Useful when the target is a dynamic sensor value.                                                           |
| `scale`                | `float \| int`                          | `1.0`              | Multiplier applied to the reward. Increase to make this goal's signal stronger relative to other goals in a `CoordinatedGoal`.                                                                                                 |

### Boundary semantics

When `boundary_left` and `boundary_right` are set, the goal uses `is_in_boundary()` to test whether the current sensor reading falls within `[boundary_left, boundary_right]`. Being inside the boundary is what triggers the success bonus in `ApproachGoal` and `MaintainGoal`. Normalization for reward scaling also depends on these values.

When `boundary_is_relative=True`, the boundary is checked against the **error** (`sensor_value - target_value`) rather than the raw sensor value. Use this when your target is itself a sensor that changes over time.

### `target=None` shorthand

When `target=None`, `boundary_left` and `boundary_right` must both be set. The SDK automatically computes `target = (boundary_left + boundary_right) / 2`. Omitting either boundary when `target=None` raises a `ValueError` at construction time.

### `steps_taken` attribute

Each goal instance tracks `self.steps_taken: int`, which increments by 1 on every call to `compute_reward`. This counter is used by the base `Goal.compute_termination` to enforce `stop_steps`. It is **not** reset between episodes — it accumulates across the entire training run. If you need per-episode step counting, reset it manually in `transform_sensors` when you detect a new episode.

### Helper methods available to subclasses

These methods are defined on `Goal` and available to any subclass:

```python
async def is_in_boundary(self, transformed_sensors: dict) -> bool
# Returns True if the sensor value (or error, when boundary_is_relative=True) falls
# within [boundary_left, boundary_right]. Returns False if boundaries are not set.
# Used internally by ApproachGoal and MaintainGoal to gate the success bonus.

async def compute_error(self, transformed_sensors: dict) -> float
# Returns (sensor_value - target_value) ** 2 — the squared error between the sensor
# and the target. Used internally by ApproachGoal, AvoidGoal, and MaintainGoal to
# compute their log-based reward signals. Override in a subclass to use a different
# error metric.

async def get_sensor_value(self, transformed_sensors: dict) -> float
# Returns the current value of self.sensor_name from transformed_sensors.
# If the sensor value is iterable (e.g. an array), sums all elements into a scalar.
# Raises ValueError if the sensor is absent or None.

async def get_target_value(self, transformed_sensors: dict) -> float
# Returns the current target value. If self.target is a string or Sensor, looks up
# the value dynamically from transformed_sensors. Otherwise returns self.target as-is.
```

---

## Goal types

### `ApproachGoal`

Drive a sensor value toward a fixed or dynamic target and succeed once it arrives.

```python
from amesa_core.agent.skill.goals.approach_goal import ApproachGoal

goal = ApproachGoal(
    sensor="position",
    name="reach target position",
    target=5.0,
    tolerance=0.1,
    boundary_left=4.9,
    boundary_right=5.1,
    scale=1.0,
)
```

| Behavior                   | Detail                                                                                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_reward`           | `reward = -log10(error² + ε) / 8`. Reward increases as the sensor approaches the target. A bonus of `+scale` is added when `is_in_boundary()` is `True`. |
| `compute_success_criteria` | Returns `True` when `is_in_boundary()` is `True` (i.e., sensor is within `[boundary_left, boundary_right]`).                                             |
| `compute_termination`      | Always returns `False`. Episode termination is driven by `compute_success_criteria` or `stop_steps`.                                                     |
| `can_terminate()`          | `True` — eligible for use in `THEN` strategies.                                                                                                          |

The target can be a sensor name string or `Sensor` object for a dynamic target that the agent should track in real time:

```python
goal = ApproachGoal(
    sensor="follower_pos",
    target="leader_pos",   # dynamic: follow whatever leader_pos reads
    tolerance=0.2,
    boundary_left=-0.2,
    boundary_right=0.2,
    boundary_is_relative=True,
)
```

---

### `AvoidGoal`

Penalize proximity to a target; reward the agent for staying away.

```python
from amesa_core.agent.skill.goals.avoid_goal import AvoidGoal

goal = AvoidGoal(
    sensor="obstacle_distance",
    name="avoid obstacle",
    target=0.0,
    boundary_left=-0.5,
    boundary_right=0.5,
    scale=1.0,
    should_terminate_in_boundary=True,
)
```

| Parameter (AvoidGoal-specific) | Default | Description                                                                                                                                                          |
| ------------------------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `should_terminate_in_boundary` | `False` | When `True`, the episode terminates immediately if the sensor enters `[boundary_left, boundary_right]`. Use this to model hard failure conditions (e.g., collision). |

| Behavior                   | Detail                                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_reward`           | `reward = (log10(error² + ε) / 8) + 1`. Reward increases the farther the sensor is from the target. A penalty of `-scale` is applied when `is_in_boundary()` is `True`. |
| `compute_success_criteria` | Always returns `False`. Avoidance goals do not produce success termination — only `AvoidGoal` can terminate episodes.                                                   |
| `compute_termination`      | Returns `True` if `is_in_boundary()` is `True` **and** `should_terminate_in_boundary=True`. Otherwise `False`.                                                          |
| `can_terminate()`          | `True`.                                                                                                                                                                 |

In a `CoordinatedGoal`, `AvoidGoal` is the **only** goal type that drives `compute_termination` for the combined goal (see CoordinatedGoal section below).

---

### `MaintainGoal`

Keep a sensor value near a target indefinitely; does not succeed, only accumulates reward over time.

```python
from amesa_core.agent.skill.goals.maintain_goal import MaintainGoal

goal = MaintainGoal(
    sensor="pole_theta",
    name="keep pole upright",
    target=0.0,
    tolerance=0.05,
    boundary_left=-0.418,
    boundary_right=0.418,
    scale=1.0,
)
```

| Behavior                   | Detail                                                                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_reward`           | `reward = -log10(error² + ε) / 8`. A bonus of `+scale` is added each step that `is_in_boundary()` is `True`, continuously rewarding sustained on-target performance. |
| `compute_success_criteria` | Always returns `False`. There is no success condition — the goal never terminates an episode on its own.                                                             |
| `compute_termination`      | Always returns `False`.                                                                                                                                              |
| `can_terminate()`          | `False` — **cannot** be the first goal in a `THEN` strategy.                                                                                                         |

`MaintainGoal` is appropriate when you want the agent to remain in a state as long as possible without ever declaring "done." Pair it with an `AvoidGoal` or `stop_steps` to bound episode length.

---

### `MaximizeGoal`

Push a sensor value as high as possible; no explicit success condition.

```python
from amesa_core.agent.skill.goals.maximize_goal import MaximizeGoal

goal = MaximizeGoal(
    sensor="score",
    name="maximize score",
    scale=1.0,
    stop_steps=1000,
)
```

| Behavior                   | Detail                                                                                                                                                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `compute_reward`           | Monotonically increasing function of `sensor_value`. Uses `tanh`-based shaping: `reward = -tanh(-log10(\|value\| + ε) / 8) + 1.0`. Clipped to a linear interpolation in `[-5, 5]` for numerical stability. Reward is highest at large positive values. |
| `compute_success_criteria` | Always returns `False`.                                                                                                                                                                                                                                |
| `compute_termination`      | Always returns `False`. Use `stop_steps` to bound episodes.                                                                                                                                                                                            |
| `can_terminate()`          | `False` — **cannot** be the first goal in a `THEN` strategy.                                                                                                                                                                                           |

---

### `MinimizeGoal`

Push a sensor value as low as possible; no explicit success condition.

```python
from amesa_core.agent.skill.goals.minimize_goal import MinimizeGoal

goal = MinimizeGoal(
    sensor="energy_use",
    name="minimize energy",
    scale=1.0,
    stop_steps=1000,
)
```

| Behavior                   | Detail                                                                                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_reward`           | Mirror of `MaximizeGoal`: highest reward at small (negative) values. `reward = -tanh(log10(\|value\| + ε) / 8) + 1.0`, with linear interpolation in `[-5, 5]`. |
| `compute_success_criteria` | Always returns `False`.                                                                                                                                        |
| `compute_termination`      | Always returns `False`.                                                                                                                                        |
| `can_terminate()`          | `False` — **cannot** be the first goal in a `THEN` strategy.                                                                                                   |

---

## Using a goal as a `Skill`'s teacher

### Pattern 1 — subclass the goal directly

Subclass a goal type and implement the `SkillTeacher` methods you want to override. The goal's `compute_reward`, `compute_success_criteria`, and `compute_termination` become the skill's implementations.

```python
from amesa_core.agent.skill.goals.maintain_goal import MaintainGoal
from amesa_core.agent.skill.skill import Skill
from typing import Dict, List

class BalanceTeacher(MaintainGoal):

    def __init__(self):
        super().__init__(
            sensor="pole_theta",
            name="maintain pole upright",
            target=0.0,
            tolerance=0.05,
            boundary_left=-0.418,
            boundary_right=0.418,
            scale=1.0,
        )

    async def filtered_sensor_space(self) -> List[str]:
        return ["pole_theta", "pole_alpha", "cart_pos", "cart_vel"]

    async def transform_sensors(self, sensors, action) -> Dict:
        return sensors

    async def transform_action(self, transformed_sensors: Dict, action):
        return action

skill = Skill("balance", BalanceTeacher, training_cycles=100)
```

`filtered_sensor_space`, `transform_sensors`, and `transform_action` must still be implemented (or left to defaults) — the goal only provides the reward/success/termination logic.

### Pattern 2 — subclass `CoordinatedGoal` to combine multiple goals

Pass a list of goal instances to `CoordinatedGoal.__init__` to compose a multi-objective teacher. See the [CoordinatedGoal section](#coordinatedgoal) below.

---

## `CoordinatedGoal`

`CoordinatedGoal` wraps a list of goals and combines their reward, success, and termination signals according to a coordination strategy.

```python
from amesa_core.agent.skill.goals.coordinated_goal import CoordinatedGoal, GoalCoordinationStrategy
from amesa_core.agent.skill.goals.maintain_goal import MaintainGoal
from amesa_core.agent.skill.goals.avoid_goal import AvoidGoal
from amesa_core.agent.skill.skill import Skill
from typing import Dict, List

class CartpoleTeacher(CoordinatedGoal):

    def __init__(self):
        pole_goal = MaintainGoal(
            "pole_theta",
            name="keep pole upright",
            target=0.0,
            boundary_left=-0.418,
            boundary_right=0.418,
            scale=2.0,
        )
        cart_goal = MaintainGoal(
            "cart_pos",
            name="keep cart centered",
            target=0.0,
            boundary_left=-2.4,
            boundary_right=2.4,
            scale=1.0,
        )
        super().__init__(
            goals=[pole_goal, cart_goal],
            goals_coordination_strategy=GoalCoordinationStrategy.AND,
            weights=[2.0, 1.0],
        )

    async def filtered_sensor_space(self) -> List[str]:
        return ["cart_pos", "cart_vel", "pole_theta", "pole_alpha"]

    async def transform_sensors(self, sensors, action) -> Dict:
        return sensors

    async def transform_action(self, transformed_sensors: Dict, action):
        return action

skill = Skill("cartpole", CartpoleTeacher, training_cycles=200)
```

### `CoordinatedGoal` constructor

```python
CoordinatedGoal(
    goals: List[Goal],
    goals_coordination_strategy: int = GoalCoordinationStrategy.AND,
    weights: Optional[List[float | int]] = None,
)
```

| Parameter                     | Type                         | Default                        | Description                                                                                                               |
| ----------------------------- | ---------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `goals`                       | `List[Goal]`                 | **required**                   | The list of goal instances to combine. At least one goal is required.                                                     |
| `goals_coordination_strategy` | `int`                        | `GoalCoordinationStrategy.AND` | How to combine rewards and resolve success/termination. See strategy table below.                                         |
| `weights`                     | `List[float \| int] \| None` | `None`                         | Per-goal reward multipliers. Length must equal `len(goals)`. When `None` or empty, all goals are weighted equally at `1`. |

### `GoalCoordinationStrategy`

| Constant | Value | Reward                                                     | Success                                                                                                                            | Termination                                               |
| -------- | ----- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `AND`    | `1`   | Weighted sum of all goal rewards: `Σ(weight_i × reward_i)` | All `ApproachGoal` instances must return `True` from `compute_success_criteria`. Non-`ApproachGoal` goals are ignored for success. | Terminates if **any** `AvoidGoal` in the list terminates. |
| `OR`     | `2`   | Maximum single-goal reward: `max(reward_i)`                | Same as `AND` — all `ApproachGoal` instances must succeed.                                                                         | Same as `AND` — terminates if any `AvoidGoal` terminates. |
| `THEN`   | `3`   | Reward from the **first unsatisfied goal** in sequence.    | The first goal in the list that returns `True` from `compute_success_criteria` provides the reward; the next goal then takes over. | Same as `AND`.                                            |

#### `THEN` constraint

Goals used as non-final entries in a `THEN` chain **must** be self-terminating (`can_terminate()` returns `True`). `MaximizeGoal`, `MinimizeGoal`, and `MaintainGoal` all return `can_terminate() = False` and will emit a warning if placed in a `THEN` chain. `ApproachGoal` and `AvoidGoal` return `can_terminate() = True` and are safe in any position.

### How `CoordinatedGoal` handles `compute_success_criteria`

`CoordinatedGoal.compute_success_criteria` only returns `True` if **at least one `ApproachGoal` is present in `goals`** and **all `ApproachGoal` instances return `True`**. Goals of other types (Maintain, Maximize, Minimize, Avoid) do not contribute to success evaluation.

### How `CoordinatedGoal` handles `compute_termination`

`CoordinatedGoal.compute_termination` returns `True` if **any `AvoidGoal` in `goals` returns `True`** from its own `compute_termination`. Goals of other types do not contribute to termination from the coordinated wrapper.

---

## `can_terminate()` reference

| Goal type         | `can_terminate()`                     | Notes                           |
| ----------------- | ------------------------------------- | ------------------------------- |
| `ApproachGoal`    | `True`                                | Safe in any position in `THEN`. |
| `AvoidGoal`       | `True`                                | Safe in any position in `THEN`. |
| `MaintainGoal`    | `False`                               | Cannot be first in `THEN`.      |
| `MaximizeGoal`    | `False`                               | Cannot be first in `THEN`.      |
| `MinimizeGoal`    | `False`                               | Cannot be first in `THEN`.      |
| `CoordinatedGoal` | `True` if any child `can_terminate()` | Delegates to children.          |

---

## Job JSON schema

Goals are serialized into the `impl_cls_data.goals` array inside each skill's config block. Each entry has a `type` discriminator field.

### `ApproachGoal`

```json
{
  "type": "ApproachGoal",
  "sensor_name": "position",
  "name": "reach target",
  "target": 5.0,
  "tolerance": 0.1,
  "stop_value": null,
  "stop_steps": null,
  "boundary_left": 4.9,
  "boundary_right": 5.1,
  "boundary_is_relative": false,
  "scale": 1.0
}
```

### `AvoidGoal`

```json
{
  "type": "AvoidGoal",
  "sensor_name": "obstacle_distance",
  "name": "avoid wall",
  "target": 0.0,
  "tolerance": 0.1,
  "stop_value": null,
  "scale": 1.0,
  "boundary_left": -0.5,
  "boundary_right": 0.5,
  "boundary_is_relative": false,
  "should_terminate_in_boundary": true
}
```

### `MaintainGoal`

```json
{
  "type": "MaintainGoal",
  "sensor_name": "pole_theta",
  "name": "keep upright",
  "target": 0.0,
  "tolerance": 0.05,
  "stop_value": null,
  "stop_steps": null,
  "boundary_left": -0.418,
  "boundary_right": 0.418,
  "boundary_is_relative": false,
  "scale": 1.0
}
```

### `MaximizeGoal` / `MinimizeGoal`

```json
{
  "type": "MaximizeGoal",
  "sensor_name": "score",
  "name": "maximize score",
  "stop_value": null,
  "stop_steps": 1000,
  "tolerance": 0.1,
  "boundary_left": null,
  "boundary_right": null,
  "scale": 1.0
}
```

### `CoordinatedGoal`

```json
{
  "type": "CoordinatedGoal",
  "goals_coordination_strategy": 1,
  "weights": [2.0, 1.0],
  "goals": [
    { "type": "MaintainGoal", "sensor_name": "pole_theta", ... },
    { "type": "MaintainGoal", "sensor_name": "cart_pos", ... }
  ]
}
```

`goals_coordination_strategy` is an integer: `1` = AND, `2` = OR, `3` = THEN.

---

## ⚠️ Quirks

**`boundary_left` and `boundary_right` are required for success bonuses** — `ApproachGoal` and `MaintainGoal` only award their in-boundary bonus if both boundaries are set. If you omit them, `is_in_boundary()` always returns `False`, and the success bonus is never awarded. Goals will still train (via the log-error reward) but will never trigger `compute_success_criteria`.

**`MaintainGoal`, `MaximizeGoal`, and `MinimizeGoal` never succeed** — Their `compute_success_criteria` always returns `False`. If used alone as a skill's `impl_cls`, the episode will only end via `compute_termination` (which also returns `False` for all three) or via the trainer's `stop_steps` limit. Always set `stop_steps` or pair with an `AvoidGoal` in a `CoordinatedGoal` to guarantee episode termination.

**`AvoidGoal.compute_success_criteria` always returns `False`** — `AvoidGoal` contributes reward but never produces a success signal. The only output it can produce to end an episode is via `compute_termination` (when `should_terminate_in_boundary=True`).

**`weights` must match `len(goals)`** — If `weights` is provided and its length differs from `len(goals)`, `CoordinatedGoal.__init__` raises a `ValueError`.

**Sensor values are summed when iterable** — If the sensor returns an array or list (e.g., a multi-dimensional observation), `get_sensor_value()` sums all elements into a scalar before computing the reward. Design your sensor space so goal sensors return scalar values.

**`compute_reward` must return a Python `float`** — Values in `transformed_sensors` are numpy scalars. Arithmetic on them produces numpy types, which RLlib rejects with a `ValueError`. Always wrap the return value explicitly:

```python
# In a custom compute_reward override on a goal subclass:
async def compute_reward(self, transformed_sensors, action, sim_reward):
    reward = await super().compute_reward(transformed_sensors, action, sim_reward)
    return float(reward)
```
