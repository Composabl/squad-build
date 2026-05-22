# Goals

v1 includes a goal system that provides higher-level building blocks for defining skill objectives. Goals are an alternative to implementing `SkillTeacher` reward methods manually — they express *what* a skill should achieve, rather than *how* to compute reward step-by-step.

## Import

```python
from amesa_core import (
    ApproachGoal,
    MaximizeGoal,
    MinimizeGoal,
    AvoidGoal,
    MaintainGoal,
    CoordinatedGoal,
)
```

---

## Goal classes

### `ApproachGoal`

Drive a sensor value toward a target.

```python
ApproachGoal(
    sensor: str,           # sensor name to observe
    target: float,         # desired value to approach
    tolerance: float = 0.1,  # success band around target
)
```

**Behavior** — Reward increases as the sensor approaches `target`. The skill is considered successful when `abs(sensor - target) <= tolerance`.

```python
goal = ApproachGoal("air_temp", target=22.0, tolerance=0.5)
```

---

### `MaximizeGoal`

Drive a sensor value as high as possible.

```python
MaximizeGoal(
    sensor: str,            # sensor name to maximize
    upper_bound: float = 1.0,   # optional clipping bound for normalization
)
```

**Behavior** — Reward is proportional to the sensor value. Use when you want the agent to increase a metric without a specific target.

```python
goal = MaximizeGoal("efficiency", upper_bound=100.0)
```

---

### `MinimizeGoal`

Drive a sensor value as low as possible.

```python
MinimizeGoal(
    sensor: str,
    lower_bound: float = 0.0,
)
```

**Behavior** — Reward increases as the sensor approaches `lower_bound`. Use for minimizing error, energy, or cost signals.

```python
goal = MinimizeGoal("temp_error", lower_bound=0.0)
```

---

### `AvoidGoal`

Penalize the agent for entering a sensor value range.

```python
AvoidGoal(
    sensor: str,
    low: float,
    high: float,
    penalty: float = -1.0,
)
```

**Behavior** — When `low <= sensor <= high`, the agent receives `penalty` each step. Use to encode safety constraints or forbidden zones.

```python
goal = AvoidGoal("air_temp", low=0.0, high=5.0, penalty=-10.0)
```

---

### `MaintainGoal`

Keep a sensor value within a range.

```python
MaintainGoal(
    sensor: str,
    low: float,
    high: float,
    tolerance: float = 0.0,
)
```

**Behavior** — Reward is positive while `low <= sensor <= high`. Penalizes deviation outside the range. Similar to `AvoidGoal` but framed as a positive maintenance objective.

```python
goal = MaintainGoal("humidity", low=0.4, high=0.6)
```

---

### `CoordinatedGoal`

Combine multiple goals with a logical strategy (AND, OR, sequential).

```python
CoordinatedGoal(
    goals: list[Goal],
    strategy: GoalCoordinationStrategy,
)
```

**Behavior** — Aggregates rewards from the sub-goals according to `strategy`. Use to express compound objectives.

```python
from amesa_core import CoordinatedGoal, GoalCoordinationStrategy

goal = CoordinatedGoal(
    goals=[
        ApproachGoal("air_temp", target=22.0, tolerance=0.5),
        MaintainGoal("humidity", low=0.4, high=0.6),
    ],
    strategy=GoalCoordinationStrategy.AND,
)
```

---

## `GoalCoordinationStrategy`

```python
from amesa_core import GoalCoordinationStrategy

GoalCoordinationStrategy.AND    # all sub-goals must be satisfied for success
GoalCoordinationStrategy.OR     # any one sub-goal being satisfied counts as success
GoalCoordinationStrategy.THEN   # sub-goals must be satisfied in sequence
```

| Strategy | Reward aggregation | Success condition |
|---|---|---|
| `AND` | Sum of sub-goal rewards | All sub-goals successful simultaneously |
| `OR` | Max of sub-goal rewards | At least one sub-goal successful |
| `THEN` | Active sub-goal reward | Sub-goals completed in order |

---

## How goals relate to `SkillTeacher`

Goals are a higher-level alternative to implementing `SkillTeacher` methods manually. When you attach goals to a skill, the SDK generates the reward function, success criteria, and termination logic for you.

```python
from amesa_core.agent.skill.skill import Skill
from amesa_core import ApproachGoal, MaintainGoal, CoordinatedGoal, GoalCoordinationStrategy

skill = Skill(
    "greenhouse-climate",
    GreenhouseTeacher,
    training_cycles=100,
)
skill.set_goal(
    CoordinatedGoal(
        goals=[
            ApproachGoal("air_temp", target=22.0, tolerance=0.5),
            MaintainGoal("humidity", low=0.4, high=0.6),
        ],
        strategy=GoalCoordinationStrategy.AND,
    )
)
```

You can still override specific methods in `SkillTeacher` when goals don't capture all the nuance you need. The goal-generated reward is used as a baseline; if you implement `compute_reward` yourself, it takes precedence.

---

## Full example

```python
from amesa_core.agent.agent import Agent
from amesa_core.agent.sensors.sensor import Sensor
from amesa_core.agent.skill.skill import Skill
from amesa_core.agent.skill.skill_teacher import SkillTeacher
from amesa_core import (
    ApproachGoal,
    MaintainGoal,
    AvoidGoal,
    CoordinatedGoal,
    GoalCoordinationStrategy,
)
from amesa_train import Trainer


class ClimateTeacher(SkillTeacher):
    """Minimal teacher — goal system handles reward and success."""

    async def filtered_sensor_space(self):
        return ["air_temp", "target_temp", "humidity"]

    async def transform_action(self, transformed_sensors, action):
        return action

    async def compute_reward(self, transformed_sensors, action, sim_reward):
        # Delegating to goal system — return sim_reward as passthrough
        return sim_reward

    async def compute_success_criteria(self, transformed_sensors, action):
        temp_ok = abs(transformed_sensors.get("air_temp", 0.0) - 22.0) < 0.5
        humidity_ok = 0.4 <= transformed_sensors.get("humidity", 0.0) <= 0.6
        return temp_ok and humidity_ok


agent = Agent()
agent.add_sensors([
    Sensor("air_temp"),
    Sensor("humidity"),
    Sensor("target_temp"),
])

skill = Skill(
    "climate",
    ClimateTeacher,
    training_cycles=50,
    train_batch_size=2000,
)
skill.set_goal(
    CoordinatedGoal(
        goals=[
            ApproachGoal("air_temp", target=22.0, tolerance=0.5),
            MaintainGoal("humidity", low=0.4, high=0.6),
            AvoidGoal("air_temp", low=0.0, high=5.0, penalty=-5.0),
        ],
        strategy=GoalCoordinationStrategy.AND,
    )
)
agent.add_skill(skill)

config = {"target": {"local": {"address": "localhost:1337"}}}
trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=10)
finally:
    trainer.close()
```

---

## ⚠️ Quirks

**Goals do not replace `filtered_sensor_space`** — You must still implement `filtered_sensor_space()` in your teacher and include all sensor names that the goal references.

**`set_goal` vs manual teacher methods** — If you call `skill.set_goal(...)` and also implement `compute_reward` in your teacher, the teacher method takes precedence. Goals are most useful when you want to skip the manual reward engineering entirely.

**`CoordinatedGoal` nesting** — `CoordinatedGoal` can itself contain other `CoordinatedGoal` instances, allowing tree-structured objective hierarchies.
