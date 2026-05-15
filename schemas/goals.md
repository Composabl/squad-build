# Goals Schema — Declarative Teacher Objectives

Goals are pre-built `SkillTeacher` subclasses for standard RL objectives. Use a Goal
instead of writing `compute_reward`, `compute_success_criteria`, and
`compute_termination` from scratch when your objective fits one of the five patterns.

## Import

```python
from amesa_core.agent.skill.goals.maintain_goal import MaintainGoal
from amesa_core.agent.skill.goals.maximize_goal import MaximizeGoal
from amesa_core.agent.skill.goals.minimize_goal import MinimizeGoal
from amesa_core.agent.skill.goals.approach_goal import ApproachGoal
from amesa_core.agent.skill.goals.avoid_goal import AvoidGoal
from amesa_core.agent.skill.goals.coordinated_goal import CoordinatedGoal
```

## Five Goal Types

| Goal | Objective | Reward | Success | Termination |
|---|---|---|---|---|
| `MaintainGoal` | Keep sensor near `target` | `-abs(sensor - target)` | `abs(error) <= stop_distance` | `abs(error) > stop_distance * 2` (approximate) |
| `MaximizeGoal` | Drive sensor as high as possible | `+sensor_value` | Configurable | Configurable |
| `MinimizeGoal` | Drive sensor as low as possible | `-sensor_value` | Configurable | Configurable |
| `ApproachGoal` | Move sensor toward `target` | Decreasing distance | Within `stop_distance` | Configurable |
| `AvoidGoal` | Stay away from `target` | Positive when distant | Distance >= `stop_distance` | Distance < `stop_distance` |

## Constructor Parameters (all Goal types)

```python
GoalType(
    sensor_variable: str,   # name of the sensor variable to apply the goal to
    description: str,       # human-readable description
    target: float = None,   # target value (Maintain, Approach)
    stop_distance: float = None,  # distance at which success/failure triggers
)
```

## Usage as a Teacher

Subclass the goal and implement only what you need to override:

```python
from amesa_core.agent.skill.goals.maintain_goal import MaintainGoal

class BalanceTeacher(MaintainGoal):
    def __init__(self):
        super().__init__(
            "pole_theta",                  # sensor variable name
            "Maintain pole to upright",    # description
            target=0.0,                    # target value
            stop_distance=0.418            # radians ≈ 24 degrees
        )
        # You can still override compute_reward, compute_termination, etc.

    async def transform_action(self, transformed_sensors, action):
        return action   # required — inherited goals don't provide this

    async def filtered_sensor_space(self):
        return ["pole_theta", "cart_position", "cart_velocity"]
```

## `CoordinatedGoal` — Combining Multiple Goals in One Teacher

`CoordinatedGoal` (unrelated to `SkillCoordinatedSet`) is a single-skill reward
combiner. It merges multiple Goal objectives using AND, OR, or THEN logic.

```python
from amesa_core.agent.skill.goals.coordinated_goal import CoordinatedGoal, GoalCoordinationStrategy

class MultiObjectiveTeacher(CoordinatedGoal):
    def __init__(self):
        temp_goal     = MaintainGoal("temperature", "Hold temperature", target=80.0, stop_distance=2.0)
        pressure_goal = MaintainGoal("pressure",    "Hold pressure",    target=5.0,  stop_distance=0.5)
        super().__init__(
            goals=[temp_goal, pressure_goal],
            goals_coordination_strategy=GoalCoordinationStrategy.AND,
            weights=[0.7, 0.3],   # relative reward weighting
        )

    async def transform_action(self, transformed_sensors, action):
        return action

    async def filtered_sensor_space(self):
        return ["temperature", "pressure"]
```

## Coordination Strategies

| Strategy | Behavior |
|---|---|
| `GoalCoordinationStrategy.AND` | Weighted sum of all goal rewards |
| `GoalCoordinationStrategy.OR` | Max of all goal rewards |
| `GoalCoordinationStrategy.THEN` | Reward from first incomplete goal (sequential completion) |

## Registration

Register exactly like any other `SkillTeacher` — the Goal class is already a Teacher:

```python
from amesa_core import Skill

skill = Skill("balance", BalanceTeacher, training_cycles=500)
skill.add_scenario(Scenario({"pole_theta": {"data": 0.1, "type": "set_value"}}))
agent.add_skill(skill)
```

## When to Use Goals vs. Teachers

| Use Goals when | Use Teacher when |
|---|---|
| Objective is maintain/maximize/minimize/approach/avoid | Need non-standard reward shaping |
| Success = being within a distance of a target | Success depends on multiple sensors in a custom way |
| You want less boilerplate | You need full control over every aspect of reward |

## Important: `CoordinatedGoal` ≠ `SkillCoordinatedSet`

Despite the shared prefix, these are completely separate concepts:

- `CoordinatedGoal` — combines multiple Goals **inside one skill's teacher**
- `SkillCoordinatedSet` — runs multiple skills **in parallel** as a multi-agent system

Do not confuse them. See [coordinated-skills.md](coordinated-skills.md) for
multi-agent coordination.
