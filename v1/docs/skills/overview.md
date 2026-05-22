# Skills — Overview

A **Skill** is the fundamental unit of an agent. Each skill encapsulates one learned or rule-based behavior.

## Skill types

| Type | Class | Use | v1 |
|---|---|---|---|
| **Skill** | `Skill` | A single ability — one teacher or controller | ✅ |
| **Selector** | `SkillSelector` | Chooses which child skill to run each step | ✅ |
| **Group** | `SkillGroup` | Pipelines two skills: output of first → input of second | ✅ |
| **Coordinated** | `SkillCoordinatedSet` / `SkillCoordinatedPopulation` | Multi-agent: a coach directs a set of skills | ✅ |

> **Goal system** — v1 includes `ApproachGoal`, `MaximizeGoal`, `MinimizeGoal`, `AvoidGoal`, `MaintainGoal`, and `CoordinatedGoal` classes as a higher-level way to express skill objectives without implementing teacher reward methods from scratch. See [../goals.md](../goals.md).

## Implementation types

Each `Skill` or `SkillSelector` is backed by exactly one implementation class:

| Implementation | Base class | Description |
|---|---|---|
| **Teacher** | `SkillTeacher` | Machine-learned via RL — you define reward, success, action transform |
| **Controller** | `SkillController` | Deterministic / rule-based — you compute the action directly |
| **Coach** | `SkillCoach` | Used with `SkillCoordinated` — coordinates multiple sub-skills |

The SDK infers the implementation type from the class you pass:

```python
Skill("my-skill", MyTeacher)      # → TEACHER type
Skill("my-skill", MyController)   # → CONTROLLER type
```

## Instantiating skills

```python
from amesa_core.agent.skill.skill import Skill

skill = Skill(
    "skill-name",           # unique name string
    MyTeacher,              # class (not instance)
    training_cycles=100,    # PPO update iterations for this skill
    train_batch_size=4000,  # samples per PPO batch
    workers=1,              # rollout workers
    envs_per_worker=1,      # envs per worker
)
skill.add_scenario(my_scenario)
```

### Common kwargs

| kwarg | Applies to | Effect |
|---|---|---|
| `training_cycles` | Teacher/Selector | Number of PPO update iterations per training cycle |
| `train_batch_size` | Teacher/Selector | Samples collected per PPO batch |
| `fc_layers` | Teacher/Selector | Override fully-connected layer sizes |
| `workers` | Teacher/Selector | Number of Ray rollout workers |
| `envs_per_worker` | Teacher/Selector | Envs per rollout worker |
| `custom_action_space` | Teacher/Selector | Override inferred action space |

> **v1 note** — In v1, training hyperparameters (`training_cycles`, `train_batch_size`, `workers`, `envs_per_worker`) are set as kwargs on `Skill(...)`, not in the trainer config.

## Adding scenarios

```python
skill.add_scenario(scenario)            # single Scenario object
skill.add_scenario({"key": 1.0})        # raw dict (auto-converted)
skill.add_scenario([scenario_a, scenario_b])  # list
```

## See also

- [teacher.md](teacher.md) — `SkillTeacher` interface
- [controller.md](controller.md) — `SkillController` interface
- [selector.md](selector.md) — `SkillSelector` usage
- [group.md](group.md) — `SkillGroup` usage
- [coordinated.md](coordinated.md) — `SkillCoach` / `SkillCoordinated` usage
- [../goals.md](../goals.md) — goal system
