# Skills — Overview

A **Skill** is the fundamental unit of an agent. Each skill encapsulates one learned or rule-based behavior.

## Skill types

| Type            | Class                                                | Use                                                     | v2         |
| --------------- | ---------------------------------------------------- | ------------------------------------------------------- | ---------- |
| **Skill**       | `Skill`                                              | A single ability — one teacher or controller            | ✅         |
| **Selector**    | `SkillSelector`                                      | Chooses which child skill to run each step              | ✅         |
| **Group**       | `SkillGroup`                                         | Pipelines two skills: output of first → input of second | ✅         |
| **Coordinated** | `SkillCoordinatedSet` / `SkillCoordinatedPopulation` | Multi-agent: a coach directs a set of skills            | ❌ not yet |

## Implementation types

Each `Skill` or `SkillSelector` is backed by exactly one implementation class:

| Implementation | Base class        | Description                                                           |
| -------------- | ----------------- | --------------------------------------------------------------------- |
| **Teacher**    | `SkillTeacher`    | Machine-learned via RL — you define reward, success, action transform |
| **Controller** | `SkillController` | Deterministic / rule-based — you compute the action directly          |
| **Coach**      | `SkillCoach`      | Used with `SkillCoordinated` — coordinates multiple sub-skills        |

The SDK infers the implementation type from the class you pass:

```python
Skill("my-skill", MyTeacher)      # → TEACHER type
Skill("my-skill", MyController)   # → CONTROLLER type
```

## Instantiating skills

```python
from amesa_core.agent.skill.skill import Skill

skill = Skill(
    "skill-name",       # unique name string
    MyTeacher,          # class (not instance)
    training_cycles=50, # optional kwarg
)
skill.add_scenario(my_scenario)
```

### Common kwargs

| kwarg                 | Applies to       | Effect                                  |
| --------------------- | ---------------- | --------------------------------------- |
| `training_cycles`     | Teacher/Selector | Override training cycles for this skill |
| `train_batch_size`    | Teacher/Selector | Override PPO batch size                 |
| `fc_layers`           | Teacher/Selector | Override fully-connected layer sizes    |
| `workers`             | Teacher/Selector | Number of rollout workers               |
| `envs_per_worker`     | Teacher/Selector | Envs per rollout worker                 |
| `custom_action_space` | Teacher/Selector | Override inferred action space          |

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
