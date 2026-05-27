# SkillCoach and SkillCoordinated

`SkillCoach` + `SkillCoordinated` enable multi-agent coordination: a coach directs a set of sub-skills, each receiving its own reward signal.

> **Note:** Coordinated skills are an advanced feature.

## SkillCoach

The coach equivalent of `SkillTeacher` — but it distributes rewards and coordinates actions across multiple sub-skills.

### Required methods

```python
async def compute_reward(self, transformed_sensors: dict, action, sim_reward) -> dict
# Returns a dict mapping sub-skill names → individual float rewards.
# Keys must exactly match the names of child skills added via add_skill().
# Each child's PPO policy is updated using its own reward independently.

async def compute_success_criteria(self, transformed_sensors: dict, action) -> bool
# Returns True when the coordinated set has achieved its overall objective.
# Ends the episode for all sub-skills simultaneously.

async def compute_termination(self, transformed_sensors: dict, action) -> bool
# Returns True to abort the episode early (e.g. unsafe state).
# Terminates all sub-skills simultaneously.
```

### Method parameter types

| Parameter | Type | Description |
|---|---|---|
| `transformed_sensors` | `dict[str, Any]` | Sensor readings after `transform_sensors` has run. Shared across all sub-skills — the coach sees the full observation. |
| `action` | `dict[str, Any]` | A dict mapping each sub-skill name to that sub-skill's action output. Access a specific sub-skill's action with `action["sub-skill-name"]`. |
| `sim_reward` | `float` | Scalar reward from the simulator. May be used, ignored, or distributed among sub-skills as desired. |

### Optional methods

```python
async def transform_sensors(self, sensors, action) -> dict
# Default: pass-through. Pre-process the observation before it reaches compute_reward
# and compute_success_criteria. sensors is the raw observation dict from the sim.
# action is the dict of sub-skill actions from the previous step.

async def transform_action(self, transformed_sensors, action)
# Default: pass-through. Post-process sub-skill actions before they are sent to the sim.
# action is a dict mapping sub-skill names to raw policy outputs.

async def compute_action_mask(self, transformed_sensors, action)
# Default: None (no masking). Return a per-sub-skill mask dict to restrict the action
# space of individual sub-skills. Keys are sub-skill names; values follow the same
# mask shape conventions as SkillTeacher.compute_action_mask.
```

### Example

```python
from amesa_core.agent.skill.skill_coach import SkillCoach
from typing import Dict

class MyCoach(SkillCoach):

    async def compute_reward(self, transformed_sensors: Dict, action, sim_reward):
        # Return per-sub-skill rewards
        return {
            "striker": transformed_sensors.get("goal_proximity", 0.0),
            "defender": -transformed_sensors.get("opponent_proximity", 0.0),
        }

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        return transformed_sensors.get("score", 0) > 5

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        return transformed_sensors.get("time_elapsed", 0) > 300
```

## SkillCoordinatedSet

Wraps a `SkillCoach` and a **fixed, named set** of child skills into a coordinated unit. Each child trains with its own reward as distributed by the coach. All child skills run simultaneously each step.

```python
from amesa_core.agent.skill.skill_coordinated import SkillCoordinatedSet
from amesa_core.agent.skill.skill import Skill

coordinated = SkillCoordinatedSet("team-skill", MyCoach)
coordinated.add_skill(Skill("striker", StrikerTeacher, training_cycles=50))
coordinated.add_skill(Skill("defender", DefenderTeacher, training_cycles=50))

agent.add_coordinated_skill(coordinated)
```

`SkillCoordinatedSet` constructor:

```python
SkillCoordinatedSet(
    name: str,                           # unique name for this coordinated unit
    impl_cls: type[SkillCoach],          # the coach class (not an instance)
    config: dict | SkillSchema = {},     # optional; same kwargs accepted as Skill(...)
    **kwargs,                            # training_cycles, train_batch_size, workers, etc.
)
```

### Managing child skills

```python
coordinated.add_skill(skill: Skill) -> None
# Appends a child Skill to this coordinated unit. The skill's name must exactly match
# a key returned by the coach's compute_reward dict. Call once per child skill before
# registering the coordinated unit with agent.add_coordinated_skill().

coordinated.get_skills() -> List[Skill]
# Returns the list of child Skill objects currently registered.

coordinated.get_skill_names() -> List[str]
# Returns the list of child skill name strings in registration order.
```

The child skill names passed to `add_skill(Skill("name", ...))` must exactly match the keys returned by `compute_reward` in the coach.

## SkillCoordinatedPopulation

Like `SkillCoordinatedSet`, but the coach manages a **population** of dynamically-scaled sub-skill instances. All instances share the same policy and are trained as a population. Useful for variable-size teams or population-based training where the number of agents may change between episodes.

```python
from amesa_core.agent.skill.skill_coordinated import SkillCoordinatedPopulation

coordinated = SkillCoordinatedPopulation("population-skill", MyCoach)
coordinated.add_skill(Skill("agent-template", AgentTeacher, training_cycles=50))

agent.add_coordinated_skill(coordinated)
```

In `SkillCoordinatedPopulation`, a single child skill acts as a **template**. The population size is determined at runtime by the environment. The coach's `compute_reward` should return rewards keyed to each active instance.

## Full coordinated training example

```python
from amesa_core import (
    Agent, Sensor, Skill, SkillCoach, SkillCoordinatedSet,
)
from amesa_core.agent.sensors.sensor import Sensor
from amesa_train import Trainer

class TeamCoach(SkillCoach):
    async def compute_reward(self, transformed_sensors, action, sim_reward):
        return {
            "attacker": transformed_sensors.get("attack_score", 0.0),
            "defender": -transformed_sensors.get("threat_level", 0.0),
        }

    async def compute_success_criteria(self, transformed_sensors, action) -> bool:
        return transformed_sensors.get("match_won", False)

    async def compute_termination(self, transformed_sensors, action) -> bool:
        return transformed_sensors.get("match_over", False)

agent = Agent()
agent.add_sensors([Sensor("attack_score"), Sensor("threat_level"), Sensor("match_won"), Sensor("match_over")])

coordinated = SkillCoordinatedSet("team", TeamCoach)
coordinated.add_skill(Skill("attacker", AttackerTeacher, training_cycles=50))
coordinated.add_skill(Skill("defender", DefenderTeacher, training_cycles=50))
agent.add_coordinated_skill(coordinated)

config = {"target": {"local": {"address": "localhost:1337"}}}
trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=10)
finally:
    trainer.close()
```

---

## Uploading as a portable component

To upload a coordinated skill (set or population) as a standalone portable component, include a `pyproject.toml` alongside your coach implementation:

```toml
[project]
name = "my-coordinated-skill"
version = "0.1.0"
description = "A coordinated skill coach."
authors = [{ name = "Your Name", email = "you@example.com" }]
dependencies = [
    "amesa-core",
]

[amesa]
type = "skill-coordinated-set"
entrypoint = "my_module.coach:MyCoach"
```

- `type` must be `"skill-coordinated-set"` or `"skill-coordinated-population"`
- `entrypoint` is `"module.path:ClassName"` pointing to the `SkillCoach` subclass

Publish with:

```bash
amesa_cli skill publish <path-to-directory>
```

The CLI validates the `pyproject.toml`, packages the directory as a `.tar.gz` archive, creates a skill record, and uploads the implementation.

---

## Job JSON schema

Each entry in `skills_coordinated[]` in the agent JSON follows `SkillCoordinatedSchema`. The `type` is either `"SkillCoordinatedSet"` or `"SkillCoordinatedPopulation"`.

```json
{
  "name": "team-skill",
  "type": "SkillCoordinatedSet",
  "config": {
    "remote_address": null,
    "impl_cls": {
      "cls_name": "MyCoach",
      "cls_module": "my_agent.coach",
      "cls_src": "<base64-pickle>",
      "cls_deps": []
    },
    "impl_cls_data": { "guidance": null, "goals": [], "constraints": null },
    "learning": {
      "training_cycles": null,
      "train_batch_size": 4000,
      "replay_buffer_size": 50000,
      "rl_algo": "PPO"
    },
    "resources": {
      "workers": 1,
      "learner_workers": 0,
      "envs_per_worker": 1
    },
    "model": {
      "checkpoint_uri": "/tmp/amesa",
      "fc_layers": [256, 256]
    },
    "model_io": {},
    "scenarios": [],
    "scenarios_current_idx": 0,
    "skills": []
  }
}
```

| Field                | Default  | Description                                                            |
| -------------------- | -------- | ---------------------------------------------------------------------- |
| `remote_address`     | `null`   | URL of a remotely-hosted coach; when set, `impl_cls` is ignored        |
| `impl_cls`           | —        | Serialized coach class (produced by `Agent.export()`)                  |
| `impl_cls_data`      | —        | Guidance, goals, and constraints attached to the coach                 |
| `learning`           | —        | PPO hyperparameters (same fields as `SkillTeacher`; see training_job.md) |
| `resources`          | —        | Worker/compute allocation (same fields as `SkillTeacher`)              |
| `model`              | —        | Checkpoint and network architecture config                             |
| `model_io`           | —        | Sensor and action space definitions                                    |
| `scenarios`          | `[]`     | Scenarios to cycle through during training                             |
| `scenarios_current_idx` | `0`   | Index of the currently active scenario                                 |
| `skills`             | `[]`     | Serialized child skills managed by this coordinated set                |

---

## ⚠️ Quirks

**`compute_reward` returns a dict** — Keys must match the names of the child skills exactly.

**Coach is not a Teacher** — Passing a `SkillCoach` to a plain `Skill(...)` will raise an error. Use `SkillCoordinatedSet` or `SkillCoordinatedPopulation` instead.

**`add_coordinated_skill` vs `add_skill`** — Use `agent.add_coordinated_skill(coordinated)` to register a coordinated set on the agent. Using `agent.add_skill(...)` will not set up the coordination infrastructure.
