# SkillCoach and SkillCoordinated

`SkillCoach` + `SkillCoordinated` enable multi-agent coordination: a coach directs a set of sub-skills, each receiving its own reward signal.

> **✅ Fully supported in v1.** `SkillCoordinatedSet` and `SkillCoordinatedPopulation` are supported by the v1 Ray/RLlib training stack.

> **Note:** Coordinated skills are an advanced feature.

## SkillCoach

The coach equivalent of `SkillTeacher` — but it distributes rewards and coordinates actions across multiple sub-skills.

### Required methods

```python
async def compute_reward(self, transformed_sensors: dict, action, sim_reward) -> dict
# Returns a dict mapping sub-skill names to their individual rewards

async def compute_success_criteria(self, transformed_sensors: dict, action) -> bool
async def compute_termination(self, transformed_sensors: dict, action) -> bool
```

### Optional methods

```python
async def transform_sensors(self, sensors, action) -> dict   # default: pass-through
async def transform_action(self, transformed_sensors, action)  # default: pass-through
async def compute_action_mask(self, transformed_sensors, action)  # default: None
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

Wraps a `SkillCoach` and a fixed set of child skills into a coordinated unit. Each child trains with its own reward as distributed by the coach.

```python
from amesa_core.agent.skill.skill_coordinated import SkillCoordinatedSet
from amesa_core.agent.skill.skill import Skill

coordinated = SkillCoordinatedSet("team-skill", MyCoach)
coordinated.add_skill(Skill("striker", StrikerTeacher))
coordinated.add_skill(Skill("defender", DefenderTeacher))

agent.add_coordinated_skill(coordinated)
```

## SkillCoordinatedPopulation

Like `SkillCoordinatedSet`, but the coach manages a **population** of dynamically-scaled sub-skill instances. Useful for variable-size teams or population-based training.

```python
from amesa_core.agent.skill.skill_coordinated import SkillCoordinatedPopulation

coordinated = SkillCoordinatedPopulation("population-skill", MyCoach)
coordinated.add_skill(Skill("agent-template", AgentTeacher))

agent.add_coordinated_skill(coordinated)
```

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
