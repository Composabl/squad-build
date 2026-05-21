# SkillCoach and SkillCoordinated

`SkillCoach` + `SkillCoordinated` enable multi-agent coordination: a coach directs a set of sub-skills, each receiving its own reward signal.

> **⚠️ Not available in v2 training.** `SkillCoordinatedSet` and `SkillCoordinatedPopulation` are **not supported** by the v2 Redis-stream training stack. The v2 episode manager, skill processor, and run loop have no coordinated-skill handling. Use coordinated skills only with the legacy (non-v2) training target. For most use cases, `SkillTeacher` or `SkillSelector` are sufficient.

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

Wraps a `SkillCoach` and a set of child skills into a coordinated unit.

```python
from amesa_core.agent.skill.skill_coordinated import SkillCoordinatedSet

coordinated = SkillCoordinatedSet("team-skill", MyCoach)
coordinated.add_skill(Skill("striker", StrikerTeacher))
coordinated.add_skill(Skill("defender", DefenderTeacher))

agent.add_coordinated_skill(coordinated)
```

---

## ⚠️ Quirks

**`compute_reward` returns a dict** — Keys must match the names of the child skills exactly.

**Coach is not a Teacher** — Passing a `SkillCoach` to a plain `Skill(...)` will raise an error. Use `SkillCoordinatedSet` or `SkillCoordinatedPopulation` instead.
