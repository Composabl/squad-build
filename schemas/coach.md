# Coach Schema — `SkillCoach`

A `SkillCoach` is the coordination layer for multi-agent RL. It runs alongside all
child skills simultaneously, adds a shared bonus reward, controls episode termination,
and optionally post-processes the combined action dict. It is hand-coded (not trained).

Used with `SkillCoordinatedSet` and `SkillCoordinatedPopulation`. See
[coordinated-skills.md](coordinated-skills.md) for how to wire a Coach into an agent.

## Import

```python
from amesa_core import SkillCoach
# or:
from composabl_core import SkillCoach
```

## Interface Contract

```python
class SkillCoach(ABC):

    # ── Required ────────────────────────────────────────────────────────

    async def compute_reward(
        self,
        multi_obs: dict,     # {skill_name: obs_dict} — each child's observation
        multi_action: dict,  # {skill_name: action}   — each child's action
        multi_reward: dict   # {skill_name: float}    — each child's teacher reward
    ) -> float:
        """
        Return a single float bonus added EQUALLY to all children's rewards.
        final_reward[child] = teacher_reward[child] + coach_bonus

        To differentiate bonuses per skill, encode that logic in each child's
        SkillTeacher.compute_reward — the coach cannot weight children differently.
        """

    async def compute_success_criteria(
        self,
        transformed_sensors: dict,
        action
    ) -> bool:
        """Return True to mark the episode as a success."""

    async def is_compute_done(
        self,
        transformed_sensors: dict,
        action
    ) -> bool:
        """
        Return True to terminate the episode.
        ⚠️ Override THIS method for termination, not compute_termination.
        The processor calls is_compute_done(), not compute_termination().
        Default: return False
        """
        return False

    # ── Optional ────────────────────────────────────────────────────────

    async def transform_action(
        self,
        transformed_sensors: dict,
        action: dict   # combined {skill_name: action} dict
    ) -> dict:
        """
        Post-process the combined action dict before sending to sim.
        Default: return action unchanged.
        ⚠️ Input differs between _execute() and step() call paths —
           _execute passes raw sim sensors; step() passes multi_sensors.
        """
        return action

    async def compute_action_mask(
        self,
        transformed_sensors: dict,
        action
    ):
        """Returns None by default."""
        return None
```

## Required Methods Summary

| Method | Returns | Notes |
|---|---|---|
| `compute_reward` | `float` | Bonus added equally to ALL children |
| `compute_success_criteria` | `bool` | Episode success signal |
| `is_compute_done` | `bool` | Episode termination — override THIS not `compute_termination` |

## Termination: Use `is_compute_done`

```python
# ✅ Correct:
async def is_compute_done(self, transformed_sensors, action) -> bool:
    return any(transformed_sensors[k]["crashed"] for k in transformed_sensors)

# ❌ Wrong — compute_termination is not called by the processor:
async def compute_termination(self, transformed_sensors, action) -> bool:
    return ...  # silently ignored
```

## Minimal Working Example

```python
from amesa_core import SkillCoach
from typing import Dict

class FleetCoach(SkillCoach):

    async def compute_reward(
        self,
        multi_obs: Dict,
        multi_action: Dict,
        multi_reward: Dict
    ) -> float:
        # Bonus when all agents are near their targets
        all_close = all(obs["distance_to_target"] < 5.0 for obs in multi_obs.values())
        return 2.0 if all_close else 0.0

    async def compute_success_criteria(self, transformed_sensors, action) -> bool:
        return all(
            transformed_sensors[k]["distance_to_target"] < 1.0
            for k in transformed_sensors
        )

    async def is_compute_done(self, transformed_sensors, action) -> bool:
        return any(
            transformed_sensors[k].get("crashed", False)
            for k in transformed_sensors
        )

    async def transform_action(self, transformed_sensors, action: dict) -> dict:
        # Clip all agents' thrust to safe range
        for key in action:
            if "thrust" in action[key]:
                action[key]["thrust"] = min(action[key]["thrust"], 1.0)
        return action
```

## Coach vs. Teacher

| | `SkillTeacher` | `SkillCoach` |
|---|---|---|
| Learns | Yes (RL policy) | No (hand-coded) |
| Scope | One skill | All children simultaneously |
| `compute_reward` returns | float for that skill | float bonus for ALL children |
| Termination method | `compute_termination` | `is_compute_done` |
| Training | Ray single-agent PPO | Ray multi-agent PPO |

## Reward Flow

```
Each step:
  for each child skill:
    teacher_reward[skill] = child.SkillTeacher.compute_reward(obs, action, next_obs)

  coach_bonus = await coach.compute_reward(multi_obs, multi_action, teacher_reward)

  final_reward[skill] = teacher_reward[skill] + coach_bonus  ← for every child
```

## Episode Termination Flow

```python
multi_terminated["__all__"] = (
    await coach.is_compute_done(sim_sensors, action)
) or sim_terminated
# When __all__ is True, all individual skill terminated flags are also set True
```

## Key Behavioral Notes

- Coach bonus is added equally — there is no per-child weighting from the coach.
- Child skills execute in dict insertion order (sequential for-loop), not concurrently.
- `compute_reward` on the coach receives the full multi-obs/action/reward dicts.
- `transform_action` receives different inputs in `_execute` vs `step` code paths
  (inconsistency in the SDK — be defensive).
- Override `is_compute_done` for termination. `compute_termination` is not in the
  `SkillCoach` ABC and is not called.
