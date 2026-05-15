# Coordinated Skills

## Overview

Coordinated skills are a first-class SDK pattern for **multi-agent RL** — every child skill
executes in the same timestep, in parallel, every step. This distinguishes them from the other
two skill-composition primitives:

| Pattern | Skills active per step | Action produced | Orchestrator | Training |
|---|---|---|---|---|
| `SkillSelector` | 1 (chosen by RL index) | Single scalar/vector | SkillTeacher (learns to route) | Multi-policy, one at a time |
| `SkillGroup` | 1 sequential pipeline | First skill → obs for second | None | Independent, standalone |
| `SkillCoordinatedSet` | **All N, simultaneously** | `Dict` keyed by skill name | `SkillCoach` (hand-coded) | Joint multi-agent PPO |
| `SkillCoordinatedPopulation` | **All N×M, simultaneously** | `Dict` keyed by `"{pop}-{idx}"` | `SkillCoach` (hand-coded) | Joint multi-agent PPO |

The coordinator is a hand-coded `SkillCoach`, not a learned policy. There is no "routing"
decision — all children always run.

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
  Sim sends               │         SkillCoordinatedSet              │
  Dict obs space          │                                          │
  keyed by skill name     │   sim_sensors["skill_a"] ──► Skill A ──► action_a ─┐
                          │   sim_sensors["skill_b"] ──► Skill B ──► action_b ─┼──► Joint Action Dict ──► Sim
                          │   sim_sensors["skill_c"] ──► Skill C ──► action_c ─┘
                          │                                          │
                          │   SkillCoach.transform_action(           │
                          │       sim_sensors, {a, b, c})            │
                          │   ──► post-processes the combined dict   │
                          │                                          │
                          │   SkillCoach.compute_reward(             │
                          │       multi_obs, multi_action,           │
                          │       multi_reward) ──► float bonus      │
                          │       added equally to ALL children      │
                          └─────────────────────────────────────────┘
```

**vs. SkillSelector:**

```
  SkillSelector (orchestration pattern)         SkillCoach (coordinated pattern)
  ─────────────────────────────────────         ───────────────────────────────
  Learned RL policy                             Hand-coded Python class
  Outputs integer index → picks ONE child       No routing; ALL children always run
  Registered via agent.add_skill_selector()     Registered via agent.add_coordinated_skill()
  Uses SingleAgentEnv                           Uses MultiAgentEnv (Ray RLlib)
```

---

## The Two Concrete Classes

### `SkillCoordinatedSet`

Holds N **heterogeneous** named skills. Each child is a distinct `Skill` with its own policy,
obs slice, and action fragment.

```python
from composabl_core import SkillCoordinatedSet, Skill

SkillCoordinatedSet(
    name: str,                                          # unique name for this coordinated skill
    impl_cls: SkillCoach,                               # Coach class — NOT SkillTeacher
    config: Union[dict, SkillCoordinatedSchema] = {},   # optional config
    **kwargs:
        skills: List[Skill],         # add children inline
        train_batch_size: int,       # → skill_config.learning.train_batch_size
        training_cycles: int,        # → skill_config.learning.training_cycles
)
```

**Methods:**

| Method | Description |
|---|---|
| `add_skill(skill: Skill)` | Add a child skill |
| `get_skills() -> List[Skill]` | Return all child skills |
| `get_skill_names() -> List[str]` | Return child skill name list |
| `set_sim_sensor_space(space: gym.Space)` | **Must be a `Dict` space** keyed by child skill names |
| `set_action_space(space: gym.Space)` | **Must be a `Dict` space** keyed by child skill names |
| `get_action_space() -> amesa_spaces.Dict` | Composed dict of each child's action space |
| `add_scenario(scenario)` | Add curriculum scenario (inherited) |

### `SkillCoordinatedPopulation`

Holds N **population groups**, each group containing `amount` copies of the same skill
template. Use this for homogeneous multi-agent problems (e.g., five robots of type A, three
of type B).

```python
from composabl_core import SkillCoordinatedPopulation, SkillPopulation

SkillCoordinatedPopulation(
    name: str,
    impl_cls: SkillCoach,
    config: Union[dict, SkillCoordinatedSchema] = {},
    **kwargs  # same kwargs as Set
)
```

Children are added as `SkillPopulation` objects, not bare `Skill` objects:

```python
SkillPopulation(
    name: str,                              # population group name, e.g. "car"
    impl_cls: Union[SkillTeacher, Skill],   # teacher class or Skill instance
    amount: int = 1                         # number of copies in this population
)
```

At runtime, instances are named `"{name}-1"`, `"{name}-2"`, ..., `"{name}-{amount}"`.

**Key difference:** calling `add_skill()` on a `SkillCoordinatedPopulation` raises an
`Exception`. You must use `add_population(population: SkillPopulation)`.

### Comparison Table

| | `SkillCoordinatedSet` | `SkillCoordinatedPopulation` |
|---|---|---|
| Child unit | `Skill` | `SkillPopulation` |
| Add method | `add_skill()` | `add_population()` |
| Agent naming | `"{skill_name}"` | `"{pop_name}-{idx}"` |
| Heterogeneous skills | Yes | Per group (each group is homogeneous) |
| Use case | Different roles (sensor, actuator) | Fleets, swarms, repeated agents |
| Policy per group | One per Skill | One per population group name |
| Ray agent count | N | sum of all `amount` values |

---

## SkillCoach

`SkillCoach` is the coordinator ABC for both `SkillCoordinatedSet` and
`SkillCoordinatedPopulation`. It is **not a SkillTeacher** and does not learn. It provides
four coordination hooks called by the processor each step.

```python
from composabl_core import SkillCoach
from typing import Dict

class SkillCoach(ABC):

    async def compute_reward(
        self,
        multi_obs: Dict,      # {skill_name: obs} — each child's observation
        multi_action: Dict,   # {skill_name: action} — each child's action
        multi_reward: Dict    # {skill_name: float} — each child's teacher reward
    ) -> float:
        # Returns a SINGLE float bonus added equally to ALL child rewards
        ...

    async def compute_success_criteria(
        self,
        transformed_sensors: Dict,
        action
    ) -> bool:
        # Episode success — return True to mark episode as successful
        ...

    async def is_compute_done(
        self,
        transformed_sensors: Dict,
        action
    ) -> bool:
        # Episode termination — return True to end the episode
        # NOTE: override THIS method, not compute_termination (see Gotchas)
        return False

    # Optional:
    async def transform_action(
        self,
        transformed_sensors: Dict,
        action: Dict
    ) -> Dict:
        # Post-process the combined action dict before sending to sim
        # Default returns action unchanged
        return action

    async def compute_action_mask(self, transformed_sensors: Dict, action):
        # Returns None by default
        return None
```

**SkillCoach vs SkillTeacher:**

| | `SkillTeacher` | `SkillCoach` |
|---|---|---|
| Learns | Yes (RL policy) | No (hand-coded) |
| Scope | Single skill | All children simultaneously |
| `compute_reward` returns | float for that one skill | float bonus added to ALL children |
| Termination method | `compute_termination` | `is_compute_done` |
| Action manipulation | `transform_action(obs, action)` | `transform_action(sensors, action_dict)` |

---

## Training

Coordinated skills use Ray RLlib's **multi-agent PPO** with one `PolicySpec` per child skill
(Set) or per population group (Population). The agent environment extends `MultiAgentEnv`.

### Policy Structure

```
SkillCoordinatedSet:
  policies_dict = {
      "skill_a": PolicySpec(observation_space=..., action_space=...),
      "skill_b": PolicySpec(...),
  }
  policy_map_fn = lambda agent_id: agent_id   # identity

SkillCoordinatedPopulation:
  policies_dict = {
      "car":   PolicySpec(...),   # shared by car-1, car-2
      "plane": PolicySpec(...),   # shared by plane-1, plane-2, plane-3
  }
```

All policies train **simultaneously in the same Ray run** (unlike `SkillSelector` which trains
one child at a time in topological order).

### Reward Flow

```
Each step:
  for each child skill:
    teacher_reward = child.SkillTeacher.compute_reward(obs, action, next_obs)
    multi_reward[skill_name] = teacher_reward

  coach_bonus = await coach.compute_reward(multi_obs, multi_action, multi_reward)

  for each skill_name:
    final_reward[skill_name] = multi_reward[skill_name] + coach_bonus
```

The coach bonus is **added equally to all children**. There is no per-child weighting from the
coach. Differentiated bonuses must be encoded into individual `SkillTeacher.compute_reward`
methods.

### Termination

```python
multi_terminated["__all__"] = (await coach.is_compute_done(sim_sensors, action)) or sim_terminated
# when __all__ is True, all individual skill terminated flags are also set True
```

### Training-Specific Config

- `train_batch_size` — set on the top-level coordinated skill, not on individual children.
  Children's `learning` configs provide `fc_layers` for their `PolicySpec` only.
- `sample_timeout_s` — automatically scaled by `len(policies_to_train)` (Set) or
  `sum(population.amount)` (Population).
- **Checkpoint** — saved via Ray `algo.save()` + per-child ONNX export to
  `{child.checkpoint_uri}/policies/{child.name}/{child.name}.onnx`. Restore is attempted on
  init; silent failure if path missing → trains from scratch.

### How This Differs from Standard Training Paths

| | Single-agent (Ray) | Redis Streams V2 | Coordinated Skills |
|---|---|---|---|
| Ray mode | SingleAgentEnv | External (Redis) | MultiAgentEnv |
| # policies | 1 | 1 | N (one per child) |
| Joint training | N/A | N/A | Yes — all policies in one run |
| Topological order | No | No | No — all simultaneous |

---

## ⚠️ The `SkillCoordinated` ABC — Orphaned Base Class

`SkillCoordinated` is defined in `skill_coordinated.py` as an abstract base class. It
inherits from both `Skill` and `abc.ABC`, and defines the interface that coordinated skills
should implement.

**Neither concrete class inherits from it.**

```
Skill
├── SkillCoordinatedSet         (inherits Skill directly)
├── SkillCoordinatedPopulation  (inherits Skill directly)
└── SkillCoordinated (ABC)      (inherits Skill + abc.ABC — no child inherits THIS)
```

### What This Means in Practice

- `isinstance(skill, SkillCoordinated)` always returns `False` for both concrete classes.
- You cannot use `SkillCoordinated` as a type guard or isinstance check anywhere in your code.
- The ABC's `raise NotImplementedError` stubs are unreachable — they are never triggered.
- The `make_skill_processor()` dispatch in `amesa_train` routes by `isinstance` checks against
  the concrete classes directly, not against the ABC.

If you are writing framework extension code that needs to detect coordinated skills, check for
`isinstance(skill, SkillCoordinatedSet)` or `isinstance(skill, SkillCoordinatedPopulation)`
explicitly.

There is also a typo on the ABC: `get_namme()` (double `m`) is defined but never called by
any processor or framework code. `get_name()` from the `Skill` base class is used instead.

---

## ⚠️ `CoordinatedGoal` — The Naming Trap

`CoordinatedGoal` is **not related to coordinated skills**. Despite the shared prefix, it is a
completely separate concept.

**What `CoordinatedGoal` actually is:** a single-skill reward combiner that merges multiple
`Goal` objects inside one skill's teacher using AND / OR / THEN logic.

```python
from composabl_core import CoordinatedGoal, GoalCoordinationStrategy

class MyTeacher(SkillTeacher):
    def __init__(self):
        self.goal = CoordinatedGoal(
            goals=[MaximizeGoal(...), AvoidGoal(...)],
            goals_coordination_strategy=GoalCoordinationStrategy.AND,
            weights=[0.7, 0.3]
        )
```

`GoalCoordinationStrategy.AND` — weighted sum of all goal rewards  
`GoalCoordinationStrategy.OR` — max of all goal rewards  
`GoalCoordinationStrategy.THEN` — reward from first incomplete goal (sequential completion)

**What to use for multi-agent coordination:** `SkillCoordinatedSet` or
`SkillCoordinatedPopulation` with a `SkillCoach`.

There is zero connection between `CoordinatedGoal` and `SkillCoordinatedSet` or
`SkillCoordinatedPopulation` at the source level.

---

## Config Parameters

### `SkillCoordinatedOptions` (top-level)

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `remote_address` | `Optional[str]` | `None` | Remote coach host |
| `impl_cls` | `Optional[Union[str, CLSSchema]]` | `None` | Serialized coach class reference |
| `impl_cls_data` | `SkillImplDataSchema` | `SkillImplDataSchema()` | Goals/guidance data |
| `learning` | `LearningConfig` | `LearningConfig()` | PPO hyperparams — `train_batch_size`, `gamma`, etc. Active at top level |
| `resources` | `ResourcesConfig` | `ResourcesConfig()` | Worker/GPU config |
| `skills` | `Union[SKILLS_POPULATION, SKILLS_SET]` | `[]` | Serialized child skills |
| `scenarios` | `List[ScenarioSchema]` | `[]` | Curriculum scenarios |
| `scenarios_current_idx` | `Optional[int]` | `0` | Current scenario pointer |
| `model` | `ModelConfig` | `ModelConfig()` | ⚠️ `fc_layers` **NOT used** — code comment confirms; children define their own layers |
| `model_io` | `ModelIOConfig` | `ModelIOConfig()` | ⚠️ **Never read** by processor or trainer at the parent level |

### Per-child `SkillDRLOptions`

Each child `Skill` carries its own `learning`, `resources`, `model`, `model_io`. The child's
`model.fc_layers` **is** active — it populates the child's `PolicySpec` in Ray. The child's
`learning` config is **not** used for `train_batch_size`; that comes from the top-level
coordinated skill.

---

## Gotchas

1. **Sim MUST expose a `Dict` obs/action space keyed by exact skill names.** For `SkillCoordinatedSet`, the sim returns `Dict({"skill_a": Space, "skill_b": Space})`. For Population, keys are `"{pop_name}-{idx}"` (e.g., `"car-1"`, `"car-2"`). No automatic slicing or broadcasting occurs — a key mismatch causes `KeyError` at `set_sim_sensor_space()` initialization.

2. **Child skills live in `agent.skills_coordinated`, not `agent.skills`.** Register the coordinated skill with `agent.add_coordinated_skill()`. Children are stored inside `coordinated_skill.skill_config.skills` and accessible via `agent.skills_coordinated`. They are **not** returned by `agent.get_skills()` and **not** discoverable via `agent.get_node_by_name()` unless the lookup explicitly walks `skills_coordinated` children.

3. **`SkillCoordinatedOptions.model.fc_layers` is silently ignored.** The parent-level model architecture config does nothing. No warning is emitted. Set layer sizes on each child `Skill`'s own config.

4. **`SkillCoordinatedOptions.model_io` is silently ignored at the parent level.** The parent-level `model_io` is never read by the processor or trainer. Set `model_io` on each child `Skill` individually.

5. **Coach `compute_reward` adds a single bonus equally to ALL children.** There is no per-child weighting from the coach. To differentiate bonuses per skill, encode that logic in each child's `SkillTeacher.compute_reward`, not in the coach.

6. **Child execution inside `_execute()` is a sequential for-loop, not truly concurrent.** `init()` and `reset()` use `asyncio.gather()`. `_execute()` and `step()` iterate children in dict insertion order. For latency-sensitive deployments this is effectively serialized despite the multi-agent setup.

7. **Override `is_compute_done`, not `compute_termination`.** The `SkillCoach` ABC lists `compute_termination` in its docstring, but the processor calls `coach.is_compute_done()`. Subclass `SkillCoach` and override `is_compute_done` for termination logic.

8. **`transform_action` receives different inputs in `_execute` vs `step`.** In the `_execute` path, `coach.transform_action(sim_sensors, actions)` passes the raw sim sensor dict. In the `step` path, it passes `multi_sensors` (filtered/processed). The input to coach methods is inconsistent between the two code paths.

9. **Population trainer registers one policy per group name; processor creates instances named `"{group}-{idx}"`.** Ray's identity `policy_map_fn` tries to look up `"car-1"` in a policy dict that only has `"car"`. In practice, all instances of a population share one policy — confirm that your Ray version resolves this mapping at runtime before deploying.

10. **`SkillCoordinated` ABC is structurally orphaned.** Both concrete classes inherit directly from `Skill`, not from `SkillCoordinated`. `isinstance(skill, SkillCoordinated)` returns `False` for both. The ABC's stubs are unreachable. Do not use it for type guards. (Full details in [the orphaned ABC section above](#️-the-skillcoordinated-abc--orphaned-base-class).)

11. **`CoordinatedGoal` is unrelated to coordinated skills.** The shared prefix is a naming accident. `CoordinatedGoal` is a single-skill reward combiner with no connection to `SkillCoordinatedSet` or `SkillCoordinatedPopulation`. (Full details in [the naming trap section above](#️-coordinatedgoal--the-naming-trap).)

---

## Examples

### Minimal `SkillCoordinatedSet`

```python
from composabl_core import Agent, Skill, SkillCoordinatedSet, SkillCoach, SkillTeacher
from typing import Dict

class IncrementTeacher(SkillTeacher):
    async def compute_reward(self, obs, action, next_obs) -> float:
        return 1.0

    async def compute_success_criteria(self, obs, action) -> bool:
        return False

    async def transform_action(self, obs, action):
        return action

    def filtered_sensor_space(self):
        return ["my_sensor"]

class CoordinatedCoach(SkillCoach):
    async def compute_reward(
        self,
        multi_obs: Dict,
        multi_action: Dict,
        multi_reward: Dict
    ) -> float:
        # Bonus applied equally to all children
        return 0.1

    async def compute_success_criteria(self, transformed_sensors, action) -> bool:
        return False

    async def is_compute_done(self, transformed_sensors, action) -> bool:
        return False

skill1 = Skill("skill1", IncrementTeacher)
skill2 = Skill("skill2", IncrementTeacher)

coordinated = SkillCoordinatedSet("my_coordinated", CoordinatedCoach)
coordinated.add_skill(skill1)
coordinated.add_skill(skill2)

# Sim must return Dict({"skill1": obs_space, "skill2": obs_space})
coordinated.set_sim_sensor_space(gym.spaces.Dict({
    "skill1": gym.spaces.Box(...),
    "skill2": gym.spaces.Box(...),
}))

agent = Agent()
agent.add_sensors(sensors)
agent.add_coordinated_skill(coordinated)
```

### `SkillCoordinatedPopulation`

```python
from composabl_core import SkillCoordinatedPopulation, SkillPopulation

# 2 car agents + 3 plane agents — 5 total agents in joint multi-agent PPO
population_cars = SkillPopulation("car", CarTeacher, amount=2)
population_planes = SkillPopulation("plane", PlaneTeacher, amount=3)

coordinated = SkillCoordinatedPopulation("fleet", FleetCoach)
coordinated.add_population(population_cars)
coordinated.add_population(population_planes)

# Sim must return Dict with keys: "car-1", "car-2", "plane-1", "plane-2", "plane-3"
coordinated.set_sim_sensor_space(gym.spaces.Dict({
    "car-1": gym.spaces.Box(...),
    "car-2": gym.spaces.Box(...),
    "plane-1": gym.spaces.Box(...),
    "plane-2": gym.spaces.Box(...),
    "plane-3": gym.spaces.Box(...),
}))

agent = Agent()
agent.add_coordinated_skill(coordinated)
```

### Coach with `transform_action`

```python
class FleetCoach(SkillCoach):
    async def compute_reward(self, multi_obs, multi_action, multi_reward) -> float:
        # Bonus for collective behavior (e.g., all agents near goal)
        all_near = all(obs["distance"] < 5.0 for obs in multi_obs.values())
        return 2.0 if all_near else 0.0

    async def compute_success_criteria(self, sensors, action) -> bool:
        return all(sensors[k]["distance"] < 1.0 for k in sensors)

    async def is_compute_done(self, sensors, action) -> bool:
        # NOTE: override is_compute_done, not compute_termination
        return any(sensors[k]["crashed"] for k in sensors)

    async def transform_action(self, sensors, action: dict) -> dict:
        # Clip all agents' thrust to safe range
        for key in action:
            action[key]["thrust"] = min(action[key]["thrust"], 1.0)
        return action
```
