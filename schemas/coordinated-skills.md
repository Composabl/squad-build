# Coordinated Skills Schema

Coordinated skills run all child skills simultaneously every step, using Ray RLlib's
multi-agent PPO. Use them for multi-agent problems: fleets, swarms, collaborative
robotics, traffic networks.

## Two Concrete Classes

| Class | Children | Use when |
|---|---|---|
| `SkillCoordinatedSet` | `Skill` objects (heterogeneous) | Different roles per agent |
| `SkillCoordinatedPopulation` | `SkillPopulation` objects | Many copies of the same policy |

Both use a `SkillCoach` as coordinator (see [coach.md](coach.md)).

---

## `SkillCoordinatedSet`

### Import

```python
from amesa_core import SkillCoordinatedSet, Skill
# or:
from composabl_core import SkillCoordinatedSet
```

### Constructor

```python
SkillCoordinatedSet(
    name: str,                                         # unique name for this coordinated skill
    impl_cls: SkillCoach,                              # Coach class (not instance)
    config: dict | SkillCoordinatedSchema = {},        # optional config
    skills: list[Skill] = None,                        # add children inline
    train_batch_size: int = None,                      # top-level PPO batch size
    training_cycles: int = None,
)
```

### Methods

| Method | Description |
|---|---|
| `add_skill(skill: Skill)` | Add a child skill |
| `get_skills()` | Return all child skills |
| `get_skill_names()` | Return child skill name list |
| `set_sim_sensor_space(space)` | **Must** be a `Dict` space keyed by child skill names |
| `set_action_space(space)` | **Must** be a `Dict` space keyed by child skill names |
| `add_scenario(scenario)` | Add curriculum scenario |

### Registration

```python
agent.add_coordinated_skill(coordinated_skill)
```

Children are stored in `agent.skills_coordinated`, NOT `agent.skills`. They are not
returned by `agent.get_skills()`.

### Minimal Example

```python
from amesa_core import Agent, Skill, SkillCoordinatedSet, SkillCoach, SkillTeacher
import gymnasium as gym

class AgentTeacher(SkillTeacher):
    async def compute_reward(self, obs, action, sim_reward): return sim_reward
    async def compute_success_criteria(self, obs, action): return False
    async def transform_action(self, obs, action): return action
    async def filtered_sensor_space(self): return ["position", "velocity"]

class TeamCoach(SkillCoach):
    async def compute_reward(self, multi_obs, multi_action, multi_reward): return 0.0
    async def compute_success_criteria(self, sensors, action): return False
    async def is_compute_done(self, sensors, action): return False

skill1 = Skill("agent-1", AgentTeacher)
skill2 = Skill("agent-2", AgentTeacher)

coordinated = SkillCoordinatedSet("team", TeamCoach)
coordinated.add_skill(skill1)
coordinated.add_skill(skill2)

# Sim MUST expose Dict obs/action spaces keyed by exact skill names
coordinated.set_sim_sensor_space(gym.spaces.Dict({
    "agent-1": gym.spaces.Box(low=-1, high=1, shape=(2,)),
    "agent-2": gym.spaces.Box(low=-1, high=1, shape=(2,)),
}))
coordinated.set_action_space(gym.spaces.Dict({
    "agent-1": gym.spaces.Box(low=-1, high=1, shape=(1,)),
    "agent-2": gym.spaces.Box(low=-1, high=1, shape=(1,)),
}))

agent = Agent()
agent.add_sensors(sensors)
agent.add_coordinated_skill(coordinated)
```

---

## `SkillCoordinatedPopulation`

### Import

```python
from amesa_core import SkillCoordinatedPopulation, SkillPopulation
# or:
from composabl_core import SkillCoordinatedPopulation, SkillPopulation
```

### `SkillPopulation` Constructor

```python
SkillPopulation(
    name: str,                              # population group name (e.g. "car")
    impl_cls: SkillTeacher | Skill,         # teacher class for this population
    amount: int = 1                         # number of agents in this group
)
```

Instances are named `"{name}-1"`, `"{name}-2"`, ..., `"{name}-{amount}"`.

### Constructor

```python
SkillCoordinatedPopulation(
    name: str,
    impl_cls: SkillCoach,
    config: dict | SkillCoordinatedSchema = {},
    **kwargs  # same kwargs as SkillCoordinatedSet
)
```

### Key Difference from Set

Use `add_population()` — NOT `add_skill()`:

```python
# ✅ Correct:
coordinated.add_population(SkillPopulation("car", CarTeacher, amount=3))

# ❌ Wrong — raises Exception:
coordinated.add_skill(Skill(...))
```

### Minimal Example

```python
from amesa_core import SkillCoordinatedPopulation, SkillPopulation
import gymnasium as gym

population_cars   = SkillPopulation("car",   CarTeacher,   amount=2)
population_planes = SkillPopulation("plane", PlaneTeacher, amount=3)

coordinated = SkillCoordinatedPopulation("fleet", FleetCoach)
coordinated.add_population(population_cars)
coordinated.add_population(population_planes)

# Sim must expose keys: "car-1", "car-2", "plane-1", "plane-2", "plane-3"
coordinated.set_sim_sensor_space(gym.spaces.Dict({
    "car-1":   gym.spaces.Box(low=-1, high=1, shape=(4,)),
    "car-2":   gym.spaces.Box(low=-1, high=1, shape=(4,)),
    "plane-1": gym.spaces.Box(low=-1, high=1, shape=(6,)),
    "plane-2": gym.spaces.Box(low=-1, high=1, shape=(6,)),
    "plane-3": gym.spaces.Box(low=-1, high=1, shape=(6,)),
}))
```

---

## Comparison Table

| | `SkillCoordinatedSet` | `SkillCoordinatedPopulation` |
|---|---|---|
| Child unit | `Skill` | `SkillPopulation` |
| Add method | `add_skill()` | `add_population()` |
| Agent naming | `"{skill_name}"` | `"{pop_name}-{idx}"` |
| Heterogeneous | Yes | Per group (group is homogeneous) |
| Policies | One per Skill | One per population group name |
| Use case | Different roles | Fleets, swarms, replicated agents |

---

## Training Behavior

- All policies train **simultaneously** in one Ray run (unlike `SkillSelector` which
  trains children sequentially in DAG order).
- Uses `MultiAgentEnv` (Ray RLlib).
- One `PolicySpec` per child skill (Set) or per population group (Population).
- `train_batch_size` is set at the top-level coordinated skill — not on children.
- Child `model.fc_layers` IS used (for the child's `PolicySpec`).
- Top-level `model.fc_layers` and `model_io` are **silently ignored**.

## Reward Flow

```
Per step:
  teacher_reward[child] = child.teacher.compute_reward(obs, action, sim_reward)
  coach_bonus           = coach.compute_reward(multi_obs, multi_action, teacher_reward)
  final_reward[child]   = teacher_reward[child] + coach_bonus   ← for every child
```

The coach bonus is equal for all children. Per-child differentiation must be in
individual Teacher reward functions.

---

## Critical Rules

| Rule | Detail |
|---|---|
| Sim must use Dict spaces | `set_sim_sensor_space()` and `set_action_space()` must receive `gym.spaces.Dict` keyed by exact skill/population-instance names. Key mismatch → `KeyError` at init. |
| Children in `skills_coordinated` not `skills` | `agent.get_skills()` and pre-training validation do not see them. Register via `agent.add_coordinated_skill()`. |
| Use `is_compute_done` not `compute_termination` | The processor calls `coach.is_compute_done()`. `compute_termination` is not in the Coach ABC and is never called. |
| `SkillCoordinated` ABC is orphaned | Neither concrete class inherits from `SkillCoordinated`. `isinstance(skill, SkillCoordinated)` always returns `False`. Use concrete class checks. |
| `CoordinatedGoal` ≠ coordinated skills | Despite the name, `CoordinatedGoal` is a single-skill reward combiner. See [goals.md](goals.md). |
| Child execution is sequential | `_execute()` iterates children in insertion order — it is NOT concurrent despite multi-agent training setup. |
| Top-level `model_io` ignored | Set `model_io` on each child `Skill` individually if needed. |
