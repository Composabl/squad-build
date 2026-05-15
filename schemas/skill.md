# Skill Schema — `Skill`, `SkillSelector`, `SkillGroup`

These three wrappers are the structural building blocks of an agent. They hold
implementations (Teachers, Controllers, Coaches) and compose them into the agent's
decision DAG.

---

## `Skill` — Single Skill Wrapper

### Import

```python
from amesa_core.agent.skill.skill import Skill
# or:
from amesa_core import Skill
```

### Constructor

```python
Skill(
    name: str,                      # unique name within the agent
    impl_cls,                       # SkillTeacher subclass OR SkillController subclass (class, not instance)
    scenarios: list = None,         # optional list of Scenario objects
    **kwargs                        # training config (only applies if impl_cls is SkillTeacher)
)
```

Type inference is automatic: `issubclass(impl_cls, SkillTeacher)` → DRL skill;
`issubclass(impl_cls, SkillController)` → controller skill.

### Training kwargs (Teacher skills only)

```python
Skill(
    "temperature-control",
    MyTeacher,
    train_batch_size=4000,
    training_cycles=200,
    fc_layers=[256, 256],
    workers=2,
    learner_workers=0,
    envs_per_worker=1,
    num_cpus_per_worker=1.0,
    num_gpus_per_worker=0.0,
    custom_action_space=None,
)
```

Training kwargs are **silently dropped** when `impl_cls` is a `SkillController`.

### Adding Scenarios

```python
skill = Skill("my-skill", MyTeacher)
skill.add_scenario(Scenario({"temp": {"data": 80, "type": "is_equal"}}))
skill.add_scenario(Scenario({"temp": {"data": [70, 90], "type": "is_between"}}))
```

Or pass scenarios at construction:

```python
skill = Skill("my-skill", MyTeacher, scenarios=[scenario1, scenario2])
```

### Registration

```python
agent.add_skill(skill)
```

---

## `SkillSelector` — Orchestrator / Router

A selector chooses which child skill runs each step. The routing logic is either:
- **Learned**: a `SkillTeacher` subclass that outputs an integer index via its RL policy
- **Deterministic**: a `SkillController` subclass whose `compute_action()` returns an integer index

### Import

```python
from amesa_core.agent.skill.skill_selector import SkillSelector
```

### Constructor

```python
SkillSelector(
    name: str,
    impl_cls,                      # SkillTeacher or SkillController — returns int index
    children: list[str],           # names of child skills this selector routes among
)
```

### Registration

```python
selector = SkillSelector(
    "mode-selector",
    ModeSelectorTeacher,           # routes using learned RL policy
    children=["fast-mode", "safe-mode"]
)
agent.add_selector_skill(selector, ...)
```

### Selector Implementation (SkillController example)

```python
from amesa_core import SkillController

class ModeSelector(SkillController):
    async def compute_action(self, transformed_sensors, action) -> int:
        # Return 0-based index of child to activate
        if abs(transformed_sensors["position"][0]) < 1.0:
            return 0   # "fast-mode"
        return 1       # "safe-mode"

    async def filtered_sensor_space(self):
        return ["position"]

    async def compute_success_criteria(self, transformed_sensors, action):
        return False

    async def compute_termination(self, transformed_sensors, action):
        return False
```

### Selector Implementation (SkillTeacher example)

```python
from amesa_core import SkillTeacher

class AdaptiveSelector(SkillTeacher):
    async def compute_reward(self, transformed_sensors, action, sim_reward):
        # Reward the selector for choosing well
        return sim_reward

    async def compute_success_criteria(self, transformed_sensors, action):
        return transformed_sensors["error"] < 0.1

    async def transform_action(self, transformed_sensors, action):
        return action   # action IS the child index (already int)

    async def filtered_sensor_space(self):
        return ["error", "mode"]
```

### Action Masks on Selectors

Selectors always have discrete action spaces (the integer index), so action masks
always apply:

```python
async def compute_action_mask(self, transformed_sensors, action):
    if transformed_sensors["battery"] < 0.1:
        return [False, True]  # disable "fast-mode", force "safe-mode"
    return None
```

---

## `SkillGroup` — Plan → Execute Pipeline

A SkillGroup links exactly two skills in a directed pipeline: the first skill's
output action is injected into the second skill's observation dict at every step.

### Import

```python
from amesa_core.agent.skill.skill_group import SkillGroup
# or:
from composabl_core import SkillGroup
```

### Constructor

```python
SkillGroup(
    first_skill: Skill = None,   # runs first; action injected into second's obs
    second_skill: Skill = None,  # receives first skill's action in its sensor dict
)
```

### Registration — CRITICAL

`add_skill_group()` does NOT register children in `agent.skills`. The safest pattern:

```python
# ✅ Use add_skill() with a SkillGroup — registers children AND the group atomically
agent.add_skill(SkillGroup(skill_a, skill_b))

# ✅ Or register children first, then the group:
agent.add_skill(skill_a)
agent.add_skill(skill_b)
agent.add_skill_group(SkillGroup(skill_a, skill_b))

# ❌ Wrong — children never added to agent.skills, invisible to training:
agent.add_skill_group(SkillGroup(skill_a, skill_b))
```

### How the Pipeline Works

First skill's action is injected into second skill's sensor dict under the key
`first_skill.get_name()` (i.e., the first skill's name string).

The second skill's Teacher must include that key in `filtered_sensor_space()`:

```python
# first_skill = Skill("approach", ...)

class ExecuteTeacher(SkillTeacher):
    async def filtered_sensor_space(self):
        # "approach" = first_skill.get_name()
        return ["position", "velocity", "approach"]

    async def transform_sensors(self, sensors, action):
        approach_action = sensors.get("approach")  # first skill's output
        ...
        return sensors
```

If `"approach"` is not in `filtered_sensor_space()`, the injected action is
**silently dropped** — no warning.

### Training Order

Skills in a group train independently in DAG order. The SDK trains from bottom to
top: `second_skill` must reach competence first, then `first_skill` trains against
the learned `second_skill` policy.

During `first_skill` training, `second_skill` runs with `is_training=False`
(inference only — its weights do not update).

### Limitations

- Exactly 2 skills. No N-way chaining.
- Nesting (`SkillGroup` inside `SkillGroup`) silently no-ops.
- `update_skill_groups()` is broken/deprecated — do not call it.
- `SkillGroup.id` UUID is generated but never read — cannot look up by ID.
- Context manager (`with SkillGroup() as sg:`) works but `__exit__` is a no-op.

### Context Manager Style

```python
with SkillGroup() as sg:
    sg.set_first_skill(Skill("plan", PlanTeacher, scenarios))
    sg.set_second_skill(Skill("execute", ExecuteTeacher, scenarios))

agent.add_skill(sg.get_first_skill())
agent.add_skill(sg.get_second_skill())
agent.add_skill_group(sg)
```

---

## Composition Patterns

### Hierarchy / Sequence (Selector → Child Skills)

```python
navigation = SkillSelector("nav", NavigationSelector, [
    "path-planning",
    "obstacle-avoidance",
    "target-approach"
])
```

### Plan → Execute (SkillGroup)

```python
plan_skill    = Skill("plan",    MixerPlanTeacher)      # decides set point
execute_skill = Skill("execute", MPCController)          # determines coolant flow
agent.add_skill(SkillGroup(plan_skill, execute_skill))
```

### Fault Tolerance (Selector with fallback)

```python
class FaultSelector(SkillController):
    async def compute_action(self, obs, action) -> int:
        return 0 if obs["system_health"][0] > 0.8 else 1  # 0=normal, 1=safe-mode

fallback = SkillSelector("fault-tolerant", FaultSelector, ["normal-ctrl", "safe-mode"])
```
