# Agent

`Agent` assembles sensors, perceptors, and skills into a deployable agent.

## Three construction APIs

All three produce identical agents — use whichever matches your style.

### 1. Declarative API

Add components one at a time via `add_*` methods.

```python
from amesa_core.agent.agent import Agent
from amesa_core.agent.sensors.sensor import Sensor
from amesa_core.agent.perceptor.perceptor import Perceptor
from amesa_core.agent.skill.skill import Skill
from amesa_core.agent.skill.skill_selector import SkillSelector
from amesa_core.agent.skill.skill_group import SkillGroup

agent = Agent()

# Sensors
agent.add_sensors([
    Sensor("x", sensor_mapping="lambda s: s['x']"),
    Sensor("y", sensor_mapping="lambda s: s['y']"),
])

# Perceptors (optional)
agent.add_perceptor(my_perceptor)

# Simple skill
skill = Skill("my-skill", MyTeacher, training_cycles=100)
skill.add_scenario({"x": [0.0, 1.0]})
agent.add_skill(skill)

# Selector skill
selector = SkillSelector("my-selector", SelectorTeacher)
selector.add_child(skill_a)
selector.add_child(skill_b)
agent.add_selector_skill(selector, fixed_order=True, repeat=False)

# Coordinated skill (fully supported in v1)
from amesa_core.agent.skill.skill_coordinated import SkillCoordinatedSet
coordinated = SkillCoordinatedSet("team", MyCoach)
coordinated.add_skill(Skill("striker", StrikerTeacher))
agent.add_coordinated_skill(coordinated)
```

### 2. Tree Builder API

Pass the full skill tree at once.

```python
agent = Agent()
agent.add_sensors([s1, s2])

agent.create_from_tree(
    SkillSelector(
        "top-selector", SelectorTeacher,
        children=[
            Skill("skill-a", TeacherA),
            SkillGroup(
                Skill("pipe-1", TeacherB),
                Skill("pipe-2", TeacherC),
            ),
        ]
    )
)
```

### 3. Context Manager API

Build the tree interactively with `with` blocks.

```python
with SkillSelector("top-selector", SelectorTeacher) as ss:
    with Skill("skill-a", TeacherA) as sa:
        sa.add_scenario(scenario_a)
        ss.add_child(sa)

    with SkillGroup() as sg:
        sg.set_first_skill(Skill("pipe-1", TeacherB))
        sg.set_second_skill(Skill("pipe-2", TeacherC))
        ss.add_child(sg)

    agent.add_selector_skill(ss, fixed_order=True, repeat=False)
```

## Key methods

| Method | Signature | Notes |
|---|---|---|
| `add_sensor` | `(sensor: Sensor)` | Add one sensor |
| `add_sensors` | `(sensors: list[Sensor])` | Add multiple sensors |
| `add_perceptor` | `(perceptor: Perceptor)` | Add one perceptor |
| `add_perceptors` | `(perceptors: list[Perceptor])` | Add multiple |
| `add_skill` | `(skill: Skill)` | Add a leaf skill |
| `add_selector_skill` | `(selector, children=None, fixed_order=False, repeat=True)` | Add a selector |
| `add_skill_group` | `(group: SkillGroup)` | Add a skill group |
| `add_coordinated_skill` | `(coordinated)` | Add a coordinated skill (fully supported in v1) |
| `create_from_tree` | `(root_skill)` | Build from a skill tree |
| `export(directory)` | — | Serialize agent to disk |
| `load(file)` | — | (static) Deserialize agent from a JSON file |
| `draw_text()` | — | Print agent graph to console |

## Visualizing

```python
print(agent)         # prints summary with graph
agent.draw_text()    # same, more detailed
agent.draw()         # matplotlib visualization (blocking)
```

---

## ⚠️ Quirks

**Child skills must be registered on the agent** — When using a `SkillSelector`, each child `Skill` must also be added to the agent via `add_skill()`. The declarative API's `add_selector_skill(selector, [child1, child2])` does this automatically. In the tree/context-manager APIs, the agent collects children from the tree.

**Sensor order matters** — Sensors are used in declaration order when mapping array observations. If your sim returns a numpy array (not a dict), define sensors in the same order as the array dimensions.

**`add_skill` vs `add_selector_skill`** — `add_skill` is for leaf `Skill` objects. `add_selector_skill` is for `SkillSelector` objects. Passing a `SkillSelector` to `add_skill` (or vice versa) will silently misbehave.

**`add_coordinated_skill` is fully supported** — Unlike v2, v1 fully supports `add_coordinated_skill(...)`. See [skills/coordinated.md](skills/coordinated.md).
