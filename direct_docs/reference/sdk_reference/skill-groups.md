# Skill Groups

## Overview

A `SkillGroup` is a **runtime composition mechanism** that links exactly two `Skill` instances in a directed pipeline: the first skill's output action is injected into the second skill's observation space at every step, so the second skill can condition its policy on what the first skill just did.

`SkillGroup` is **not** a subclass of `Skill`. It has no `SkillTeacher`, no `SkillController`, and no training logic of its own. It is a standalone wrapper that holds two skills and describes the data flow between them.

**Key distinction from multi-skill agents:** Adding multiple skills to an agent via `add_skill()` places those skills in the scheduling DAG for independent selection or coordination. A `SkillGroup` is different — it fuses two skills into a single step-time pipeline where the first skill's action *becomes part of the second skill's observation*. The group itself is invisible to the DAG; it lives in a separate flat list (`agent.skill_groups`) outside `agent.skills`, `agent.skill_selectors`, and `agent.skills_coordinated`.

**Runtime-only:** The group pipeline only activates at inference and step time. During training, each member skill trains independently as a normal skill in the DAG. The `SkillGroup` relationship has no effect on the training loop, reward signals, or gradient updates.

---

## Architecture

### Step-Time Pipeline

```
        Observation (from sim)
               │
               ▼
   ┌───────────────────────┐
   │  Skill A (first_skill) │
   │  Teacher / Controller  │
   └───────────┬───────────┘
               │ Action A
               │
               ├─────────────────────────────────────┐
               │                                     │
               ▼                                     ▼
   injected into Skill B's              Skill B sensor dict:
   observation dict as key                { ..., "skill-a": Action_A }
   "skill-a" (first_skill.get_name())
               │
               ▼
   ┌───────────────────────┐
   │  Skill B (second_skill)│
   │  sees Action_A in obs  │
   └───────────┬───────────┘
               │
               ▼
          Final Action
         (returned to sim)
```

### Placement Inside the Agent

```
┌──────────────────────────────────────────────────────────────┐
│  Agent                                                        │
│                                                               │
│  agent.skills          ← DAG nodes (training, selection)     │
│   ├── Skill A  ◄──────────────────── also registered here    │
│   └── Skill B  ◄──────────────────── also registered here    │
│                                                               │
│  agent.skill_groups    ← flat list, outside DAG              │
│   └── SkillGroup(A, B) ← pipeline description                │
│                                                               │
│  SkillSelector / Orchestration layer                         │
│   └── sees A and B as normal DAG skills                      │
│       (SkillGroup is transparent to selector)                │
└──────────────────────────────────────────────────────────────┘
```

The `SkillSelector` is unrelated to `SkillGroup`. Groups operate at the processor level and are transparent to orchestration. See the [Orchestration reference](orchestration.md) for how the DAG and selector layer work.

---

## API Surface

### `SkillGroup` Constructor

```python
from composabl_core.agent.skill.skill_group import SkillGroup

class SkillGroup:
    def __init__(
        self,
        first_skill: Skill = None,
        second_skill: Skill = None,
    )
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `first_skill` | `Skill` | Yes (at `init()`) | Runs first; its action is injected into the second skill's observation |
| `second_skill` | `Skill` | Yes (at `init()`) | Receives the first skill's action in its sensor dict; its action is the final output |

**Constraints:**
- Exactly 2 skills. No variadic constructor, no `add_skill()` method on `SkillGroup`.
- Both defaults to `None`. Validation fires at `await sg.init()`, **not** at construction. `SkillGroup(None, None)` succeeds silently.
- `SkillGroup` is **not** a `Skill` subclass — you cannot nest a group inside another group.

### Instance Methods

| Method | Signature | Notes |
|---|---|---|
| `init()` | `async def init()` | Validates both skills non-`None`, calls `skill.init()` on each. Raises `Exception("SkillGroup needs at least 2 skills")` if either is `None`. |
| `set_first_skill()` | `def set_first_skill(first_skill: Skill)` | Setter |
| `set_second_skill()` | `def set_second_skill(second_skill: Skill)` | Setter |
| `get_first_skill()` | `def get_first_skill() -> Skill` | Getter |
| `get_second_skill()` | `def get_second_skill() -> Skill` | Getter |
| `get_skills()` | `def get_skills() -> List[Skill]` | Returns `[first_skill, second_skill]` |
| `to_list()` | `def to_list() -> list` | Exact duplicate of `get_skills()` — no distinct behavior |
| `get_name()` | `def get_name() -> str` | Returns `"sg-[{first_name},{second_name}]"` |
| `to_json()` | `def to_json() -> list` | Serializes to `[first_skill.to_json(), second_skill.to_json()]` |
| `from_json()` | `@staticmethod from_json(data: list) -> SkillGroup` | Deserializes; raises if `len(data) < 2` or data is not a list |

### Context Manager

```python
with SkillGroup() as sg:
    sg.set_first_skill(Skill("skill-a", TeacherA))
    sg.set_second_skill(Skill("skill-b", TeacherB))
```

`__exit__` is a no-op. The context manager exists purely as a syntactic grouping convenience.

### `Agent.add_skill_group()`

```python
agent.add_skill_group(skill_group: SkillGroup)
```

Appends the `SkillGroup` to `agent.skill_groups`. **Does not register the member skills in `agent.skills`.** Children remain invisible to training validation, `get_all_skills()`, and DAG traversal unless you also call `agent.add_skill()` for each child.

---

## Registering Skills Correctly

This is the most common mistake when working with `SkillGroup`. The registration trap is silent: no error, no warning — the member skills simply don't participate in training.

### ❌ Incorrect — children never registered

```python
skill_a = Skill("approach", ApproachTeacher)
skill_b = Skill("target", TargetTeacher)

sg = SkillGroup(skill_a, skill_b)
agent.add_skill_group(sg)

# skill_a and skill_b are in agent.skill_groups ONLY.
# They are NOT in agent.skills.
# get_all_skills() will not return them.
# Training validation in trainer.py will not see them.
# They will be skipped during training.
```

### ✅ Correct — register children first, then the group

```python
skill_a = Skill("approach", ApproachTeacher)
skill_b = Skill("target", TargetTeacher)

# Register each child skill individually first
agent.add_skill(skill_a)
agent.add_skill(skill_b)

# Then register the group
sg = SkillGroup(skill_a, skill_b)
agent.add_skill_group(sg)
```

### ✅ Alternative — use `add_skill()` with a `SkillGroup` instance

The `Agent.add_skill()` method detects a `SkillGroup` argument and:
1. Adds the group to `agent.skill_groups`
2. Calls `agent.add_skill()` on each child — children are correctly registered

```python
skill_a = Skill("approach", ApproachTeacher)
skill_b = Skill("target", TargetTeacher)

agent.add_skill(SkillGroup(skill_a, skill_b))
# Both skills are now in agent.skills AND agent.skill_groups.
```

This is the safest single-call pattern.

---

## Training Behavior

Each skill in a group trains **independently** as a normal standalone skill. The `SkillGroup` structure has no effect on the training loop.

**When `second_skill` (B) is training:**
- A `SkillProcessor` for `first_skill` (A) is built with `is_training=False`.
- A runs its saved/initialized policy as a static oracle — whatever policy it holds at training time.
- A's action is injected into B's observation space under key `first_skill.get_name()`.
- B trains with Ray/RLlib against this enriched observation.

**What this means for reward shaping:**
- Each skill trains on its own reward signal, defined by its own `Teacher.compute_reward()`.
- There is no joint reward, no shared gradient, no coordinated update between group members.
- Skills are trained sequentially (DAG order), not jointly.

**Training-time gotchas:**
- If `first_skill` has not been trained yet when `second_skill` starts training, `first_skill` runs an **untrained policy**. Train `first_skill` to a satisfactory policy before training `second_skill`.
- `first_skill`'s processor runs with `is_training=False` — it will not update during `second_skill`'s training run, even if `first_skill` has a valid checkpoint that could be fine-tuned.

---

## Sensor & Action Space Handling

### Action Space Propagation

`Agent.set_action_space()` propagates the sim action space to all skill group members:

```python
for skill_group in self.skill_groups:
    for skill in skill_group.get_skills():
        skill.set_action_space(action_space)
```

Both member skills receive the same sim action space.

### The `filtered_sensor_space` Requirement

At `SkillProcessor.init()` time, the first skill's action space is injected into the second skill's sensor space under the key `first_skill.get_name()`:

```python
amesa_space[skill_name] = skill_processor.get_action_space()
```

This means `second_skill`'s `composabl_sensor_space` will contain a key equal to the first skill's name. That key passes through `filtered_sensor_space` filtering.

**If `second_skill`'s Teacher does not include the first skill's name in `filtered_sensor_space`, the injected action is silently dropped.** The second skill trains and runs as if the first skill's action was never provided.

```python
class TargetTeacher(SkillTeacher):
    def filtered_sensor_space(self, sensors):
        # ✅ Include first skill's name to receive its action
        return ["position", "velocity", "approach"]  # "approach" = first_skill.get_name()

    def transform_sensors(self, obs, action, goal, sensor_space_list):
        approach_action = obs["approach"]  # first skill's action, available here
        # ...
```

```python
class TargetTeacher(SkillTeacher):
    def filtered_sensor_space(self, sensors):
        # ❌ Missing "approach" — first skill's action silently dropped
        return ["position", "velocity"]
```

---

## Nesting

**Nesting is not supported.** A `SkillGroup` cannot contain another `SkillGroup`. Attempting to construct a chain where a skill participates in two groups (e.g., skill A feeds into group 1, and group 1's output feeds into group 2) does not raise an error — but the nested group processors are silently not built.

**What actually happens:** When a skill runs as part of a group (`for_skill_group=True`), `SkillProcessorBase.init()` sets `self.skill_group_skill_processors = []` and `self.post_skill_group_skill_processor = None`, discarding all sub-group processor assembly. No warning is emitted. The second-level pipeline simply does not execute.

**Hard limit:** There is no 3-way pipeline, no N-ary group, and no `add_skill()` on `SkillGroup` itself.

**Recommended alternatives for chains longer than 2:**

1. **Compose behaviors in Teachers:** Have skill C's Teacher call an explicit function from skill A or B's logic. Less clean but no framework limitation.

2. **Use a `SkillCoach`:** A Coach can orchestrate multiple sub-skills in a structured reward-shaping hierarchy. See the Coach documentation.

3. **Use a `SkillSelector` with shared sensor context:** Pass enriched observations (including prior skill actions) to a selector that routes between downstream specialists.

4. **Structure your pipeline as two independent groups per agent:** If A → B and C → D are each valid pairs, register both groups. They do not compose into a single chain but they operate independently.

---

## Config Parameters

`SkillGroup` has **no config parameters of its own**. Each member skill uses its own `SkillSchema` config, identical to any standalone skill. There is no `SkillGroupOptions`, no group-level `LearningConfig`, and no group-level `ModelConfig`.

| Config | Applies To | Notes |
|---|---|---|
| `SkillSchema` | Each member `Skill` | Same as standalone skill |
| `LearningConfig` | Each member `Skill.config.learning` | Applied per-skill |
| `ModelConfig` | Each member `Skill.config.model` | Applied per-skill |
| `filtered_sensor_space` | Second skill's `SkillTeacher` | Must include `first_skill.get_name()` to expose first skill's action |

**Dead / ignored items in the `SkillGroup` class:**

| Item | Location | Status |
|---|---|---|
| `update_skill_groups()` | `agent.py:1078` | ⚠️ `@deprecated` — logic is broken: matches `sg.get_first_skill().get_name() == skill_group.get_name()`, which compares a skill name to a group name (`sg-[a,b]`). Always mismatches. Never use. |
| `get_node_by_name()` group block | `agent.py:791-792` | ⚠️ Commented out — lookup by group name is dead. Groups cannot be looked up by name. |
| `SkillGroup.id` | `skill_group.py:52` | ⚠️ UUID generated at construction but never read anywhere. Cannot be used for lookup. |
| `skill_group_action` kwarg | `skill_processor_base.py:174` | ⚠️ Passed to `_execute()` but never surfaced to `SkillTeacher` hooks. Teachers cannot access it. |
| `to_list()` | `skill_group.py:90` | Exact duplicate of `get_skills()` — no distinct behavior |

---

## Gotchas

1. **`add_skill_group()` does NOT register children in `agent.skills`.** Calling `agent.add_skill_group(sg)` without prior `agent.add_skill()` calls for each member leaves both skills in `agent.skill_groups` only. They are excluded from `get_all_skills()`, the pre-training validation pass, and DAG traversal. Always call `agent.add_skill(skill_a)` and `agent.add_skill(skill_b)` first — or use `agent.add_skill(SkillGroup(...))` as the single safe alternative.

2. **`get_all_skills()` is blind to skill group members.** The method returns only `skills.values() + skill_selectors.values() + skills_coordinated.values()`. Any code that iterates `get_all_skills()` — including the pre-training validation loop in `trainer.py` — will not see group members unless they are also registered via `add_skill()`.

3. **`filtered_sensor_space` silent drop.** The first skill's action is injected into the enriched sensor dict under key `first_skill.get_name()`. If the second skill's Teacher does not return that key from `filtered_sensor_space()`, the action is silently filtered out. No warning is emitted. The second skill trains and steps as if the pipeline connection does not exist.

4. **Nesting silently no-ops.** If a skill participates in two groups, the `for_skill_group=True` flag causes sub-group processors to be discarded when the skill runs inside the outer group. No error, no warning. Any expected second-level pipeline behavior simply does not occur.

5. **Hard limit of exactly 2 skills.** There is no variadic constructor and no `add_skill()` method on `SkillGroup`. A→B→C chains cannot be expressed as a single group. Workarounds involve nesting groups — but gotcha #4 means that does not work either.

6. **`first_skill` runs with `is_training=False` during second skill training.** When training the second skill, the first skill's processor is built in inference mode. It uses its saved policy — which may be untrained if training order is wrong. Train first skills before second skills.

7. **`None` skills are not caught at construction.** `SkillGroup(None, None)` constructs without error. The `Exception("SkillGroup needs at least 2 skills")` only fires at `await sg.init()`. If `init()` is not called (e.g., in a custom harness), the broken group is never detected.

8. **`update_skill_groups()` is broken and @deprecated.** The method matches `sg.get_first_skill().get_name() == skill_group.get_name()`, comparing a skill name to a group name. This never matches. Do not call this method.

9. **`SkillGroup.id` is never read.** A UUID is generated at construction but no SDK code reads it. Groups cannot be looked up by ID — only by member skill name.

10. **`skill_group_action` dict is not passed to Teacher hooks.** The dict `{first_skill_name: action}` is assembled internally and passed to `_execute()`, but `SkillTeacher` methods (`transform_sensors`, `transform_action`, `compute_reward`, etc.) never receive it as a parameter. The only way the second skill's Teacher can see the first skill's action is through the injected observation key (see gotcha #3).

11. **Multiple groups sharing a skill accumulate processors.** If skill B appears as `second_skill` in two different `SkillGroup` objects, `skill_group_skill_processors` will contain two processors. Both run at step time and both inject their results into the sensor dict. No deduplication or conflict detection exists.

---

## Examples

### Minimal Correct Setup

```python
from composabl_core import Agent, Skill, SkillGroup
from composabl_core.agent.scenario import Scenario

# Define skills normally
approach_skill = Skill("approach", ApproachTeacher, [Scenario({"target_x": 10.0})])
target_skill = Skill("target", TargetTeacher, [Scenario({"target_x": 10.0})])

# Build agent
agent = Agent()
agent.set_sim_sensor_space(sensor_space)
agent.set_action_space(action_space)

# ✅ Register children first, then the group
agent.add_skill(approach_skill)
agent.add_skill(target_skill)
agent.add_skill_group(SkillGroup(approach_skill, target_skill))
```

The second skill's Teacher must include the first skill's name in `filtered_sensor_space`:

```python
class TargetTeacher(SkillTeacher):
    def filtered_sensor_space(self, sensors):
        # "approach" is approach_skill.get_name()
        return ["position", "velocity", "approach"]

    def transform_sensors(self, obs, action, goal, sensor_space_list):
        approach_action = obs.get("approach")  # first skill's action
        # Use approach_action to inform target selection
        return obs

    def compute_reward(self, obs, action, goal):
        # obs["approach"] is available here too
        ...
```

### Alternative Single-Call Registration

```python
# add_skill() with a SkillGroup instance handles both registrations atomically
agent.add_skill(SkillGroup(approach_skill, target_skill))
# Equivalent to:
#   agent.add_skill(approach_skill)
#   agent.add_skill(target_skill)
#   agent.add_skill_group(SkillGroup(approach_skill, target_skill))
```

### Context Manager Style

```python
with SkillGroup() as sg:
    sg.set_first_skill(Skill("approach", ApproachTeacher, scenarios))
    sg.set_second_skill(Skill("target", TargetTeacher, scenarios))

agent.add_skill(sg.get_first_skill())
agent.add_skill(sg.get_second_skill())
agent.add_skill_group(sg)
```

### Common Mistake + Fix

```python
# ❌ Registration trap — children invisible to training
skill_a = Skill("approach", ApproachTeacher, scenarios)
skill_b = Skill("target", TargetTeacher, scenarios)
agent.add_skill_group(SkillGroup(skill_a, skill_b))
# Both skills missing from agent.skills — training validation skips them

# ✅ Fix: register children explicitly before the group
agent.add_skill(skill_a)
agent.add_skill(skill_b)
agent.add_skill_group(SkillGroup(skill_a, skill_b))
```

### Multiple Groups on the Same Agent

```python
skill_a = Skill("sense", SenseTeacher, scenarios)
skill_b = Skill("react", ReactTeacher, scenarios)
skill_c = Skill("plan", PlanTeacher, scenarios)
skill_d = Skill("execute", ExecuteTeacher, scenarios)

agent.add_skill(skill_a)
agent.add_skill(skill_b)
agent.add_skill(skill_c)
agent.add_skill(skill_d)

# Two independent groups — each is a separate pipeline
agent.add_skill_group(SkillGroup(skill_a, skill_b))
agent.add_skill_group(SkillGroup(skill_c, skill_d))
```

Both groups operate independently at step time. They do not compose into a chain.
