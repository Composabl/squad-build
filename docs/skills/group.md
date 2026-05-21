# SkillGroup

`SkillGroup` connects two skills in a pipeline: the output (action) of the **first** skill is forwarded as input to the **second** skill. Use it to chain behaviors where one skill's decision feeds into the next.

## Import

```python
from amesa_core.agent.skill.skill_group import SkillGroup
from amesa_core.agent.skill.skill import Skill
```

## Constructor

```python
SkillGroup(
    first_skill: Skill = None,
    second_skill: Skill = None,
)
```

## Usage

```python
skill_a = Skill("preprocessor", PreprocessorTeacher)
skill_b = Skill("executor", ExecutorTeacher)

group = SkillGroup(skill_a, skill_b)
```

## Context manager style

```python
with SkillGroup() as sg:
    sg.set_first_skill(Skill("preprocessor", PreprocessorTeacher))
    sg.set_second_skill(Skill("executor", ExecutorTeacher))
```

## Adding to a selector

A `SkillGroup` is typically added as a child of a `SkillSelector`:

```python
selector.add_child(group)
# or via agent directly:
agent.add_selector_skill(selector, [skill_a_standalone, group, selector_2])
```

## Methods

| Method                    | Description                       |
| ------------------------- | --------------------------------- |
| `set_first_skill(skill)`  | Set the first (upstream) skill    |
| `set_second_skill(skill)` | Set the second (downstream) skill |
| `get_first_skill()`       | Return first skill                |
| `get_second_skill()`      | Return second skill               |
| `get_skills()`            | Return `[first, second]`          |

---

## ⚠️ Quirks

**Exactly two skills** — A `SkillGroup` always has exactly two skills. There is no support for chaining 3+ skills in a single group; nest groups if needed.

**Both skills required** — If either `first_skill` or `second_skill` is `None` when `init()` is called, an exception is raised.

**Name is auto-generated** — `SkillGroup.get_name()` returns `"sg-[first_name,second_name]"`. You cannot set a custom name.
