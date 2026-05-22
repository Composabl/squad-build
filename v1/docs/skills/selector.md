# SkillSelector

`SkillSelector` chooses which child skill runs at each step. It is itself backed by a `SkillTeacher` (learned selection) or `SkillController` (rule-based selection).

## Constructor

```python
SkillSelector(
    name: str,
    impl_cls: type[SkillTeacher | SkillController],
    config: dict = {},
    **kwargs,
)
```

### kwargs

| kwarg | Effect |
|---|---|
| `children` | List of child `Skill` objects or name strings |
| `fixed_order` | `True` → children run in declared order (default: variable) |
| `repeat` | `True` → selector loops after all children complete |
| `training_cycles` | Training cycles for the selector's own policy |
| `train_batch_size` | PPO batch size |
| `needs_warmup` | `True` → selector waits for children to pre-train |

## Adding children

```python
from amesa_core.agent.skill.skill_selector import SkillSelector
from amesa_core.agent.skill.skill import Skill

selector = SkillSelector("my-selector", MyTeacher)
selector.add_child(skill_a)
selector.add_child(skill_b)

# Or via kwarg at construction
selector = SkillSelector("my-selector", MyTeacher, children=[skill_a, skill_b])
```

## Adding to the agent

```python
agent.add_selector_skill(selector, fixed_order=True, repeat=False)
```

`fixed_order` and `repeat` can also be set here (they override the selector's own config).

## Context manager style

```python
with SkillSelector("my-selector", MyTeacher) as ss:
    ss.add_child(Skill("child-1", ChildTeacher))
    ss.add_child(Skill("child-2", ChildTeacher))
    agent.add_selector_skill(ss)
```

## Selector as a controller

If the selection logic is deterministic (e.g., rule-based routing), pass a `SkillController` subclass:

```python
SkillSelector("my-selector", MyRoutingController)
```

---

## ⚠️ Quirks

**Children required before training** — A selector with no children will raise an exception when training initializes. Always add at least one child.

**Children are stored by name reference** — Internally, children are stored as name strings in the serialized config. The `Skill` objects you pass are registered on the `Agent`, not held by the selector. Make sure every child is also added to the agent via `agent.add_skill()` (the declarative API handles this automatically).

**`fixed_order=True` + `repeat=False`** — The selector runs each child in order exactly once per episode, then terminates. This is the most common pattern for curriculum-style training.
