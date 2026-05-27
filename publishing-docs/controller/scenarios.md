# SkillController Scenarios

Controllers can use the same scenario mechanism as teacher-based skills.

```python
skill = Skill("my-controller-skill", MyController)
skill.add_scenario({"initial_pos": [0.0, 5.0], "target": 3.0})
```

Use scenarios to vary initial conditions and evaluate controller robustness across conditions.
