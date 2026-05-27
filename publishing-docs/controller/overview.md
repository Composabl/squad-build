# SkillController Overview

`SkillController` is for deterministic, hand-authored policies (non-ML).

Use controllers when action logic is explicit (PID/rules/state machine) and you do not want PPO training.

## Typical use

1. Implement `compute_action(...)`.
2. Add success/termination logic.
3. Attach via `Skill("name", MyController)`.
4. Execute with the same runtime interfaces as teacher-based skills.
