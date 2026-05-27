# SkillTeacher Overview

`SkillTeacher` is the base class for ML-trained skills. A teacher defines:

- reward (`compute_reward`)
- success criteria (`compute_success_criteria`)
- termination criteria (`compute_termination`)
- action post-processing (`transform_action`)
- which sensors are used (`filtered_sensor_space`)

Use it when the policy should be learned (PPO), not hard-coded.

## When to use

- **Use SkillTeacher** when behavior is learned from reward.
- **Use SkillController** when behavior is deterministic/rule-based.
- **Use Goals** (`ApproachGoal`, `MaintainGoal`, etc.) when you want reusable reward/termination logic packaged as teacher subclasses.

## Minimal flow

1. Implement a teacher subclass.
2. Attach it to a `Skill("name", MyTeacher, ...)`.
3. Add the skill to an `Agent`.
4. Train with `Trainer.train(...)`.
