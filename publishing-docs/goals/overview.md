# Goals Overview

Goals are reusable `SkillTeacher` subclasses for standardized reward/success/termination logic.

Common goals:

- `ApproachGoal`
- `AvoidGoal`
- `MaintainGoal`
- `MaximizeGoal`
- `MinimizeGoal`
- `CoordinatedGoal` (composition wrapper)

Goals are best used by subclassing and attaching them to a `Skill` as the teacher implementation.
