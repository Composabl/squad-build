# CoordinatedGoal

`CoordinatedGoal` combines multiple goals with a coordination strategy.

## Constructor concepts

- `goals: List[Goal]`
- `goals_coordination_strategy` (`AND`, `OR`, `THEN`)
- optional `weights`

## Behavior highlights

- Success logic is primarily driven by `ApproachGoal` members.
- Termination logic is primarily driven by `AvoidGoal` members.
- `weights` length must match `len(goals)` when provided.
- Goal order matters for `THEN`.

## `can_terminate()` guidance

`ApproachGoal` and `AvoidGoal` are safe as leading steps in staged chains; `MaintainGoal`, `MaximizeGoal`, and `MinimizeGoal` generally are not self-terminating.
