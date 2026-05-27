# Goal Types

## `ApproachGoal`

Drive sensor values toward target; can report success when within tolerance/boundary.

## `AvoidGoal`

Penalize risky regions; usually contributes termination behavior (never success).

## `MaintainGoal`

Reward staying near a target band; does not inherently succeed.

## `MaximizeGoal`

Reward larger values for a target sensor.

## `MinimizeGoal`

Reward smaller values for a target sensor.

## Choosing quickly

- Reach setpoint: `ApproachGoal`
- Stay in safe corridor: `MaintainGoal` + `AvoidGoal`
- Push metric up/down continuously: `MaximizeGoal` / `MinimizeGoal`
