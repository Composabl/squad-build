# Goal Base Interface

All goal types inherit shared constructor parameters from `Goal`.

## Core parameters

- `sensor`
- `name`
- `target`
- `tolerance`
- `stop_value`
- `stop_steps`
- `boundary_left`
- `boundary_right`
- `boundary_is_relative`
- `scale`

## Shared helper methods

- `is_in_boundary(...)`
- `compute_error(...)`
- `get_sensor_value(...)`
- `get_target_value(...)`

## Important behaviors

- `target=None` requires both boundaries and resolves to midpoint.
- `steps_taken` increments on reward calls and is not auto-reset per episode.
