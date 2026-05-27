# SkillTeacher Quirks

- `compute_success_criteria` can end episodes immediately.
- `compute_termination` defaults to `False` (unlike controllers where termination is required).
- Missing sensor keys plus permissive defaults can create false success.
- `self.action_space` in teacher code does not automatically become trainer action space.
- For trainer action-space override, use `get_custom_action_space()`.
- `compute_reward` must return a Python `float` (not numpy scalar/array).
- `filtered_sensor_space()` returns names (`list[str]`), not `Sensor` objects.
