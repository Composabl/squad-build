# SkillController Interface

`SkillController` is the deterministic/rule-based interface for a `Skill`.

## Full scaffold

```python
from amesa_core.agent.skill.skill_controller import SkillController
from typing import Dict

class MyController(SkillController):
    async def compute_action(self, transformed_sensors: Dict, action):
        """Produce the simulator action from transformed sensors.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Previous action (useful for stateful controllers).
        :returns: Action consumable by the sim action space.
        :raises NotImplementedError: if not overridden in the subclass.
        """
        error = float(transformed_sensors.get("error", 0.0))
        control = -0.5 * error
        return [float(control)]

    async def filtered_sensor_space(self):
        """Declare which sensor keys this controller reads.

        :returns: List of sensor key strings.
        :rtype: list[str]
        :raises NotImplementedError: if not overridden in the subclass.
        """
        return ["value", "target", "error"]

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        """Define when the controlled behavior is successful.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Current action taken by the agent.
        :returns: ``True`` when the success condition is met.
        :rtype: bool
        :raises NotImplementedError: if not overridden in the subclass.
        """
        return abs(float(transformed_sensors.get("error", 1.0))) < 0.05

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        """Define when the episode should terminate.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Current action taken by the agent.
        :returns: ``True`` when the episode should end.
        :rtype: bool
        :raises NotImplementedError: if not overridden in the subclass.
        """
        return float(transformed_sensors.get("value", 0.0)) > 100.0

    async def transform_sensors(self, sensors) -> Dict:
        """Derive controller-specific features from raw sensors.

        Computes ``error`` as the signed difference between ``value`` and
        ``target`` so downstream control logic only needs to read a single key.

        :param sensors: Raw sensor dict from the environment.
        :returns: Transformed sensor dict with an added ``error`` key.
        :rtype: Dict
        """
        value = float(sensors.get("value", 0.0))
        target = float(sensors.get("target", 0.0))
        return {**sensors, "error": value - target}
```

## Methods and intended use

### `compute_action(self, transformed_sensors, action)` (required)

Computes the control output applied to the simulator at each step. Use it for deterministic control laws, rule engines, and stateful policy logic.

### `filtered_sensor_space(self) -> list[str]` (required)

Declares the controller's sensor dependency set. Use it to explicitly limit observations to what control logic needs.

### `compute_success_criteria(self, transformed_sensors, action) -> bool` (required)

Defines success conditions for the controlled objective. Use it for tolerance bands, state goals, and composite completion checks.

### `compute_termination(self, transformed_sensors, action) -> bool` (required)

Defines termination conditions (for example safety limits, dead-ends, or max bounds). This is required for controllers.

### `transform_sensors(self, sensors) -> dict` (optional)

Preprocesses raw sensors before controller logic executes. Use it for normalization, derived features, and compact state construction.

## Method contracts

- `transformed_sensors`: post-`transform_sensors` sensor dict.
- `action`: previous action (useful for stateful controllers).
- `compute_action`: must return a value consumable by the sim action space.
- `transform_sensors` takes only `sensors` (no `action` argument).
