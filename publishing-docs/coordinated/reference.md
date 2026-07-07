# Coordinated Interface

`AgentCoach` defines reward/success/termination across a coordinated group of child agents.

## Full scaffold

```python
from amesa_core.orchestration.agent.agent_coach import AgentCoach
from typing import Dict

class MyCoach(AgentCoach):
    # required
    async def compute_reward(self, transformed_sensors: Dict, action, sim_reward):
        """Return per-child rewards for coordinated learning.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Dict of child actions keyed by child agent name.
        :param sim_reward: Simulator reward for the current step.
        :returns: Dict mapping child agent names to scalar reward values.
        :rtype: dict[str, float]
        :note: Reward dict keys must exactly match child agent names in the
            coordinated set or population.
        """
        return {
            "striker": float(transformed_sensors.get("goal_proximity", 0.0)),
            "defender": float(-transformed_sensors.get("opponent_proximity", 0.0)),
        }

    # required
    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        """Define success for the coordinated group as a whole.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Dict of child actions keyed by child agent name.
        :returns: ``True`` when the coordinated objective is complete.
        :rtype: bool
        """
        return bool(transformed_sensors.get("score", 0) > 5)

    # required
    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        """Define coordinated termination logic for the full unit.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Dict of child actions keyed by child agent name.
        :returns: ``True`` when the coordinated episode should stop.
        :rtype: bool
        """
        return bool(transformed_sensors.get("time_elapsed", 0) > 300)

    # optional
    async def transform_sensors(self, sensors, action) -> Dict:
        """Build transformed shared features from raw sensors.

        Normalises raw positional sensors into ``[0, 1]`` proximity values
        that both children can share without unit-specific scaling.

        :param sensors: Raw sensor dict from the environment.
        :param action: Dict of child actions keyed by child agent name.
        :returns: Transformed sensor dict.
        :rtype: Dict
        """
        field_length = 100.0
        return {
            **sensors,
            "goal_proximity":     float(sensors.get("ball_x", 0.0)) / field_length,
            "opponent_proximity": float(sensors.get("opp_x",  0.0)) / field_length,
        }

    # optional
    async def transform_action(self, transformed_sensors: Dict, action):
        """Rewrite child-action dict before env application.

        Clips the striker's continuous action to ``[-1, 1]`` to prevent
        out-of-range commands from reaching the simulator.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Dict of child actions keyed by child agent name.
        :returns: Rewritten per-child action dict.
        :rtype: dict
        """
        clipped = dict(action)
        if "striker" in clipped:
            clipped["striker"] = [max(-1.0, min(1.0, float(v))) for v in clipped["striker"]]
        return clipped

    # optional
    async def compute_action_mask(self, transformed_sensors: Dict, action):
        """Mask invalid child actions at runtime.

        Disables all striker actions when the ball is outside striking range,
        forcing the defender to act until the ball is in play.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Dict of child actions keyed by child agent name.
        :returns: Dict of per-child masks, or ``None`` per child for no mask.
        :rtype: dict
        """
        in_range = float(transformed_sensors.get("goal_proximity", 0.0)) > 0.2
        return {
            "striker":  None if in_range else [False, False],
            "defender": None,
        }
```
