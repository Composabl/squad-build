from __future__ import annotations

from typing import Dict, List

from amesa_core.agent.skill.skill_teacher import SkillTeacher
from amesa_core.spaces import Box


class GreenhouseTeacher(SkillTeacher):
    """Learned controller for greenhouse climate regulation."""

    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))

    async def compute_reward(self, transformed_sensors: Dict, action, sim_reward: float) -> float:
        temp_error = transformed_sensors.get("temp_error")
        if temp_error is None:
            temp_error = transformed_sensors.get("air_temp", 0.0) - transformed_sensors.get(
                "target_temp", 0.0
            )

        humidity_error = transformed_sensors.get("humidity_error")
        if humidity_error is None:
            humidity_error = transformed_sensors.get("humidity", 0.0) - transformed_sensors.get(
                "target_humidity", 0.0
            )

        action_value = float(action[0]) if action is not None else 0.0
        reward = -abs(temp_error) - 0.5 * abs(humidity_error) - 0.05 * (action_value**2)
        return float(reward)

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        temp = float(transformed_sensors.get("air_temp", 0.0))
        humidity = float(transformed_sensors.get("humidity", 0.0))
        return temp < 0.0 or temp > 50.0 or humidity < 0.0 or humidity > 1.0

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        temp_error = abs(float(transformed_sensors.get("temp_error", 0.0)))
        humidity_error = abs(float(transformed_sensors.get("humidity_error", 0.0)))
        return temp_error < 0.3 and humidity_error < 0.05

    async def filtered_sensor_space(self) -> List[str]:
        return [
            "air_temp",
            "target_temp",
            "ambient_temp",
            "humidity",
            "target_humidity",
            "ambient_humidity",
            "temp_error",
            "humidity_error",
            "temp_delta",
        ]

    async def transform_sensors(self, sensors, action):
        return sensors

    async def transform_action(self, transformed_sensors: Dict, action):
        return action if action is not None else self.action_space.sample()
