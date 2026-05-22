from __future__ import annotations

import gymnasium as gym
import numpy as np

from v1.full_example.sim.scenarios import materialize_scenario


class GreenhouseSim(gym.Env):
    """Simple greenhouse climate simulator."""

    metadata = {"render_modes": []}
    SENSOR_KEYS = [
        "air_temp",
        "target_temp",
        "ambient_temp",
        "humidity",
        "target_humidity",
        "ambient_humidity",
    ]

    def __init__(self, env_init=None):
        self.env_init = env_init or {}
        self.scenario = {}
        self.step_count = 0
        self.max_steps = int(self.env_init.get("max_steps", 240))
        self.state = None
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "air_temp": gym.spaces.Box(low=0.0, high=50.0, shape=(), dtype=np.float32),
                "target_temp": gym.spaces.Box(low=0.0, high=50.0, shape=(), dtype=np.float32),
                "ambient_temp": gym.spaces.Box(low=0.0, high=50.0, shape=(), dtype=np.float32),
                "humidity": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "target_humidity": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
                "ambient_humidity": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            }
        )

    def make(self, env_id: str, env_init: dict):
        self.env_init = env_init if env_init else self.env_init
        self.max_steps = int(self.env_init.get("max_steps", self.max_steps))
        self.step_count = 0
        self._apply_scenario()
        return {"id": "greenhouse_sim", "max_episode_steps": self.max_steps}

    def sensor_space_info(self):
        return self.observation_space

    def action_space_info(self):
        return self.action_space

    def action_space_sample(self):
        return self.action_space.sample()

    def reset(self):
        self.step_count = 0
        self._apply_scenario()
        return dict(self.state), {}

    def step(self, action):
        action_value = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        temp = float(self.state["air_temp"])
        target_temp = float(self.state["target_temp"])
        ambient_temp = float(self.state["ambient_temp"])
        humidity = float(self.state["humidity"])
        target_humidity = float(self.state["target_humidity"])
        ambient_humidity = float(self.state["ambient_humidity"])

        temp += 0.15 * (ambient_temp - temp) + 1.2 * action_value
        humidity += 0.05 * (ambient_humidity - humidity) - 0.03 * action_value

        temp = float(np.clip(temp, 0.0, 50.0))
        humidity = float(np.clip(humidity, 0.0, 1.0))

        reward = -abs(temp - target_temp) - 4.0 * abs(humidity - target_humidity) - 0.02 * (
            action_value**2
        )

        self.step_count += 1
        terminated = self.step_count >= self.max_steps

        self.state = {
            "air_temp": float(temp),
            "target_temp": float(target_temp),
            "ambient_temp": float(ambient_temp),
            "humidity": float(humidity),
            "target_humidity": float(target_humidity),
            "ambient_humidity": float(ambient_humidity),
        }
        return dict(self.state), float(reward), terminated, False, {}

    def close(self):
        return None

    def set_scenario(self, scenario):
        if scenario:
            self.scenario = scenario.sample() if hasattr(scenario, "sample") else dict(scenario)
        else:
            self.scenario = {}
        self.max_steps = int(self.scenario.get("max_steps", self.max_steps))
        self._apply_scenario()

    def get_scenario(self):
        return self.scenario

    def get_render(self):
        return None

    def _scenario_value(self, key: str, default: float) -> float:
        if key in self.scenario:
            return float(self.scenario[key])
        return float(self.env_init.get(key, default))

    def _apply_scenario(self):
        temp = self._scenario_value("initial_temp", 20.0)
        target_temp = self._scenario_value("target_temp", 22.0)
        ambient_temp = self._scenario_value("ambient_temp", 15.0)
        humidity = self._scenario_value("initial_humidity", 0.5)
        target_humidity = self._scenario_value("target_humidity", 0.55)
        ambient_humidity = self._scenario_value("ambient_humidity", 0.45)
        self.state = {
            "air_temp": float(temp),
            "target_temp": float(target_temp),
            "ambient_temp": float(ambient_temp),
            "humidity": float(humidity),
            "target_humidity": float(target_humidity),
            "ambient_humidity": float(ambient_humidity),
        }
