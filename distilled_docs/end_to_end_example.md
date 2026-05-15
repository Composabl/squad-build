## Sim

```python
# file: my_sim.py
import gymnasium as gym
import numpy as np
from amesa_core.networking.sim.server_amesa import ServerAmesa


class MicrogridSim(gym.Env):
    """
    Simulates a microgrid control task.
    Accepts 4-dim actions, returns 6-dim observations.
    """

    def __init__(self, env_init=None):
        self.env_init = env_init or {}
        self.state = None
        self.step_count = 0
        self.max_steps = 360

    async def make(self, env_id: str, env_init: dict):
        """Initialize the simulator with env_init parameters."""
        self.env_init = env_init if env_init else self.env_init
        self.max_steps = self.env_init.get("max_steps", 360)
        self.state = np.zeros(6, dtype=np.float32)
        self.step_count = 0
        return {"id": "microgrid_sim", "max_episode_steps": self.max_steps}

    async def sensor_space_info(self) -> gym.Space:
        """6-dim continuous observations."""
        return gym.spaces.Box(low=-5.0, high=5.0, shape=(6,), dtype=np.float32)

    async def action_space_info(self) -> gym.Space:
        """4-dim continuous actions."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    async def action_space_sample(self):
        """Sample a random action."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)).sample()

    async def reset(self):
        """Reset simulator and return initial observation."""
        self.state = np.array([1.2, 0.55, 1.8, 0.15, 0.7, 6.0], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    async def step(self, action):
        """Execute one step with simple dynamics."""
        solar_kw, soc, load_kw, price, temp, hour = self.state
        charge, discharge, curtail, fan = action

        soc = np.clip(soc + 0.03 * charge - 0.04 * discharge, 0.0, 1.0)
        solar_kw = np.clip(solar_kw - 0.1 * curtail, 0.0, 3.0)
        load_kw = np.clip(load_kw + 0.05 * (fan - 0.2), 0.2, 3.5)
        temp = np.clip(temp + 0.08 * load_kw - 0.2 * fan, 0.0, 2.0)
        hour = (hour + 1.0) % 24.0

        self.state = np.array([solar_kw, soc, load_kw, price, temp, hour], dtype=np.float32)
        self.step_count += 1

        grid_mismatch = load_kw - solar_kw - 1.2 * soc
        reward = float(-abs(grid_mismatch) - 0.1 * temp)
        terminated = self.step_count >= self.max_steps

        return self.state, reward, terminated, False, {}

    async def close(self):
        pass

    async def set_scenario(self, scenario: dict):
        """Set scenario parameters (initial conditions, reward config, etc.)."""
        if scenario:
            self.state = np.array([
                scenario.get("solar_kw", 1.0),
                scenario.get("battery_soc", 0.5),
                scenario.get("load_kw", 1.6),
                scenario.get("grid_price", 0.2),
                scenario.get("inverter_temp", 0.6),
                scenario.get("hour_of_day", 8.0),
            ], dtype=np.float32)

    async def get_scenario(self):
        return None


# Wrap in ServerAmesa
from amesa_core.networking.sim.server_amesa import ServerAmesa


class ServerImpl(ServerAmesa):
    def __init__(self, *args, **kwargs):
        self.env_init = kwargs.get("env_init", {})
        self.env = MicrogridSim(self.env_init)

    async def make(self, env_id: str, env_init: dict):
        return await self.env.make(env_id, env_init)

    async def sensor_space_info(self) -> gym.Space:
        return await self.env.sensor_space_info()

    async def action_space_info(self) -> gym.Space:
        return await self.env.action_space_info()

    async def action_space_sample(self):
        return await self.env.action_space_sample()

    async def reset(self):
        return await self.env.reset()

    async def step(self, action):
        return await self.env.step(action)

    async def close(self):
        await self.env.close()

    async def set_scenario(self, scenario):
        await self.env.set_scenario(scenario)

    async def get_scenario(self):
        return await self.env.get_scenario()

    async def get_render(self):
        return None
```

## Perceptor (Optional)

```python
# file: my_perceptor.py
from amesa_core.agent.perceptor.perceptor import Perceptor
from amesa_core.agent.perceptor.perceptor_impl import PerceptorImpl


class LoadDeltaPerceptor(PerceptorImpl):
    """Computes load delta between steps."""

    def __init__(self):
        self.last_load = None

    async def compute(self, obs_spec, obs):
        current_load = obs[2]  # load_kw is index 2

        if self.last_load is None:
            delta = 0.0
        else:
            delta = current_load - self.last_load

        self.last_load = current_load
        return {"load_delta": delta}

    def filtered_sensor_space(self, obs):
        return []

    def __await__(self):
        async def _noop():
            return self

        return _noop().__await__()


LOAD_DELTA_PERCEPTOR = Perceptor(
    "load-delta",
    LoadDeltaPerceptor,
    "Computes load delta between steps.",
)
```

## Teacher Skill

```python
# file: my_teacher.py
from amesa_core.agent.skill.skill_teacher import SkillTeacher
from amesa_core.spaces import Box
from typing import Dict, List


class GridStabilityTeacher(SkillTeacher):
    """Learns to stabilize grid balance and keep SOC healthy."""

    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,))

    async def compute_reward(self, transformed_sensors: Dict, action, sim_reward: float) -> float:
        solar_kw = transformed_sensors.get("solar_kw", 0.0)
        load_kw = transformed_sensors.get("load_kw", 0.0)
        soc = transformed_sensors.get("battery_soc", 0.5)
        mismatch = abs(load_kw - solar_kw - soc)
        reward = -mismatch - 0.2 * abs(soc - 0.6)
        return float(reward)

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        soc = transformed_sensors.get("battery_soc", 0.5)
        return soc < 0.05 or soc > 0.95

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        mismatch = abs(transformed_sensors.get("load_kw", 0.0) - transformed_sensors.get("solar_kw", 0.0))
        return mismatch < 0.1

    async def filtered_sensor_space(self) -> List[str]:
        return ["solar_kw", "battery_soc", "load_kw", "grid_price", "inverter_temp", "hour_of_day", "load_delta"]

    async def transform_sensors(self, sensors, action):
        return sensors

    async def transform_action(self, transformed_sensors: Dict, action):
        return action if action is not None else self.action_space.sample()
```

## Building the Agent

```python
# file: build_agent.py
from amesa_core.agent.agent import Agent
from amesa_core.agent.sensors.sensor import Sensor
from amesa_core.agent.skill.skill import Skill
from my_teacher import GridStabilityTeacher
from my_perceptor import LOAD_DELTA_PERCEPTOR


def build_agent():
    agent = Agent()

    # Add sensors
    agent.add_sensors([
        Sensor("solar_kw", "Solar production (kW)"),
        Sensor("battery_soc", "Battery state of charge"),
        Sensor("load_kw", "Total load (kW)"),
        Sensor("grid_price", "Grid price index"),
        Sensor("inverter_temp", "Inverter temperature"),
        Sensor("hour_of_day", "Hour of day"),
    ])

    # Add perceptor
    agent.add_perceptor(LOAD_DELTA_PERCEPTOR)

    # Add skill
    skill = Skill(
        "grid-stability",
        GridStabilityTeacher,
        training_cycles=6,
    )

    # Add scenario
    summer_peak = {
        "scenario_name": "summer_peak",
        "solar_kw": [1.5, 2.8],
        "battery_soc": [0.35, 0.75],
        "load_kw": [1.8, 3.0],
        "grid_price": [0.1, 0.35],
        "inverter_temp": [0.5, 1.2],
        "hour_of_day": [11.0, 18.0],
    }
    skill.add_scenario(summer_peak)

    agent.add_skill(skill)
    return agent
```

### Running Training (V2 Event-Based)

```python
# file: run_training.py
import os
import subprocess
import time
from amesa_train.trainer import Trainer
from build_agent import build_agent

REDIS_URL = "redis://localhost:6380"
SIM_IMAGE = "my-sim-microgrid:latest"


def start_redis():
    """Start a Docker Redis container."""
    result = subprocess.run(
        ["docker", "run", "-d", "-p", "6380:6379", "redis:7-alpine"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def wait_redis(redis_url, retries=20):
    """Wait for Redis to be ready."""
    import redis
    for i in range(retries):
        try:
            client = redis.from_url(redis_url)
            client.ping()
            client.close()
            return
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(0.5)


def main():
    # Set up environment
    os.environ.setdefault("AMESA_LICENSE", "YOUR_LICENSE_KEY")
    os.environ.setdefault("AMESA_EULA_AGREED", "1")

    # Start Redis
    print("Starting Redis…")
    container_id = start_redis()
    wait_redis(REDIS_URL)

    try:
        # Build agent
        print("Building agent…")
        agent = build_agent()

        # Configure V2 training
        config = {
            "target": {
                "type": "v2",
                "v2": {
                    "redis_url": REDIS_URL,
                    "enable_sim_group": True,
                    "sim_node_local": True,
                    "sim_image": SIM_IMAGE,
                    "perceptor_node_local": True,
                    "skill_node_local": True,
                    "episode_manager_local": True,
                    "enable_remote_skill": False,
                    "enable_auto_scale": False,
                }
            }
        }

        # Train
        print("Starting training…")
        trainer = Trainer(config)
        try:
            trainer.train(agent, train_cycles=8)
            print("✅ Training complete!")
        finally:
            trainer.close()

    finally:
        # Clean up Redis
        print("Stopping Redis…")
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


if __name__ == "__main__":
    main()
```

### Running Training (Local Ray-Based)

For comparison, here's the same agent with local (Ray) training using a local gRPC sim:

```python
# file: run_training_local.py
import os
from amesa_train.trainer import Trainer
from build_agent import build_agent


def main():
    os.environ.setdefault("AMESA_LICENSE", "YOUR_LICENSE_KEY")
    os.environ.setdefault("AMESA_EULA_AGREED", "1")

    agent = build_agent()

    # Use local Ray-based training pointing to a gRPC sim at localhost:1340
    config = {
        "target": {
            "local": {
                "address": "localhost:1340"
            }
        }
    }

    trainer = Trainer(config)
    try:
        trainer.train(agent, train_cycles=8)
        print("✅ Training complete!")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
```

### What This Example Covers

1. **Sim** — Microgrid sim implementing ServerAmesa
2. **Perceptor** — Computes load delta (derived variable)
3. **Teacher** — Stabilizes grid balance with DRL training
4. **Agent** — Combines sensors, perceptor, skill, and scenarios
5. **V2 Training** — Event-driven orchestration with Redis + EventSimProcessor + EpisodeManager
6. **Local Training** — Ray-based alternative for local development
7. **Deployment** — Both examples ready for `trainer.train()` with minimal setup
