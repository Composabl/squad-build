### Creating and Configuring the Trainer

#### Basic Setup

```python
from amesa_train.trainer import Trainer
from amesa_core.agent.agent import Agent

# Create your agent
agent = Agent()
agent.add_sensors([...])
agent.add_skill(...)

# Define trainer configuration
config = {
    "target": {
        "v2": {
            "redis_url": "redis://localhost:6379",
            "sim_image": "...", # Choose local built Docker sim image
            "initial_replicas": 4,
            "num_episode_managers": 2,
        }
    }
}

# Create the trainer
trainer = Trainer(config)
```

#### Trainer Target Configuration

The `config["target"]` field is a `TrainerTargetConfig` model with optional fields for each compute platform. Exactly one must be set:

| Field               | Type                                     | Contains           | Description                                                                        |
| ------------------- | ---------------------------------------- | ------------------ | ---------------------------------------------------------------------------------- |
| `target.local`      | TrainerTargetLocalConfig (optional)      | `address: str`     | Local Ray-based training. Address is sim gRPC endpoint (e.g., `"localhost:1337"`). |
| `target.v2`         | TrainerTargetV2Config (optional)         | Event-based config | Event-driven training using Redis Streams + episode manager orchestration.         |
| `target.docker`     | TrainerTargetDockerConfig (optional)     | Docker sim config  | Remote sim in Docker.                                                              |
| `target.kubernetes` | TrainerTargetKubernetesConfig (optional) | K8s config         | Sims on Kubernetes.                                                                |

**Exactly one target type must be specified.** The SDK automatically detects which via the `type` computed property.

#### V2-Specific Configuration (Event-Based Training)

When `target.v2` is set, use the `TrainerTargetV2Config` sub-config:

```python
config = {
    "target": {
        "v2": {
            "redis_url": "redis://localhost:6379",
            "sim_image": "...", # Choose local built Docker sim image
            "initial_replicas": 8,        # Number of sim workers
            "num_episode_managers": 4,    # Number of episode managers
            "enable_ppo_training": True,
            "ppo_training_samples": 4000,
            "enable_evaluation": False,
            "enable_historian": False,
            "enable_auto_scale": False,
        },
    },
}
```

**Key V2 fields:**

| Field                  | Type | Default                    | Description                                                                                                  |
| ---------------------- | ---- | -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `redis_url`            | str  | `"redis://localhost:6379"` | Redis connection URL for event streams. Required.                                                            |
| `sim_image`            | str  | None                       | Docker image for the simulator.                                                                              |
| `initial_replicas`     | int  | 1                          | Number of simulator instances to spawn. Should match expected concurrency.                                   |
| `num_episode_managers` | int  | 1                          | Number of episode managers. Training samples are sharded across them. Must divide `initial_replicas` evenly. |
| `enable_ppo_training`  | bool | True                       | Whether to run PPO training.                                                                                 |
| `ppo_training_samples` | int  | 4000                       | Total samples collected before triggering PPO updates.                                                       |
| `enable_evaluation`    | bool | False                      | Whether to run post-training evaluation.                                                                     |
| `enable_historian`     | bool | False                      | Whether to enable telemetry historian.                                                                       |
| `enable_auto_scale`    | bool | False                      | Whether to auto-scale sim/skill replicas during training.                                                    |

### Training Entry Point

#### Method: `trainer.train(agent, train_cycles)`

```python
trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=10)
    print("✅ Training complete!")
except Exception as e:
    print(f"❌ Training failed: {e}")
finally:
    trainer.close()
```

**Parameters:**

- `agent` — The Agent instance to train. Skills in `agent.skills` are trained sequentially or as coordinated groups.
- `train_cycles` — Number of training cycles (e.g., policy update iterations) to run. Each cycle collects experience and updates the policy.

**Behavior:**

1. **Initialization** — Trainer validates sim structure, initializes Ray (local/docker/k8s), or Redis streams (V2), and loads agent skills.
2. **Experience Collection** — Skills run episodes in the simulator, gathering transitions (obs, action, reward, next_obs).
3. **Policy Update** — PPO or other algorithm updates the policy using collected transitions.
4. **Checkpoint Saving** — Policy weights are saved to `output_dir / skill_name / policy_checkpoint_*`.
5. **Repeat** — Steps 2–4 repeat for each training cycle until `train_cycles` is reached.

#### Example: Training a Simple Agent

```python
from amesa_train.trainer import Trainer
from amesa_core.agent.agent import Agent
from amesa_core.agent.sensors.sensor import Sensor
from amesa_core.agent.skill.skill import Skill
from amesa_core.agent.skill.skill_teacher import SkillTeacher
from amesa_core.spaces import Box

# 1. Create a simple teacher
class CartpoleBalanceTeacher(SkillTeacher):
    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))

    async def compute_reward(self, sensors: dict, action, sim_reward: float) -> float:
        # Reward is negative distance from upright pole angle (0 rad)
        theta = sensors.get("theta", 0.0)
        return 1.0 - abs(theta) / 3.14159

    async def compute_termination(self, sensors: dict, action) -> bool:
        # Terminate if pole falls > 20 degrees from vertical
        theta = sensors.get("theta", 0.0)
        return abs(theta) > 0.349  # ~20 degrees

    async def compute_success_criteria(self, sensors: dict, action) -> bool:
        # Success when average episode reward > 0.95
        return False  # Let trainer decide

    async def filtered_sensor_space(self):
        return ["theta", "theta_dot"]

# 2. Build the agent
agent = Agent()
agent.add_sensors([
    Sensor("theta", "Pole angle (radians)"),
    Sensor("theta_dot", "Pole angle velocity"),
])

skill = Skill(
    "balance-pole",
    CartpoleBalanceTeacher,
    training_cycles=5
)
agent.add_skill(skill)

# 3. Configure and run training
config = {
    "target": {
        "v2": {
            "redis_url": "redis://localhost:6379",
            "sim_image": "...",
            "initial_replicas": 4,
            "num_episode_managers": 2,
        }
    }
}

trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=5)
finally:
    trainer.close()
```
