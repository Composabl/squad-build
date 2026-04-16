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
        "type": "v2",
        "v2": {
            "redis_url": "redis://localhost:6379",
            "enable_sim_group": True,
            "sim_node_local": True,
            "sim_image": "...", # Choose local built Docker sim image
            "perceptor_node_local": True,
            "skill_node_local": True,
            "episode_manager_local": True,
            "enable_remote_skill": False,
            "enable_auto_scale": False,
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

**Exactly one target type must be specified.** For V2, set `target.type = "v2"` alongside the `target.v2` block.

#### V2-Specific Configuration (Event-Based Training)

When `target.v2` is set, use the `TrainerTargetV2Config` sub-config:

```python
config = {
    "target": {
        "type": "v2",
        "v2": {
            "redis_url": "redis://localhost:6379",
            "enable_sim_group": True,
            "sim_node_local": True,
            "sim_image": "...", # Choose local built Docker sim image
            "perceptor_node_local": True,
            "skill_node_local": True,
            "episode_manager_local": True,
            "enable_remote_skill": False,
            "enable_auto_scale": False,
        },
    },
}
```

**Key V2 fields:**

| Field                   | Type | Default                    | Description                                                             |
| ----------------------- | ---- | -------------------------- | ----------------------------------------------------------------------- |
| `redis_url`             | str  | `"redis://localhost:6379"` | Redis connection URL for event streams. Required.                       |
| `sim_image`             | str  | None                       | Docker image for the simulator.                                         |
| `enable_sim_group`      | bool | True                       | Whether to enable sim groups for v2 orchestration.                      |
| `sim_node_local`        | bool | False                      | Run the sim node locally instead of using a remote worker.              |
| `perceptor_node_local`  | bool | False                      | Run the perceptor node locally.                                         |
| `skill_node_local`      | bool | False                      | Run the skill node locally.                                             |
| `episode_manager_local` | bool | False                      | Run the episode manager locally.                                        |
| `enable_remote_skill`   | bool | False                      | Allow skills to run on remote workers rather than only local execution. |
| `enable_auto_scale`     | bool | False                      | Whether to auto-scale sim/skill replicas during training.               |

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
If your sim returns dict observations, add mapping lambdas so each sensor extracts its value (e.g., `Sensor("theta", "...", lambda sensors: sensors["theta"])`).

skill = Skill(
    "balance-pole",
    CartpoleBalanceTeacher,
    training_cycles=5
)
agent.add_skill(skill)

# 3. Configure and run training
config = {
    "target": {
        "type": "v2",
        "v2": {
            "redis_url": "redis://localhost:6379",
            "enable_sim_group": True,
            "sim_node_local": True,
            "sim_image": "...",
            "perceptor_node_local": True,
            "skill_node_local": True,
            "episode_manager_local": True,
            "enable_remote_skill": False,
            "enable_auto_scale": False,
        }
    }
}

trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=5)
finally:
    trainer.close()
```

When running v2 locally, keep `enable_sim_group=True` and set local node flags (`sim_node_local`, `perceptor_node_local`, `skill_node_local`, `episode_manager_local`) so streams and consumer groups are created on the same machine. If Redis reports `MISCONF` from snapshotting failures, run it with persistence disabled (`--save "" --appendonly no --stop-writes-on-bgsave-error no`) or adjust your Redis configuration to allow writes.
