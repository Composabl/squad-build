# Skills API

### Skill Configuration

```python
from composabl_core.config import SkillConfig

skill_config = SkillConfig(
    name="temperature-control",
    type="SkillTeacher",
    config={
        "learning_rate": 0.001,
        "hidden_layers": [128, 128],
        "activation": "tanh"
    }
)
```

### Skills API

Skills define agent behaviors through different implementation strategies.

#### Skill Types

**1. SkillTeacher (Learning-based)**

```python
from composabl import SkillTeacher

class CustomTeacher(SkillTeacher):
    def __init__(self, target_position=10.0):
        self.target = target_position
        self.episode_steps = 0
        
    async def compute_reward(self, transformed_obs, action, sim_reward):
        """Calculate reward for reinforcement learning"""
        distance = abs(transformed_obs["position"] - self.target)
        
        # Shaped reward
        reward = -distance  # Negative distance
        
        # Bonus for reaching target
        if distance < 0.1:
            reward += 100
            
        # Penalty for energy usage
        reward -= 0.1 * abs(action[0])
        
        return reward
    
    async def compute_success_criteria(self, transformed_obs, action):
        """Define success condition"""
        return abs(transformed_obs["position"] - self.target) < 0.1
    
    async def compute_termination(self, transformed_obs, action):
        """Define episode termination"""
        self.episode_steps += 1
        
        # Terminate on success
        if await self.compute_success_criteria(transformed_obs, action):
            return True
            
        # Terminate on failure conditions
        if abs(transformed_obs["position"]) > 100:  # Out of bounds
            return True
            
        # Terminate on timeout
        return self.episode_steps >= 1000
    
    async def transform_sensors(self, sensors, action):
        """Preprocess sensors if needed"""
        # Normalize position to [-1, 1]
        transformed = dict(sensors)
        if "position" in transformed:
            transformed["position"] = transformed["position"] / 50.0
        return transformed
    
    async def transform_action(self, transformed_obs, action):
        """Transform action to simulator space"""
        # Clip action to valid range
        return np.clip(action, -1, 1)
    
    async def filtered_sensor_space(self):
        """Specify which sensors this skill needs"""
        return ["position", "velocity", "target"]
    
    async def compute_action_mask(self, transformed_obs, action):
        """Optional: Define valid actions"""
        # Example: Disable reverse if at boundary
        if transformed_obs["position"] <= -50:
            return [True, False]  # Can only go forward
        elif transformed_obs["position"] >= 50:
            return [False, True]  # Can only go backward
        return None  # All actions valid

# Create skill with teacher
skill = Skill("reach-target", CustomTeacher(target_position=25.0))
```

**2. SkillController (Programmatic)**

```python
from composabl import SkillController

class PIDController(SkillController):
    def __init__(self, kp=1.0, ki=0.1, kd=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
        
    async def compute_action(self, transformed_obs, action):
        """Compute PID control action"""
        # Calculate error
        error = transformed_obs["setpoint"] - transformed_obs["measurement"]
        
        # P term
        p_term = self.kp * error
        
        # I term
        self.integral += error
        i_term = self.ki * self.integral
        
        # D term
        derivative = error - self.last_error
        d_term = self.kd * derivative
        
        # Combined output
        output = p_term + i_term + d_term
        
        # Update state
        self.last_error = error
        
        return [output]
    
    async def compute_success_criteria(self, transformed_obs, action):
        """Success when error is small"""
        error = abs(transformed_obs["setpoint"] - transformed_obs["measurement"])
        return error < 0.01
    
    async def compute_termination(self, transformed_obs, action):
        """Never terminate - continuous control"""
        return False
    
    async def transform_sensors(self, sensors, action):
        """Pass through"""
        return sensors
    
    async def transform_action(self, transformed_obs, action):
        """Clip to actuator limits"""
        return np.clip(action, -100, 100)
    
    async def filtered_sensor_space(self):
        """Required sensors"""
        return ["measurement", "setpoint"]

# Create skill with controller
pid_skill = Skill("pid-control", PIDController(kp=2.0, ki=0.5, kd=0.1))
```

**3. SkillSelector**

```python
from composabl import SkillSelector

class AdaptiveSelector(SkillSelector):
    """Selects between different control strategies"""
    
    async def compute_action(self, transformed_obs, action):
        """Return selected skill index"""
        error = abs(transformed_obs["error"])
        
        if error > 10:
            return [0]  # Aggressive control
        elif error > 1:
            return [1]  # Moderate control
        else:
            return [2]  # Fine control
    
    async def compute_success_criteria(self, transformed_obs, action):
        """Success when system is stable"""
        return transformed_obs["error"] < 0.1 and transformed_obs["rate"] < 0.01
    
    async def filtered_sensor_space(self):
        return ["error", "rate", "mode"]

# Create selector with child skills
selector = SkillSelector(
    name="adaptive-control",
    implementation=AdaptiveSelector,
    children=["aggressive-pid", "moderate-pid", "fine-pid"]
)
```

**4. Coordinated Skills**

```python
from composabl import (
    SkillCoordinatedSet, 
    SkillCoordinatedPopulation,
    SkillPopulation,
    SkillCoach
)

# Coach for coordinated skills
class TeamCoach(SkillCoach):
    async def compute_reward(self, transformed_obs, action, sim_reward):
        """Reward for team coordination"""
        # Reward based on team performance
        team_distance = transformed_obs["team_spread"]
        target_reached = transformed_obs["targets_reached"]
        
        reward = target_reached * 10  # Reward for reaching targets
        reward -= team_distance * 0.1  # Penalty for spreading too far
        
        return reward
    
    async def compute_success_criteria(self, transformed_obs, action):
        return transformed_obs["all_targets_reached"]
    
    async def filtered_sensor_space(self):
        return ["team_spread", "targets_reached", "all_targets_reached"]

# Coordinate specific agents
team_set = SkillCoordinatedSet(
    name="team-coordination",
    implementation=TeamCoach,
    skills=[
        Skill("agent-1", Agent1Controller),
        Skill("agent-2", Agent2Controller),
        Skill("agent-3", Agent3Controller)
    ]
)

# Coordinate a population
swarm = SkillCoordinatedPopulation(
    name="swarm-behavior",
    implementation=SwarmCoach,
    skills=[
        SkillPopulation("drone", DroneController, amount=10),
        SkillPopulation("scout", ScoutController, amount=2)
    ]
)
```

#### Skill Composition Patterns

```python
# Hierarchical skill structure
navigation = SkillSelector("navigation", NavigationSelector, [
    Skill("path-planning", PathPlanner),
    SkillSelector("obstacle-avoidance", ObstacleSelector, [
        Skill("go-around", GoAroundObstacle),
        Skill("go-over", GoOverObstacle)
    ]),
    Skill("target-approach", ApproachTarget)
])

# Skill with fallback
class FallbackController(SkillSelector):
    async def compute_action(self, obs, action):
        # Try primary skill first
        if obs["system_health"] > 0.8:
            return [0]  # Normal operation
        else:
            return [1]  # Fallback/safe mode

fallback_skill = SkillSelector(
    "fault-tolerant-control",
    FallbackController,
    ["normal-control", "safe-mode-control"]
)
```

### Per Skill Configuration

### Algorithms

#### PPO (Proximal Policy Optimization)

```python
config = {
    "algorithm": {
        "name": "PPO",
        "config": {
            # Learning
            "lr": 5e-5,
            "lr_schedule": None,  # or [[0, 1e-3], [1000000, 1e-5]]
            
            # PPO specific
            "use_critic": True,
            "use_gae": True,
            "lambda": 0.95,
            "kl_coeff": 0.2,
            "kl_target": 0.01,
            "clip_param": 0.3,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.0,
            "entropy_coeff_schedule": None,
            
            # Training
            "num_sgd_iter": 30,
            "sgd_minibatch_size": 128,
            "shuffle_sequences": True,
            "vf_loss_coeff": 1.0,
            "model": {
                "vf_share_layers": True,
                "free_log_std": False
            },
            
            # GAE
            "gamma": 0.99,
            "normalize_advantages": True,
            
            # Batch settings
            "train_batch_size": 4000,
            "rollout_fragment_length": 200
        }
    }
}
```

#### SAC (Soft Actor-Critic)

```python
config = {
    "algorithm": {
        "name": "SAC",
        "config": {
            # Learning
            "lr": 3e-4,
            "lr_schedule": None,
            
            # SAC specific
            "twin_q": True,
            "q_model_config": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu"
            },
            "policy_model_config": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu"
            },
            "tau": 5e-3,
            "target_network_update_freq": 1,
            "initial_alpha": 1.0,
            "target_entropy": "auto",
            
            # Replay buffer
            "replay_buffer_config": {
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 1000000,
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6
            },
            
            # Training
            "train_batch_size": 256,
            "gamma": 0.99,
            "n_step": 1,
            "grad_clip": None,
            
            # Exploration
            "exploration_config": {
                "type": "StochasticSampling"
            }
        }
    }
}
```

#### DQN (Deep Q-Network)

```python
config = {
    "algorithm": {
        "name": "DQN",
        "config": {
            # Learning
            "lr": 5e-4,
            "lr_schedule": None,
            
            # DQN specific
            "dueling": True,
            "double_q": True,
            "num_atoms": 1,
            "noisy": False,
            "sigma0": 0.5,
            
            # Replay buffer
            "replay_buffer_config": {
                "type": "MultiAgentReplayBuffer",
                "capacity": 100000
            },
            
            # Exploration
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 10000
            },
            
            # Training
            "train_batch_size": 32,
            "gamma": 0.99,
            "n_step": 1,
            "target_network_update_freq": 500,
            
            # Minimum replay size
            "replay_buffer_replay_ratio": 0.0,
            "training_intensity": None
        }
    }
}
```

#### IMPALA

```python
config = {
    "algorithm": {
        "name": "IMPALA",
        "config": {
            # Learning
            "lr": 0.0005,
            "lr_schedule": None,
            
            # IMPALA specific
            "vtrace": True,
            "vtrace_clip_rho_threshold": 1.0,
            "vtrace_clip_pg_rho_threshold": 1.0,
            
            # Architecture
            "num_workers": 16,
            "num_gpus": 1,
            "num_multi_gpu_tower_stacks": 1,
            "minibatch_buffer_size": 1,
            "num_sgd_iter": 1,
            "replay_proportion": 0.0,
            "replay_buffer_num_slots": 0,
            
            # Training
            "train_batch_size": 500,
            "rollout_fragment_length": 50,
            "max_sample_requests_in_flight_per_worker": 2,
            
            # Loss
            "learner_queue_size": 16,
            "learner_queue_timeout": 300,
            "grad_clip": 40.0,
            "opt_type": "adam",
            "decay": 0.99,
            "momentum": 0.0,
            "epsilon": 0.1,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01
        }
    }
}
```

#### Custom Algorithm

```python
from ray.rllib.algorithms.algorithm import Algorithm

class CustomAlgorithm(Algorithm):
    @classmethod
    def get_default_config(cls):
        config = super().get_default_config()
        config.update({
            "custom_param": 1.0,
            "special_lr": 0.001
        })
        return config
    
    def setup(self, config):
        super().setup(config)
        # Custom setup
    
    def training_step(self):
        # Custom training logic
        result = super().training_step()
        result["custom_metric"] = self.custom_computation()
        return result

# Use custom algorithm
config = {
    "algorithm": {
        "name": "Custom",
        "class": CustomAlgorithm,
        "config": {
            "custom_param": 2.0,
            "special_lr": 0.0001
        }
    }
}
```
