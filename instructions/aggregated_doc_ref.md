# AMESA Platform — Master Reference

## 1. Core Concepts

### 1.1 What AMESA Builds

AMESA is a platform for building **multi-agent systems** that use Machine Teaching. An agent system is composed of:

- **Skill agents** — the decision-making units. Each skill handles a specific sub-task.
- **Orchestrators (selectors)** — special skills that route control to the right skill agent based on current conditions.
- **Perceptors** — a perception layer that processes raw sensor data into richer variables before it reaches the skills layer.
- **Scenarios** — defined situations (variable configurations) that the agent system must learn to handle.
- **A simulation environment** — the training ground where agents practice via episodes.

The workflow is: connect a simulator → define perception → define skills → orchestrate skills → configure scenarios → train → evaluate → deploy.

### 1.2 Skill Agent Types

| Type                              | Also Called            | How It Decides                                                    |
| --------------------------------- | ---------------------- | ----------------------------------------------------------------- |
| **Teacher** (learned skill)       | Learned skill agent    | Deep reinforcement learning                                       |
| **Controller** (programmed skill) | Programmed skill agent | Code — math, rules, optimization, MPC, PID, heuristics, API calls |
| **Selector** (orchestrator)       | Orchestrator           | Learned (DRL) or programmed                                       |

Teachers learn by practicing in simulation. Controllers execute predetermined logic. Selectors decide which child skill should be active at any given moment.

### 1.3 Terminology Quick Reference

| Term                 | Definition                                                               |
| -------------------- | ------------------------------------------------------------------------ |
| **Sensor variables** | Observations coming from the simulator — the agent's inputs              |
| **Action space**     | The set of possible actions the agent can take                           |
| **Episode**          | One complete run of the simulation from reset to termination             |
| **Reward**           | Numeric feedback after each action telling the agent how well it did     |
| **Perceptor**        | A module that transforms raw sensors into new, derived variables         |
| **Scenario**         | A named configuration of variable values/ranges representing a situation |
| **Coach**            | The teaching construct used for coordinated (multi-agent) skills         |

---

## 2. Simulation Environment

AMESA agents train inside a simulation. The simulation API extends the **Gymnasium `gymnasium.Env`** standard. You integrate your simulator by implementing the `ServerAmesa` class, which defines how the SDK talks to your simulator.

### 2.1 Setting Up the Environment Instance

#### `make(env_id, env_init)`

Creates a new instance of the environment.

- `env_id` (string) — identifier for the environment type.
- `env_init` (dict) — initial configuration parameters.

```python
async def make(self, env_id: str, env_init: dict):
    self.env_id = env_id if env_id else self.env_id
    self.env_init = env_init if env_init else self.env_init
    self.env = Sim(self.env_init)
    return {"id": "my_simulator"}
```

#### `sensor_space_info()`

Returns details about the environment's sensor (observation) space as a `gym.Space`.

```python
async def sensor_space_info(self) -> gym.Space:
    return self.env.sensor_space
```

#### `action_space_info()`

Returns the agent system's action space as a `gym.Space`.

```python
async def action_space_info(self) -> gym.Space:
    # return self.env.action_space
```

#### `action_space_sample()`

Returns a random sample from the action space. Useful for exploration.

```python
async def action_space_sample(self):
    return self.env.action_space.sample()
```

### 2.2 Running the Environment

#### `reset()`

Resets the environment and returns the first observation plus an info dict.

```python
async def reset(self):
    return self.env.reset()
```

#### `step(action)`

Applies an action and returns a tuple:

| Return Value  | Type  | Description                         |
| ------------- | ----- | ----------------------------------- |
| `observation` | —     | The observation after the action    |
| `reward`      | float | Reward received                     |
| `terminated`  | bool  | Whether the episode ended naturally |
| `truncated`   | bool  | Whether the episode was cut short   |
| `info`        | dict  | Additional metadata                 |

```python
async def step(self, action):
    return self.env.step(action)
```

#### `close()`

Signals the simulator is done; perform cleanup.

```python
async def close(self):
    self.env.close()
```

#### `set_scenario(scenario)` / `get_scenario()`

Sets or retrieves the current scenario the agent system is training on.

```python
async def set_scenario(self, scenario):
    self.env.scenario = scenario

async def get_scenario(self):
    if self.env.scenario is None:
        return None
    return self.env.scenario
```

#### Implementation notes

- **`env_init` is your sim contract.** The SDK forwards `env_init` from training → sim `make(...)`. Use it for model paths, reward config, initial state, max steps, or any custom knobs your sim needs.
- **V2 event-based sims require a transport target.** `env_init` must include either `sim_address` (connect to an existing gRPC sim) or `sim_image` (spawn a container per sim).
- **Scenario handling:** `set_scenario()` receives a `Scenario` object created from a dict. Use `scenario.sample()` to materialize concrete values for reset-time initialization.
- **Scenario serialization:** `get_scenario()` should return `None` when unset. If you embed scenario metadata in `info`, prefer a JSON string (so downstream parquet writers can store it as a string column).
- **Reward/termination responsibility:** the sim can supply reward/termination, but a skill’s teacher can override. If your sim reward is meaningful, pass it through in the teacher.

### 2.3 Visualization

#### `get_render()`

Returns the current rendered image of the environment (numpy array or string).

```python
async def get_render(self):
    return self.env.render()
```

---

## 3. Skill Agents — Learned Skills (Teachers)

Learned skills use DRL. You configure a **teacher** that provides reward signals, termination conditions, success criteria, and optional rules. The agent practices in simulation until it achieves competence.

### 3.1 Creating a Learned Skill

**Via CLI:**

```bash
AMESA login
composabl skill new
```

You'll be prompted for a name, description, and type (`teacher`). The SDK generates a folder with a Python teacher template.

### 3.2 Training with Goals (SDK)

You can use specialized teacher classes in the SDK. The five goal types are: `AvoidGoal`, `MaximizeGoal`, `MinimizeGoal`, `ApproachGoal`, `MaintainGoal`.

When you use a goal class, `compute_reward`, `compute_termination`, and `compute_success_criteria` are inherited from the goal. You can still override them.

```python
class BalanceTeacher(MaintainGoal):
    def __init__(self, *args, **kwargs):
        super().__init__("pole_theta", "Maintain pole to upright", target=0, stop_distance=0.418)
```

**Goal parameters** (applicable across goal types):

| Parameter         | Description                                     |
| ----------------- | ----------------------------------------------- |
| `sensor_variable` | The sensor variable the goal applies to         |
| `description`     | A text description of the goal                  |
| `target`          | The target value (for Maintain, Approach)       |
| `stop_distance`   | Distance from target at which to stop/terminate |

**Coordinated goals** — use `CoordinatedGoal` when a skill must balance two competing objectives simultaneously.

### 3.3 Training with Custom Rewards (SDK)

For full control, use the general Python teacher class with these functions:

#### `compute_reward(transformed_sensors, action, sim_reward)`

Returns a numeric reward signal after each action. This is the primary training feedback.

```python
def compute_reward(self, transformed_sensors, action, sim_reward):
    if self.past_sensors["state1"] < transformed_sensors["state1"]:
        return 1
    else:
        return -1
```

#### `compute_termination(transformed_sensors, action)`

Returns `True` to end the episode and start a new one. Terminate when the agent succeeds, fails, or is on a hopeless trajectory.

```python
def compute_termination(self, transformed_sensors, action):
    return False
```

#### `compute_success_criteria(transformed_sensors, action)`

Returns `True`/`False`. The platform uses this to decide when to stop training a skill and move to the next one. Also gates progression through fixed-order skill sequences.

```python
def compute_success_criteria(self, transformed_sensors, action):
    return self.counter > 100
```

Examples of success criteria strategies:

- Average episode reward crosses a threshold.
- RMSE of key variables falls below a benchmark.
- The agent beats a benchmark controller across multiple variables and trials.

### 3.4 Guiding Behavior with Rules

#### `compute_action_mask(transformed_sensors, action)`

Returns a list of 0s and 1s for each discrete action — 0 = forbidden, 1 = allowed. The mask can change every step, enabling complex conditional rules.

**Important:** Action masks work only for **discrete** action spaces. They are ignored for continuous action spaces. Since selectors always have discrete action spaces (choosing a child skill), masks always apply to selectors.

```python
def compute_action_mask(self, transformed_sensors, action):
    return [0, 1, 1]  # First action forbidden; second and third allowed
```

### 3.5 Managing Information Flow

These functions transform and filter data as it passes through the agent system.

#### `transform_sensors(sensor, action)`

Modify sensor values before they reach the skill. Common uses: unit conversion (Fahrenheit → Celsius), normalization (scaling disparate ranges to 0–1).

```python
def transform_sensor(self, sensor, action):
    return sensor
```

#### `transform_action(transformed_sensor, action)`

Modify action values for the same reasons you'd transform sensors.

```python
def transform_action(self, transformed_sensor, action):
    return action
```

#### `filtered_sensor_space()`

Return only the sensor variables this skill needs. Pass only relevant information to improve learning and performance.

```python
def filtered_sensor_space(self):
    return ["state1"]
```

---

## 4. Skill Agents — Programmed Skills (Controllers)

Controllers are deterministic, code-based skill agents. They are useful for well-understood sub-tasks where you want to use optimization, PID control, MPC, heuristics, or API calls.

### 4.1 Creating a Controller

Controllers are created only through the SDK:

```bash
AMESA login
composabl skill new
# Choose "controller" when prompted
```

This generates a folder with a `controller.py` template.

### 4.2 The Python Controller Class

#### `__init__()`

Initialize your algorithm and configuration variables. Called once when the runtime starts.

```python
def __init__(self, *args, **kwargs):
    self.counter = 0
    self.mpc = LinearMPC()
```

#### `compute_action(obs, action)`

Process the observation and return an action. This is the core decision-making function.

```python
async def compute_action(self, obs, action):
    action = self.mpc.solve(obs)
    return action
```

#### `compute_termination`, `compute_success_criteria`, `transform_sensors`, `transform_action`, `filtered_sensor_space`

These work identically to the teacher versions (see Section 3.4–3.6). Controllers don't train, but these functions connect them to the rest of the agent system during training and execution.

### 4.3 Third-Party API Integrations as Controllers

Controllers can wrap external API calls. This lets the agent system incorporate decisions from external services (machine monitoring APIs, optimization services, etc.).

```python
import requests
from composabl import SkillController

class ThirdPartyAPISkill(SkillController):
    def __init__(self, *args, **kwargs):
        self.api_url = "https://api.example.com/machine-status"

    async def compute_action(self, obs, action):
        response = self._call_api(obs)
        action = self._process_response(response)
        return action

    def _call_api(self, observation):
        try:
            response = requests.post(
                self.api_url,
                json=observation,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API call failed: {e}")
            return None

    def _process_response(self, response):
        if not response:
            return 0.0  # Default action
        action = float(response.get("action"))
        reason = response.get("reason", "No reason provided")
        print(f"Action: {action} - Reason: {reason}")
        return action

    async def transform_sensors(self, obs):
        return obs

    async def filtered_sensor_space(self):
        return ['sensor1', 'sensor2', 'sensor3']

    async def compute_success_criteria(self, transformed_obs, action):
        return False

    async def compute_termination(self, transformed_obs, action):
        return False
```

### 4.5 Adding Controllers via the SDK

```python
third_party_skill = Skill("third_party_api", ThirdPartyAPISkill)
agent.add_skill(third_party_skill)
```

Using `SkillController` signals that the skill is programmed and does not require training.

---

## 5. Orchestrating Skill Agents

Orchestration defines how multiple skill agents work together. AMESA supports three orchestration patterns.

### 5.1 Hierarchies and Sequences (Orchestrators / Selectors)

For agent systems where different skills control the system under different conditions, an **orchestrator** (also called a selector) chooses the right skill at the right time.

Orchestrators can be trained with DRL using the same goal-setting mechanism as other skills. The top-level orchestrator's goals should match the overall agent system goals.

**Two design patterns:**

| Pattern                     | Name               | When to Use                                                              |
| --------------------------- | ------------------ | ------------------------------------------------------------------------ |
| **Fixed-order sequence**    | Functional Pattern | Tasks with a fixed sequence of stages                                    |
| **Variable-order sequence** | Strategy Pattern   | Tasks requiring different strategies for different situations/conditions |

### 5.2 Skill Groups (Plan-Execute Pattern)

Instead of one skill being active at a time, **skill groups** have two or more skills working together on a decision.

The **Plan-Execute Pattern** works as follows: one skill determines _what_ the action should be; a second skill determines _how_ to implement it.

**Example — Industrial Mixer:**
A DRL plan skill decides the target temperature (set point). An MPC execute skill (controller) determines what coolant flow is needed to reach that set point.

**Training order:** AMESA always trains plan-execute groups from bottom to top. The execute skill must achieve competence before the plan skill begins training. This ensures each skill can attribute performance variations to its own actions.

### 5.3 Coordinated Skills (Multi-Agent Training)

Multiple skills learn to act **in parallel** toward a shared goal. This is also called Multi-Agent Training. Coordinated skills use a **Coach** instead of a Teacher.

**Use cases:** traffic optimization, collaborative robotics, smart grids, multiplayer game NPCs, communication networks, environmental management, healthcare logistics, supply chain optimization.

**SDK implementation**

```python
class CoordinatedCoach(Coach):
    def __init__(self):
        self.counter = 0

    def compute_reward(self, transformed_sensors, action, sim_reward):
        self.counter += 1
        return 1  # Can return per-sub-skill rewards as a dict

    def compute_success_criteria(self, transformed_sensors, action):
        return self.counter > 100

    def compute_termination(self, transformed_sensors, action):
        return self.counter > 150

    def transform_action(self, composabl_sensors, action):
        return action

# Construct the agent
s1 = Skill("skill1", IncrementTeacher)
s2 = Skill("skill2", IncrementTeacher)

a = Agent()
a.add_coordinated_skill(CoordinatedSkill(
    "my-coordinated-skill",
    CoordinatedCoach,
    [s1, s2]
))
```

The coordinated skill receives the shared observation and action spaces, distributes them to sub-skills, collects their outputs, and returns the combined result to the agent system.

---

## 6. Scenarios

Scenarios carve the simulation space into named situations defined by specific variable configurations. They enable targeted training — each skill practices only the scenarios relevant to it — and help orchestrators learn which skill to activate under which conditions.

### 6.1 Defining Scenarios by Variable Type

| Variable Type          | Definition Method    | Example                                    |
| ---------------------- | -------------------- | ------------------------------------------ |
| **Discrete**           | Named categories     | `windy`, `far_from_charger`, `low_battery` |
| **Continuous (exact)** | Single numeric value | `windspeed = 20 knots`                     |
| **Continuous (range)** | A range of values    | `windspeed between 20–40 knots`            |

Discrete variables often come from a perceptor (e.g., an ML classifier) that transforms raw sensors into categorical outputs.

#### SDK note: Scenario dicts (for coding agents)

When building agents in Python, you can pass either a `Scenario` object or a plain `dict` into `skill.add_scenario(...)`. Dicts are converted via `Scenario.from_json(...)`. The dict keys become scenario variables, and values can be:

- `{"data": <value>, "type": "is_equal"}` for constants
- `{"data": [low, high], "type": "is_between"}` for ranges
- `{"data": [v1, v2, ...], "type": "is_element_of"}` for discrete sets
- Short-hand lists (e.g., `[low, high]`) for ranges or discrete sets

Keep scenario dictionaries JSON-serializable so they can flow through trainer configs and be stored in historian outputs.

### 6.2 Example: Restaurant Operations

Three scenarios based on recipe demand levels:

| Scenario      | Recipe A | Recipe B | Recipe C |
| ------------- | -------- | -------- | -------- |
| Low demand    | 30       | 20       | 10       |
| Normal demand | 60       | 45       | 10       |
| High demand   | 100      | 50       | 25       |

The agent trains on low demand first until success criteria are met, then normal, then high. Knowledge accumulates, so later scenarios are learned faster.

### 6.3 Scenario Flows

Scenario flows define ordered sequences of scenarios for training. Without flows, AMESA connects scenarios at random. Flows are important when the order matters (e.g., practicing flying in high winds _then_ landing in the same conditions).

### 6.4 Assigning Scenarios to Skill Agents

Scenarios are assigned to individual skill agents during configuration. Not every skill needs every scenario — a landing skill doesn't practice takeoff scenarios. In the skill agent configuration modal, check the boxes next to each relevant scenario for each section.

---

## 7. Perception Layer (Perceptors)

The perception layer sits before the skills layer in the agent system architecture. Each perceptor inputs sensor variables, processes them, and outputs **new derived variables** that are automatically added to the sensor list.

Perceptors can use any Python function or library, including ML models and LLM APIs.

### 7.1 Adding Perceptors to Agent Systems

Perception always comes before orchestrators and skills — raw sensor data is transformed first, then the enriched data flows to the decision-making layers.

### 7.2 Creating a New Perceptor

Use the CLI to generate a perceptor template:

```bash
composabl perceptor new
```

This creates a directory structure:

```
perceptor_name/
├── perceptor_name/
│   ├── __init__.py
│   └── perceptor.py
├── pyproject.toml
└── README.md
```

#### Basic Perceptor Example

A perceptor that calculates the delta (change) of a sensor variable between steps:

```python
class DeltaCounter():
    def __init__(self):
        self.key = "state1"
        self.previous_value = None

    def compute(self, sensors):
        if self.previous_value is None:
            self.previous_value = sensors[self.key]
            return {"delta_counter": 0, "state2": 0}

        delta = sensors["state1"] - self.previous_value
        self.previous_value = sensors["state1"]
        return {"delta_counter": delta, "state2": 0}

    def filtered_sensor_space(self, sensors):
        return ["state1"]

# Register the perceptor with its output variable names
delta_counter = Perceptor(
    ["delta_counter", "state2"],
    DeltaCounter,
    "the change in the counter from the last two steps"
)
```

Key points:

- `compute()` returns a dict of new variable names → values.
- `filtered_sensor_space()` declares which sensors the perceptor needs as input.
- The `Perceptor()` constructor takes: output variable names, the implementation class, and a description.

### 7.3 ML Model as a Perceptor

You can wrap a pre-trained ML model (e.g., a scikit-learn model saved as a `.pkl` file) as a perceptor. This allows the agent to benefit from predictions like anomaly detection or condition classification.

#### `pyproject.toml` Configuration

```toml
[project]
name = "perc"
version = "0.1.0"
description = "Thermal runaway predictor"
dependencies = [
    "composabl-core",
    "scikit-learn",  # or whatever your model needs
]

[composabl]
type = "perceptor"
entrypoint = "perc.perceptor:ThermalRunawayPredict"
```

#### Implementation

```python
import pickle
from composabl import Perceptor, PerceptorImpl

class ThermalRunawayPredict(PerceptorImpl):
    def __init__(self, *args, **kwargs):
        self.last_Tc = 0
        self.ml_model = pickle.load(open("ml_models/ml_predict_temperature.pkl", 'rb'))

    async def compute(self, obs_spec, obs):
        if type(obs) != dict:
            obs_keys = [s.name for s in sensors]
            obs = dict(zip(obs_keys, obs))

        if self.last_Tc == 0:
            delta_Tc = 5
        else:
            delta_Tc = float(obs['Tc']) - self.last_Tc

        X = [[float(obs['Ca']), float(obs['T']), float(obs['Tc']), delta_Tc]]
        prediction = self.ml_model.predict(X)[0]
        self.last_Tc = float(obs['Tc'])

        return {"thermal_runaway_predict": prediction}

    def filtered_sensor_space(self, obs):
        return ['T', 'Tc', 'Ca']
```

### 7.4 LLM as a Perceptor

LLM perceptors add natural language capabilities to agent systems. AMESA defines three persona patterns:

| Persona           | Role                                                              | Data Flow                                    |
| ----------------- | ----------------------------------------------------------------- | -------------------------------------------- |
| **Analyst**       | Interprets sensor data and displays insights to a human operator  | Agent → LLM → Human (output only)            |
| **Executive**     | Reads external text sources and reports information to the agent  | External text → LLM → Agent (input to agent) |
| **Plant Manager** | Lets operators give instructions in natural language to the agent | Human → LLM → Agent (input to agent)         |

#### Analyst Example

Displays information to the operator but does not feed into the agent's decision-making layer (returns `0`):

```python
from composabl_core import PerceptorImpl

class AnalystPerceptor(PerceptorImpl):
    def __init__(self, *args, **kwargs):
        self.llm_client = llm_client()
        self.factory_console_client = factory_console_client()

    async def compute(self, obs_spec, obs):
        llm_response = self.llm_client.ask(
            f"You are controlling a CSTR plant, the current state is {obs}. "
            "What are your thoughts on the current state of the plant?"
        )
        self.factory_console_client.post(
            f"LLM analysis: {llm_response}"
        )
        return {"chemical_engineer_llm": 0}
```

#### Executive Example

Queries an LLM for a recommended action, which becomes a new sensor variable the skill agent's teacher considers during training:

```python
from composabl_core import PerceptorImpl

class ChemicalEngineerPerceptor(PerceptorImpl):
    def __init__(self, *args, **kwargs):
        self.llm_client = llm_client()

    async def compute(self, obs_spec, obs):
        llm_response = self.llm_client.ask(
            f"You are controlling a CSTR plant, the current state is {obs}. "
            "What action do you recommend?"
        )
        llm_action = llm_response.find("action")
        return {"chemical_engineer_llm": llm_action}
```

#### Important: Filtering Text Variables

LLM perceptors may output text fields. AMESA agent systems can include text in perceptors, but text variables **must be transformed or filtered out** in the `teacher.py` file before DRL training. Use `filtered_sensor_space()` to remove any text variables that haven't been converted to numeric types.

### 7.5 Publishing Perceptors

```bash
composabl login
composabl perceptor publish foldername
```

Select the target organization and project. Refresh the Agent Orchestration Studio to see the perceptor and add it to agent systems.

---

## 8. Training Your Agent — Trainer API & Configuration

The **Trainer** orchestrates the entire training lifecycle: spinning up simulation workers, collecting experience, updating policies, and managing checkpoints. You configure training via a config dict or TrainerConfig object, then call `trainer.train(agent, train_cycles=N)`.

### 8.1 Creating and Configuring the Trainer

#### Basic Setup

```python
from amesa_train.trainer import Trainer
from amesa_core.agent.agent import Agent

# Create your agent (see §1–7)
agent = Agent()
agent.add_sensors([...])
agent.add_skill(...)

# Define trainer configuration
# Set EITHER local OR v2 (or docker, kubernetes, etc.)
config = {
    "target": {
        "v2": {
            "redis_url": "redis://localhost:6379",
            "sim_image": "amesa/sim-xgboost:latest",
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
            "sim_image": "amesa/sim-xgboost:latest",
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
| `sim_image`            | str  | None                       | Docker image for the simulator (e.g., `"amesa/sim-xgboost:latest"`).                                         |
| `initial_replicas`     | int  | 1                          | Number of simulator instances to spawn. Should match expected concurrency.                                   |
| `num_episode_managers` | int  | 1                          | Number of episode managers. Training samples are sharded across them. Must divide `initial_replicas` evenly. |
| `enable_ppo_training`  | bool | True                       | Whether to run PPO training.                                                                                 |
| `ppo_training_samples` | int  | 4000                       | Total samples collected before triggering PPO updates.                                                       |
| `enable_evaluation`    | bool | False                      | Whether to run post-training evaluation.                                                                     |
| `enable_historian`     | bool | False                      | Whether to enable telemetry historian.                                                                       |
| `enable_auto_scale`    | bool | False                      | Whether to auto-scale sim/skill replicas during training.                                                    |

### 8.2 Training Entry Point

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
    Sensor("theta", "Pole angle (radians)", lambda s, n="theta", i=0: s[n] if isinstance(s, dict) else s[i]),
    Sensor("theta_dot", "Pole angle velocity", lambda s, n="theta_dot", i=1: s[n] if isinstance(s, dict) else s[i]),
])

skill = Skill(
    "balance-pole",
    CartpoleBalanceTeacher,
    training_cycles=5,
    custom_action_space=Box(low=-1.0, high=1.0, shape=(1,))
)
agent.add_skill(skill)

# 3. Configure and run training
config = {
    "target": {
        "local": {
            "address": "localhost:50051"
        }
    }
}

trainer = Trainer(config)
try:
    trainer.train(agent, train_cycles=5)
finally:
    trainer.close()
```

### 8.3 Checking Training Progress

During training, the SDK emits telemetry events to the historian (if enabled). Query training status via:

```python
# Check if trainer is still running
is_running = trainer.is_training()

# Get the current run ID
run_id = trainer.get_run_id()
```

Training logs are written to stdout. Look for lines like:

```
2026-04-23 10:15:00 INFO [trainer] Cycle: 1/5, Episodes: 8, Steps/sec: 218.85
2026-04-23 10:15:42 INFO [trainer] PPO update completed, loss: 0.0234
```

### 8.4 Accessing Trained Models

After training completes, model checkpoints are saved to:

```
{output_dir}/{skill_name}/policy_checkpoint_{timestamp}/
```

Example:

```bash
ls xgboost_benchmarks/xgboost-control/policy_checkpoint_20260423_101542/
# Output: checkpoint-0, algorithm_state.json, ...
```

To load a policy for inference:

```python
from ray.rllib.algorithms.ppo import PPO

# Load the checkpoint
policy = PPO.from_checkpoint("xgboost_benchmarks/xgboost-control/policy_checkpoint_20260423_101542/")
action = policy.compute_single_action(observation)[0]
```

---

## 9. Architecture Summary

The data flow through an AMESA agent system follows this path:

```
Simulator
   │
   ▼
┌──────────────────────┐
│   Perception Layer   │  ← Perceptors (ML models, LLMs, custom Python)
│  (raw sensors → new  │     transform raw data into enriched variables
│   derived variables) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Orchestrator /     │  ← Selectors route control to the right skill
│      Selector         │     based on scenario and conditions
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Skill A │ │ Skill B │  ← Teachers (DRL) or Controllers (code)
│(learned)│ │(program.)│     make decisions
└────┬────┘ └────┬────┘
     └─────┬─────┘
           │
           ▼
       Action(s)
           │
           ▼
       Simulator
```

Orchestration patterns layer on top:

- **Hierarchies/Sequences** — selector picks one skill at a time.
- **Skill Groups** — plan skill feeds into execute skill.
- **Coordinated Skills** — multiple skills act in parallel, trained by a coach.

Scenarios slice the simulation space so skills train efficiently on relevant conditions, and orchestrators learn which skill to deploy in which situation.

---

## 10. V2 Runtime Architecture — Event-Driven Training

V2 replaces Ray-based worker orchestration with a **Redis Streams-based event architecture**. Instead of Ray rollout workers collecting steps, V2 uses **components** (EventSimProcessor, EventSkillProcessor, EpisodeManager) that read/write to Redis topics. This enables horizontal scaling, easier debugging, and better integration with external systems.

### 10.1 V2 Architecture Overview

```
Redis Streams (default topic names with -sim-{i} per-sim suffix)

amesa.sim.input.sim-i  ──────┐
                             ▼
                     ┌──────────────────────┐
                     │ EventSimProcessor    │  (one per sim)
                     │ (calls sim.step())   │
                     └──────────────────────┘
                             │
                             ▼
amesa.sim.observations.sim-i ──────┐
   [optional]                      │
   amesa.perceptor.output.sim-i    │
       ▼                           │
   ┌────────────────────┐          │
   │ EventPerceptorProc │          │
   └────────────────────┘          │
       │                           │
       └──────────────┬────────────┘
                      │
   ┌──────────────────┴──────────────────┐
   │  EventSkillProcessor                │
   │  (applies teacher reward, term.)    │
   └──────────────────┬──────────────────┘
                      │
                      ▼
              amesa.skill.actions  ──────┐
                                         │
                      ┌──────────────────┘
                      │
   ┌──────────────────▼──────────────────┐
   │ EpisodeManager                      │
   │ (aggregates into episodes, trains   │
   │  PPO when sample target reached)    │
   └─────────────────────────────────────┘
```

**Note:** Topic names with `.sim-{i}` suffix isolate per-simulator traffic. All names are configurable in `TrainerTargetV2Config`.

### 10.2 Core Components

#### EventSimProcessor

Handles simulator execution. Reads `amesa.sim.input.sim-{i}` topic, calls `sim.step()`, writes observations to `amesa.sim.observations.sim-{i}`.

**Default Configuration in run_v2:**

```python
# Topics use per-sim naming
EventSimProcessor(
    env_init={
        "sim_address": "localhost:1337",  # or sim_image
    },
    redis_url="redis://localhost:6379",
    input_topic="amesa.sim.input.sim-0",      # Per-sim input
    output_topic="amesa.sim.observations.sim-0",  # Per-sim observations
    consumer_group="amesa-sim-consumer-group",
)
```

**Usage (automatic via V2 runner):**

```python
# Components are auto-instantiated by run_v2() — manual instantiation rarely needed:
from amesa_core.networking.sim.v2.event_sim_processor import EventSimProcessor

sim_proc = EventSimProcessor(
    env_init={"sim_address": "localhost:1337"},
    redis_url=redis_url,
    input_topic="amesa.sim.input.sim-0",
    output_topic="amesa.sim.observations.sim-0",
)
await sim_proc.init()
await sim_proc._run()  # Blocking loop; run in asyncio.Task
```

#### EventSkillProcessor

Applies teacher reward/termination logic to observations. Reads from input topic (perceptor output or sim observations), writes to skill action topic.

**Topics:**

- Input: `amesa.sim.observations.sim-{i}` or `amesa.perceptor.output.sim-{i}` (observations ± derived variables)
- Output: `amesa.skill.actions` (augmented with teacher reward/terminated flags)

#### EpisodeManager

Aggregates individual steps into complete episodes, triggers PPO when sample threshold is reached.

**Topics:**

- Input: `amesa.skill.actions` (individual steps from all skill processors)
- Output (if PPO enabled): aggregates & calls `run_ppo()` internally; publishes training telemetry to `amesa.episodes`

**Configuration:**

```python
episode_manager = EpisodeManager(
    skill=agent_skill,
    redis_url=redis_url,
    ppo_training_samples=1000,       # Trigger PPO every N samples
    num_episode_managers=4,           # Total EMs (for sharding)
    episode_manager_id=0,             # This EM's shard ID
    enable_ppo_training=True,         # Run PPO updates
    training_sample_group_target=250, # Sync barrier (samples per group)
)
```

### 10.3 Event Message Flow

Each step flows through the system as a series of Redis stream messages:

**1. Agent requests sim step:**

```python
# Published to amesa.sim.input.sim-{i}
{
    "type": "step",
    "sim_id": "sim-0",
    "episode_id": "ep-123",
    "action": [0.5, 0.3, 0.2, ...],
    "request_id": "req-456"
}
```

**2. Sim processor responds:**

```python
# Published to amesa.sim.observations.sim-{i}
{
    "type": "step",
    "sim_id": "sim-0",
    "episode_id": "ep-123",
    "result": {
        "observation": [1.2, 0.5, 3.1, ...],
        "terminated": False,
        "truncated": False,
        "info": {...}
    },
    "message_id": "msg-789",
    "episode_step": 5
}
```

**3. Skill processor augments with teacher logic:**

```python
# Published to amesa.skill.actions
{
    "type": "step",
    "sim_id": "sim-0",
    "episode_id": "ep-123",
    "step_number": 5,
    "sim_result": {...},  # From step 2
    "teacher_reward": 0.95,
    "teacher_terminated": False,
    "action": [0.5, 0.3, 0.2, ...],
    "sensors_filtered": {
        "observation": [1.2, 0.5, 3.1, ...]
    }
}
```

**4. Episode manager collects steps and triggers training:**

```
Steps 1–250 collected → Training sample group target reached
  ↓
PPO.train() called on collected steps
  ↓
Policy weights updated
  ↓
Next episode uses new policy
```

### 10.4 Redis Topic Names

V2 uses Redis Streams with configurable topic names (defaults shown):

| Topic                            | Direction | Content                                       | Default                |
| -------------------------------- | --------- | --------------------------------------------- | ---------------------- |
| `amesa.sim.input.sim-{i}`        | →         | Step requests with actions                    | sim_input_topic        |
| `amesa.sim.observations.sim-{i}` | ←         | Sim responses with observations               | sim_observation_topic  |
| `amesa.perceptor.output.sim-{i}` | ←         | Perceptor-derived observations (optional)     | perceptor_output_topic |
| `amesa.skill.actions`            | ←         | Augmented steps (teacher reward, termination) | skill_action_topic     |
| `amesa.episodes`                 | ←         | Training telemetry (aggregated episodes)      | episode_topic          |

**All topic names are configurable via `TrainerTargetV2Config`** (see section 8.2).

### 10.5 Environment Initialization for V2

When using V2, `env_init` is transported via Redis event to each EventSimProcessor. Ensure it includes the **sim connection** info:

```python
# Option A: Connect to a running gRPC sim server
env_init = {
    "sim_address": "localhost:1337",  # host:port
}

# Option B: Spawn a new Docker container per sim (requires sim_image)
env_init = {
    "sim_image": "amesa/sim-xgboost:latest",
    "sim_port": 1337,  # Port inside container
}
```

V2's `run_v2_async()` function publishes `env_init` to all sim processors at startup via the initialization handshake.

### 10.6 Logging and Debugging

Enable detailed logging to see event flow:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run training
trainer.train(agent, train_cycles=5)
```

Look for log lines like:

```
DEBUG [EventSimProcessor] Received message on sim_input: type=step, message_id=msg-789
DEBUG [EventSkillProcessor] Processing step 5 for ep-123
DEBUG [EpisodeManager] Episode ep-123 complete: 250 steps, reward sum: 125.4
DEBUG [PPO] PPO update: 250 samples, loss: 0.0234
```

---

## Appendix A — Complete End-to-End Example

This example ties together all layers: simulation API → perceptor → teacher → agent orchestration → training.

### A.1 Sim Implementation (ServerAmesa)

```python
# file: my_sim.py
import gymnasium as gym
import numpy as np
from amesa_core.networking.sim.server_amesa import ServerAmesa

class XGBoostSim(gym.Env):
    """
    Simulates a fermenter control problem using a pre-trained XGBoost model.
    Accepts 13-dim actions, returns 5-dim observations.
    """

    def __init__(self, env_init=None):
        self.env_init = env_init or {}
        self.model = None
        self.state = None
        self.step_count = 0
        self.max_steps = 500

    async def make(self, env_id: str, env_init: dict):
        """Initialize the simulator with env_init parameters."""
        import xgboost as xgb

        self.model = xgb.Booster()
        self.model.load_model("models/xgboost_fermenter.json")

        self.state = np.zeros(5)  # VOC, pH, Temp, Level, Yield
        self.step_count = 0
        self.max_steps = env_init.get("max_steps", 500)

        return {"id": "xgboost_fermenter", "max_episode_steps": self.max_steps}

    async def sensor_space_info(self) -> gym.Space:
        """5-dim continuous observations."""
        return gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

    async def action_space_info(self) -> gym.Space:
        """13-dim continuous actions."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)

    async def action_space_sample(self):
        """Sample a random action."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(13,)).sample()

    async def reset(self):
        """Reset simulator and return initial observation."""
        self.state = np.array([0.1, 7.0, 30.0, 0.5, 0.2], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    async def step(self, action):
        """Execute one step with the XGBoost model."""
        # Feature vector: [prev_states (5) + prev_actions (13) + current_actions (13)]
        prev_actions = np.zeros(13)  # Simplified
        features = np.concatenate([self.state, prev_actions, action])

        # XGBoost predicts reward and next state
        prediction = self.model.predict(np.array([features]))[0]
        reward = float(prediction[0])

        # Simulate state drift
        self.state = self.state * 0.95 + action[:5] * 0.05
        self.step_count += 1

        terminated = self.step_count >= self.max_steps

        return self.state, reward, terminated, False, {}

    async def close(self):
        pass

    async def set_scenario(self, scenario: dict):
        """Set scenario parameters (initial conditions, reward config, etc.)."""
        if scenario:
            self.state = np.array([
                scenario.get("VOC", 0.1),
                scenario.get("pH", 7.0),
                scenario.get("Fermenter_Temp", 30.0),
                scenario.get("Fermenter_Level", 0.5),
                scenario.get("Yield50", 0.2),
            ], dtype=np.float32)

    async def get_scenario(self):
        return None


# Wrap in ServerAmesa
from amesa_core.networking.sim.server_amesa import ServerAmesa

class ServerImpl(ServerAmesa):
    def __init__(self, *args, **kwargs):
        self.env_init = kwargs.get("env_init", {})
        self.env = XGBoostSim(self.env_init)

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

    async def close(self):
        pass

    async def set_scenario(self, scenario: dict):
        """Set scenario parameters (initial conditions, reward config, etc.)."""
        if scenario:
            self.state = np.array([
                scenario.get("VOC", 0.1),
                scenario.get("pH", 7.0),
                scenario.get("Fermenter_Temp", 30.0),
                scenario.get("Fermenter_Level", 0.5),
                scenario.get("Yield50", 0.2),
            ], dtype=np.float32)

    async def get_scenario(self):
        return None

````

### A.2 Perceptor (Optional)

```python
# file: my_perceptor.py
from amesa_core.agent.perceptor import PerceptorImpl

class YieldDeltaPerceptor(PerceptorImpl):
    """Computes yield delta between steps (change in Yield50)."""

    def __init__(self):
        self.last_yield = None

    async def compute(self, obs_spec, obs):
        current_yield = obs[4]  # Yield50 is index 4

        if self.last_yield is None:
            delta = 0.0
        else:
            delta = current_yield - self.last_yield

        self.last_yield = current_yield
        return {"yield_delta": delta}

    def filtered_sensor_space(self, obs):
        return []  # No input filtering needed
````

### A.3 Teacher Skill

```python
# file: my_teacher.py
from amesa_core.agent.skill.skill_teacher import SkillTeacher
from amesa_core.spaces import Box
from typing import Dict, List

class YieldMaximizer(SkillTeacher):
    """Learns to maximize fermenter yield over 500 steps."""

    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-1.0, high=1.0, shape=(13,))

    async def compute_reward(self, transformed_sensors: Dict, action, sim_reward: float) -> float:
        # Reward is normalized yield improvement
        yield_value = transformed_sensors.get("Yield50", 0.0)
        yield_delta = transformed_sensors.get("yield_delta", 0.0)

        # Bonus for increasing yield, penalty for decreasing
        reward = yield_delta * 10.0 + yield_value
        return float(reward)

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        # Terminate if pH goes out of bounds
        pH = transformed_sensors.get("pH", 7.0)
        return pH < 6.0 or pH > 8.0

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        # Success when Yield50 > 0.7
        return transformed_sensors.get("Yield50", 0.0) > 0.7

    async def filtered_sensor_space(self) -> List[str]:
        return ["VOC", "pH", "Fermenter_Temp", "Fermenter_Level", "Yield50", "yield_delta"]

    async def transform_sensors(self, sensors, action):
        return sensors

    async def transform_action(self, transformed_sensors: Dict, action):
        return action if action is not None else self.action_space.sample()
```

### A.4 Building the Agent

```python
# file: build_agent.py
from amesa_core.agent.agent import Agent
from amesa_core.agent.sensors.sensor import Sensor
from amesa_core.agent.skill.skill import Skill
from my_teacher import YieldMaximizer
from my_perceptor import YieldDeltaPerceptor

def build_agent():
    agent = Agent()

    # Add sensors
    agent.add_sensors([
        Sensor("VOC", "Volatile organic compounds", lambda s, i=0: s[i]),
        Sensor("pH", "pH level", lambda s, i=1: s[i]),
        Sensor("Fermenter_Temp", "Temperature (°C)", lambda s, i=2: s[i]),
        Sensor("Fermenter_Level", "Fill level (0-1)", lambda s, i=3: s[i]),
        Sensor("Yield50", "Product yield target", lambda s, i=4: s[i]),
    ])

    # Add perceptor
    agent.add_perceptor(YieldDeltaPerceptor())

    # Add skill
    skill = Skill(
        "yield-optimizer",
        YieldMaximizer,
        training_cycles=10,
    )

    # Add scenario
    baseline_scenario = {
        "scenario_name": "baseline",
        "VOC": [0.05, 0.15],
        "pH": [6.8, 7.4],
        "Fermenter_Temp": [28.0, 32.0],
        "Fermenter_Level": [0.4, 0.6],
        "Yield50": [0.2, 0.4],
    }
    skill.add_scenario(baseline_scenario)

    agent.add_skill(skill)
    return agent
```

### A.5 Running Training (V2 Event-Based)

```python
# file: run_training.py
import os
import sys
import subprocess
import time
from amesa_train.trainer import Trainer
from build_agent import build_agent

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REDIS_URL = "redis://localhost:6379"
SIM_IMAGE = "amesa/sim-xgboost:latest"

def start_redis():
    """Start a Docker Redis container."""
    result = subprocess.run(
        ["docker", "run", "-d", "-p", "6379:6379", "redis:7-alpine"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def wait_redis(redis_url, retries=30):
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
                "v2": {
                    "redis_url": REDIS_URL,
                    "sim_image": SIM_IMAGE,
                    "initial_replicas": 4,
                    "num_episode_managers": 2,
                    "enable_ppo_training": True,
                    "ppo_training_samples": 1000,
                    "enable_evaluation": False,
                    "enable_historian": False,
                }
            }
        }

        # Train
        print("Starting training…")
        trainer = Trainer(config)
        try:
            trainer.train(agent, train_cycles=10)
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

### A.6 Running Training (Local Ray-Based)

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

    # Use local Ray-based training pointing to a gRPC sim at localhost:1337
    config = {
        "target": {
            "local": {
                "address": "localhost:1337"  # gRPC sim endpoint
            }
        }
    }

    trainer = Trainer(config)
    try:
        trainer.train(agent, train_cycles=10)
        print("✅ Training complete!")
    finally:
        trainer.close()

if __name__ == "__main__":
    main()
```

### A.7 What This Example Covers

1. **Sim** — XGBoost sim implementing ServerAmesa
2. **Perceptor** — Computes yield delta (derived variable)
3. **Teacher** — Maximizes yield with DRL training
4. **Agent** — Combines sensors, perceptor, skill, and scenarios
5. **V2 Training** — Event-driven orchestration with Redis + EventSimProcessor + EpisodeManager
6. **Local Training** — Ray-based alternative for local development
7. **Deployment** — Both examples ready for `trainer.train()` with minimal setup

---

## Appendix B — Zero-to-Run (Minimal Repo for a New Sim + Agent)

This appendix is the missing glue for a **clean-room repo**: package layout, server entrypoint, and
the commands needed to run a local gRPC sim or a V2 `sim_image`. It is intended to be sufficient
context for a coding agent to build **sim + server + server_impl + full agent stack** from scratch.

### B.1 Minimal repo layout

```
my_sim_project/
├── requirements.txt             # quick local runs (pip)
├── pyproject.toml               # required if you load the sim by path/plugin
├── sim/
│   ├── __init__.py
│   ├── sim.py                   # Gymnasium Env (pure simulation logic)
│   ├── server_impl.py           # ServerAmesa adapter (SimImpl)
│   └── server.py                # gRPC/HTTP runner
├── agent/
│   ├── perceptor.py             # optional
│   ├── teacher.py               # learned skill
│   ├── controller.py            # programmed skill (optional)
│   ├── build_agent.py
│   └── run_training.py
└── README.md
```

### B.2 Dependencies (local runs)

Use the full SDK if you want training + core in a single install:

```bash
pip install amesa
```

Minimal sim-only installs typically need:

```bash
pip install amesa-core gymnasium numpy
```

> **License**: set `AMESA_LICENSE` and `AMESA_EULA_AGREED=1` in your environment before training.

### B.3 `pyproject.toml` for sim plugins (required for path loading)

When using `server_entrypoint` or any path-based sim loading, you must provide a `[amesa]` section
with an explicit entrypoint. This mirrors the CLI sim template:

```toml
[project]
name = "my-sim"
version = "0.1.0"
description = "My custom sim"
dependencies = [
  "amesa-core"
]

[amesa]
type = "sim"
entrypoint = "sim.server_impl:SimImpl"
dependencies_system = []
```

### B.4 `server_impl.py` (Sim → ServerAmesa adapter)

Use a Gymnasium `Env` for the simulation logic, then wrap it in a `ServerAmesa` implementation
that forwards `make`, `reset`, `step`, etc.

```python
# sim/server_impl.py
import gymnasium as gym
from typing import Any, Dict, SupportsFloat, Tuple
from amesa_core.networking.sim.server_amesa import ServerAmesa
from .sim import Env

class SimImpl(ServerAmesa):
    def __init__(self, *args, **kwargs):
        self.env_init = kwargs.get("env_init", {})
        self.env = Env(self.env_init)

    async def make(self, env_id: str, env_init: dict):
        self.env_init = env_init if env_init else self.env_init
        return {"id": "my_sim", "max_episode_steps": 1000}

    async def sensor_space_info(self) -> gym.Space:
        return self.env.sensor_space

    async def action_space_info(self) -> gym.Space:
        return self.env.action_space

    async def action_space_sample(self) -> Any:
        return self.env.action_space.sample()

    async def reset(self) -> Tuple[Any, Dict[str, Any]]:
        return self.env.reset()

    async def step(
        self, action
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        return self.env.step(action)

    async def close(self):
        self.env.close()
```

### B.5 `server.py` (gRPC/HTTP runner)

This is the actual server process that your trainer or V2 runner connects to:

```python
# sim/server.py
import asyncio
import os
from argparse import ArgumentParser
from amesa_core.networking.sim import server as server_make
from sim.server_impl import SimImpl

async def start(host, port, protocol, env_init):
    server = server_make.make(
        server_impl=SimImpl,
        host=host,
        port=port,
        protocol=protocol,
        env_init=env_init,
    )
    await server.start()
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    parser = ArgumentParser(description="Start the sim server")
    parser.add_argument("--host", default=os.environ.get("HOST") or "[::]")
    parser.add_argument("--port", type=int, default=os.environ.get("PORT") or 1337)
    parser.add_argument("--protocol", default="grpc")
    parser.add_argument("--env_init", type=str, default="{}")
    args = parser.parse_args()

    # env_init is parsed from string to dict
    args.env_init = eval(args.env_init)
    asyncio.run(start(args.host, args.port, args.protocol, args.env_init))
```

### B.6 Running a local gRPC sim (for `target.local`)

**Terminal A — start the sim server:**

```bash
python sim/server.py --host 0.0.0.0 --port 1337 --protocol grpc --env_init "{}"
```

**Terminal B — run training with a local target:**

```python
# agent/run_training.py (excerpt)
config = {
  "target": {
    "local": {"address": "localhost:1337"}  # gRPC sim endpoint
  }
}
trainer = Trainer(config)
trainer.train(agent, train_cycles=10)
```

### B.7 V2 `sim_image` (Dockerized sim server)

If you want V2 to **spawn the sim container per replica**, build a container that starts the sim
server on port `1337`.

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY sim ./sim
CMD ["python", "sim/server.py", "--host", "0.0.0.0", "--port", "1337", "--protocol", "grpc", "--env_init", "{}"]
```

Build + use in V2 config:

```bash
docker build -t my-sim:latest .
```

```python
config = {
  "target": {
    "v2": {
      "redis_url": "redis://localhost:6379",
      "sim_image": "my-sim:latest",
      "initial_replicas": 4,
      "num_episode_managers": 2,
    }
  }
}
```

### B.8 Full agent stack checklist (what a coding agent should create)

- **Sim**: Gymnasium `Env` + `ServerAmesa` adapter (`server_impl.py`)
- **Server**: `server.py` runner (gRPC/HTTP)
- **Perceptors**: `PerceptorImpl` classes + registration in `Agent`
- **Skills**:
  - **Teacher** (DRL): `SkillTeacher` with reward/termination/criteria
  - **Controller** (programmed): `SkillController` with `compute_action`
- **Scenarios**: dict-based scenarios added via `skill.add_scenario(...)`
- **Selectors/Groups** (optional):
  - **SkillSelector** to route between skills
  - **SkillGroup** for plan/execute pairings
- **Trainer config**: local or V2 target, plus `ppo_training_samples`, etc.

These pieces, plus the examples in Appendix A, are the **minimum complete context** needed to
build a sim + server + full agent stack in a standalone repo.
