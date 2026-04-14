## Skill Agents — Learned Skills (Teachers)

Learned skills use DRL. You configure a **teacher** that provides reward signals, termination conditions, success criteria, and optional rules. The agent practices in simulation until it achieves competence.

### Creating a Learned Skill

**Via CLI:**

```bash
AMESA login
composabl skill new
```

You'll be prompted for a name, description, and type (`teacher`). The SDK generates a folder with a Python teacher template.

### Training with Goals

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

### Training with Custom Rewards

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

### Guiding Behavior with Rules

#### `compute_action_mask(transformed_sensors, action)`

Returns a list of 0s and 1s for each discrete action — 0 = forbidden, 1 = allowed. The mask can change every step, enabling complex conditional rules.

**Important:** Action masks work only for **discrete** action spaces. They are ignored for continuous action spaces. Since selectors always have discrete action spaces (choosing a child skill), masks always apply to selectors.

```python
def compute_action_mask(self, transformed_sensors, action):
    return [0, 1, 1]  # First action forbidden; second and third allowed
```

### Managing Information Flow

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
