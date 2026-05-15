# Agent Schema — `Agent`

`Agent` is the top-level builder that wires sensors, perceptors, skills, and selectors
into a single trainable agent system. Build one agent per agent system.

## Import

```python
from amesa_core.agent.agent import Agent
# or:
from amesa_core import Agent
```

## Constructor

```python
Agent()
```

No constructor arguments. All configuration is done through builder methods.

## Builder Methods

### Sensors

```python
agent.add_sensors(sensors: list[Sensor])
```

Register all sensor variables the agent can observe. Call once, before adding skills.
Sensors must match the observation keys returned by the simulator.

```python
agent.add_sensors([
    Sensor("air_temp",    "Air temperature (C)",    lambda obs: obs["air_temp"]),
    Sensor("target_temp", "Target temperature (C)", lambda obs: obs["target_temp"]),
    Sensor("humidity",    "Relative humidity",      lambda obs: obs["humidity"]),
])
```

### Perceptors

```python
agent.add_perceptor(perceptor: Perceptor)
```

Add perceptors in the order you want them to run. Each perceptor receives the full
sensor dict (including outputs from prior perceptors).

```python
agent.add_perceptor(Perceptor("velocity",   VelocityPerceptor))
agent.add_perceptor(Perceptor("energy",     EnergyPerceptor))    # can read "velocity"
```

### Skills

```python
agent.add_skill(skill)
```

Accepts: `Skill`, `SkillGroup` (registers both children and the group atomically).

```python
agent.add_skill(Skill("approach", ApproachTeacher))
agent.add_skill(Skill("setpoint-ctrl", SetpointController))
agent.add_skill(SkillGroup(plan_skill, execute_skill))   # preferred group registration
```

### Skill Groups

```python
agent.add_skill_group(skill_group: SkillGroup)
```

Adds the group to `agent.skill_groups`. Does NOT register children in `agent.skills`.
Always prefer `agent.add_skill(SkillGroup(...))` to avoid the registration trap.

### Selectors

```python
agent.add_selector_skill(selector: SkillSelector, ...)
```

### Coordinated Skills

```python
agent.add_coordinated_skill(coordinated: SkillCoordinatedSet | SkillCoordinatedPopulation)
```

Children go into `agent.skills_coordinated`, not `agent.skills`.

### Initialization

```python
await agent.init()
```

Must be called before training or inference. Downloads portable perceptors, validates
the agent graph, caches `filtered_sensor_space()` results.

## Complete Build Pattern

```python
from amesa_core import Agent, Skill, Sensor, Perceptor, Scenario

def build_agent() -> Agent:
    agent = Agent()

    # 1. Sensors
    agent.add_sensors([
        Sensor("air_temp",    "Air temperature",   lambda obs: obs["air_temp"]),
        Sensor("target_temp", "Target temp",        lambda obs: obs["target_temp"]),
        Sensor("humidity",    "Relative humidity",  lambda obs: obs["humidity"]),
        Sensor("target_humidity", "Target humidity", lambda obs: obs["target_humidity"]),
        Sensor("ambient_temp","Ambient temp",        lambda obs: obs["ambient_temp"]),
        Sensor("ambient_humidity","Ambient humidity",lambda obs: obs["ambient_humidity"]),
    ])

    # 2. Perceptors
    agent.add_perceptor(Perceptor("climate-errors", ClimateErrorPerceptor))

    # 3. Skills
    skill = Skill(
        "greenhouse-climate",
        GreenhouseTeacher,
        training_cycles=500,
        fc_layers=[128, 128],
    )
    skill.add_scenario(Scenario({"target_temp": {"data": [18, 24], "type": "is_between"}}))
    agent.add_skill(skill)

    return agent
```

## Agent Collections

| Collection | Accessed via | Contents |
|---|---|---|
| `agent.sensors` | `agent.sensors` | All registered `Sensor` objects |
| `agent.perceptors` | `agent.perceptors` | All registered `Perceptor` wrappers |
| `agent.skills` | `agent.get_skills()` or `agent.skills.values()` | Skills and selectors (not coordinated) |
| `agent.skill_groups` | `agent.skill_groups` | `SkillGroup` objects |
| `agent.skills_coordinated` | `agent.skills_coordinated` | Coordinated skill sets/populations |

`agent.get_all_skills()` returns `skills + skill_selectors + skills_coordinated`. It
does NOT include group member skills unless they were also registered via `add_skill()`.

## Training

```python
from amesa_core.trainer import Trainer

trainer = Trainer(config={
    "target": {
        "sim": "http://localhost:5000",  # or "local" for in-process
    }
})

await trainer.train(agent, train_cycles=200)
```

## Inference

After training, load a checkpoint and run inference:

```python
from ray.rllib.algorithms.ppo import PPO

policy = PPO.from_checkpoint("benchmarks/my-skill/policy_checkpoint_20260423/")
action = policy.compute_single_action(observation)[0]
```

## Rule of Thumb for Build Order

```
1. add_sensors()          ← define what the sim provides
2. add_perceptor()        ← enrich with derived features (optional)
3. build skills           ← create Skill objects, add scenarios
4. add_skill()            ← register skills (and groups atomically)
5. add_coordinated_skill()← for multi-agent only
6. await agent.init()     ← validate and prepare
```

## Common Mistakes

| Mistake | Consequence | Fix |
|---|---|---|
| `add_skill_group()` without prior `add_skill()` | Group members invisible to training | Use `agent.add_skill(SkillGroup(...))` instead |
| Missing sensor in `filtered_sensor_space()` | Key silently missing from policy obs | Add the sensor name to the list |
| Perceptor registered after skill that depends on it | Order violation — derived key missing | Register perceptors before adding skills |
| Teacher `__init__` requires arguments | Crash on episode reset | Teacher `__init__` must accept zero arguments |
