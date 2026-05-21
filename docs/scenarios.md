# Scenarios

A `Scenario` defines the **initial state** a simulator starts in for a training episode. Providing multiple scenarios trains the skill across diverse conditions.

## Scenario value types

| Python value | Behavior |
|---|---|
| `scalar` (int/float) | Constant — always this value |
| `[low, high]` (list of 2 numbers) | Sampled uniformly from `[low, high]` each episode |

## Defining scenarios as dicts

```python
from amesa_core.agent.scenario.scenario import Scenario

scenario = Scenario({
    "initial_temp": 20.0,          # constant
    "target_temp": [21.0, 24.0],   # sampled each episode
    "ambient_temp": [10.0, 15.0],
})
```

The dict keys must match the state keys your simulator reads in `set_scenario()`.

## Adding scenarios to a skill

```python
skill = Skill("my-skill", MyTeacher)
skill.add_scenario(Scenario({"x": [0.0, 1.0]}))
skill.add_scenario(Scenario({"x": [5.0, 10.0]}))

# Or pass a list at once
skill.add_scenario([scenario_a, scenario_b])

# Or pass a raw dict — it is converted automatically
skill.add_scenario({"x": 0.5})
```

## Receiving scenarios in the sim

Your `ServerAmesa.set_scenario(scenario: dict)` receives the **sampled** scenario dict (scalars only — ranges have already been resolved). Apply it to your environment state there.

```python
async def set_scenario(self, scenario: dict):
    self.env.set_scenario(scenario)
```

## Optional: name and description

```python
Scenario({"x": 1.0}, name="center", description="Starts at center position")
```

---

## ⚠️ Quirks

**Range format** — Only `[low, high]` (exactly 2 numbers) is treated as a range. A list with one element or a tuple is handled differently. Stick to `[low, high]`.

**Key collision** — `flow_ids` is a reserved key in the scenario dict (used internally). Do not use it as a state variable name.

**No scenarios = training starts from whatever `reset()` returns** — this is valid but reduces training diversity.
