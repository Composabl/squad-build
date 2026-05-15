# Scenario Schema

Scenarios define named training situations — specific variable configurations that
carve the simulation space into distinct conditions. They enable curriculum learning:
skills practice easy scenarios first and advance to harder ones as success criteria
are met.

## Import

```python
from amesa_core.agent.scenario import Scenario
# or:
from amesa_core import Scenario
```

## Constructor

```python
Scenario(variables: dict)
```

`variables` is a dict where each key is a sensor/variable name and each value
describes the constraint:

```python
# Exact value
Scenario({"temperature": {"data": 80,           "type": "is_equal"}})

# Range (continuous)
Scenario({"temperature": {"data": [70, 90],     "type": "is_between"}})

# Discrete set
Scenario({"mode":        {"data": ["low", "high"], "type": "is_element_of"}})

# Shorthand list for range
Scenario({"temperature": [70, 90]})
```

## Variable Types

| `type` value | `data` format | Meaning |
|---|---|---|
| `"is_equal"` | single value | Variable equals exactly this value |
| `"is_between"` | `[low, high]` | Variable is sampled from this range |
| `"is_element_of"` | `[v1, v2, ...]` | Variable is one of these discrete values |
| (shorthand) | `[low, high]` | Treated as `"is_between"` |

## Adding Scenarios to a Skill

```python
skill = Skill("temperature-control", MyTeacher)

skill.add_scenario(Scenario({"temperature": {"data": 60, "type": "is_equal"}}))   # easy
skill.add_scenario(Scenario({"temperature": {"data": [60, 100], "type": "is_between"}}))  # harder
```

Skills advance through scenarios in order as `compute_success_criteria()` returns
True consistently. Knowledge accumulates across scenarios — later scenarios are
learned faster.

## Passing Scenarios to `set_scenario()`

The AMESA runtime calls `sim.set_scenario(scenario_dict)` before each episode.
Your simulator receives the scenario as a plain dict and should use it to configure
initial conditions:

```python
async def set_scenario(self, scenario):
    self.env.scenario = scenario

async def reset(self):
    obs, info = self.env.reset()
    # Apply scenario to initial conditions
    if self.env.scenario:
        temp = self.env.scenario.get("temperature", {}).get("data", 80)
        self.env.set_initial_temperature(temp)
    return obs, info
```

## Scenario Flows

Scenario flows define ordered sequences — useful when training order matters:

```python
# Without flows: AMESA connects scenarios at random
# With flows: AMESA follows a defined order

# Example: train at low wind, then land in the same conditions
skill.add_scenario(Scenario({"windspeed": 20}))   # step 1
skill.add_scenario(Scenario({"windspeed": 40}))   # step 2 — only after step 1 mastered
```

## Multiple Variables

Scenarios can constrain multiple variables simultaneously:

```python
Scenario({
    "temperature": {"data": 80,       "type": "is_equal"},
    "pressure":    {"data": [4, 6],   "type": "is_between"},
    "mode":        {"data": "normal", "type": "is_equal"},
})
```

## Scenario Assignment to Skills

Not every skill needs every scenario. A landing skill doesn't practice takeoff
scenarios. Assign only the relevant scenarios to each skill:

```python
# Takeoff skill
takeoff = Skill("takeoff", TakeoffTeacher)
takeoff.add_scenario(Scenario({"wind": [0, 20]}))

# Landing skill
landing = Skill("landing", LandingTeacher)
landing.add_scenario(Scenario({"wind": [0, 40]}))    # landing needs wider wind range
landing.add_scenario(Scenario({"terrain": "rough"}))

agent.add_skill(takeoff)
agent.add_skill(landing)
```

## JSON Serialization

Scenario dicts must be JSON-serializable (no numpy arrays, no custom objects) so
they can flow through trainer configs and be stored in historian outputs.

## `from_json` / `to_json`

```python
# From plain dict
scenario = Scenario.from_json({"temperature": {"data": 80, "type": "is_equal"}})

# To plain dict
d = scenario.to_json()
```

## Relationship to `SkillTeacher`

The current scenario is available inside a teacher via `self.scenario`:

```python
class MyTeacher(SkillTeacher):
    def add_scenario(self, scenario):
        super().add_scenario(scenario)
        # React to scenario — e.g., update target setpoint
        self.target = scenario.variables.get("temperature", {}).get("data", 80)
```

`add_scenario()` is called by the framework before training starts, not per step.

## Discrete Variables from Perceptors

Discrete scenario variables often come from a perceptor that transforms raw sensors
into categorical outputs (e.g., an ML classifier that maps sensor readings to
`"windy"`, `"calm"`, `"stormy"`).

```python
Scenario({"weather_mode": {"data": ["windy", "stormy"], "type": "is_element_of"}})
```

The perceptor produces `"weather_mode"` as a derived feature; the scenario specifies
which modes the skill should train on.
