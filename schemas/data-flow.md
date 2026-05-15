# Data Flow Reference

The complete path data takes from simulator to action, in order. Every component in
AMESA touches one stage of this pipeline. Read this before building any component.

## Per-Step Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  SIMULATOR                                                          │
│  gym.Env subclass — returns raw obs dict, reward, terminated, info  │
└────────────────────────┬────────────────────────────────────────────┘
                         │  raw obs: OrderedDict / gym.Space sample
                         │  (gRPC or in-process)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  SENSOR MAPPING                                                     │
│  map_sim_sensors_to_amesa_sensors()                                 │
│  Applies per-Sensor lambda: sensors["name"] = lambda(raw_obs)       │
│  Output: amesa_sensors = { sensor_name: value, ... }               │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PERCEPTOR PIPELINE                                                 │
│  Perceptors run in registration order                               │
│  Each receives full amesa_sensors dict, returns new keys to add     │
│  Key collisions raise an exception                                  │
│  Perceptors are reset (re-instantiated) at each episode boundary    │
└────────────────────────┬────────────────────────────────────────────┘
                         │  enriched amesa_sensors dict
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TEACHER / CONTROLLER (per active Skill)                            │
│                                                                     │
│  Teacher path (RL):                                                 │
│    teacher.transform_sensors(amesa_sensors, None)                   │
│      → transformed_sensors                                          │
│    filter_sample(transformed_sensors, filtered_sensor_space())      │
│      → filtered subset of keys                                      │
│    sensor.normalize_sample() per key                                │
│      → flat numpy array → RL model (Ray RLlib)                     │
│    teacher.compute_action_mask(transformed, prev_action)            │
│      → optional mask for discrete actions                           │
│    RL model → raw_action                                            │
│    teacher.transform_action(transformed_sensors, raw_action)        │
│      → processed_action → Simulator                                 │
│                                                                     │
│  Controller path (deterministic):                                   │
│    convert_sim_sensors_to_amesa_sensors() (full pipeline above)     │
│    controller.compute_action(amesa_obs, prev_action)                │
│      → action (coerced by sim_action_space.coerce_sample())         │
│      → Simulator                                                    │
└────────────────────────┬────────────────────────────────────────────┘
                         │  sim_reward, sim_terminated, new_obs
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING FEEDBACK (Teacher only)                                   │
│  teacher.compute_reward(transformed, action, sim_reward) → float    │
│  teacher.compute_success_criteria(transformed, action) → bool       │
│  teacher.compute_termination(transformed, action) → bool            │
└─────────────────────────────────────────────────────────────────────┘
```

## Orchestration Layer

When a `SkillSelector` is active, it sits between the perceptor pipeline output and the
individual skill execution:

```
Enriched amesa_sensors
        │
        ▼
SkillSelector (Teacher or Controller)
  compute_action() → int index
        │
        ▼
Child Skill[index] executes full Teacher/Controller pipeline above
        │
        ▼
Action → Simulator
```

## Skill Group (Plan → Execute)

```
Enriched amesa_sensors
        │
        ├──────────────────────────────┐
        ▼                              │
Skill A (first_skill)                 │
  compute_action() → action_A         │
        │                              │
        │  injected into obs dict       │
        │  as key = first_skill.name    │
        ▼                              ▼
Skill B sensor dict:
  { ...original sensors..., "skill-a": action_A }
Skill B teacher runs against enriched dict
  → final action → Simulator
```

## Coordinated Skills (Parallel Multi-Agent)

```
Sim returns Dict obs space keyed by skill name
        │
        ├── sim_sensors["skill_a"] ──► Skill A → action_a ─┐
        ├── sim_sensors["skill_b"] ──► Skill B → action_b ─┼──► Joint Dict → Sim
        └── sim_sensors["skill_c"] ──► Skill C → action_c ─┘
                                                             │
                               coach.compute_reward(multi_obs, multi_action, multi_reward)
                                 → float bonus added equally to all children
```

## Episode Lifecycle

```
agent.init()
  └── filtered_sensor_space() called once per Teacher/Controller → cached

Episode start:
  └── sim.reset() → first obs
  └── Teacher/Controller re-instantiated (new __init__() call)
  └── Perceptors re-instantiated

Per step:
  └── pipeline above runs once

Episode end (one of):
  └── teacher.compute_termination() → True   (failure; success_counter--)
  └── teacher.compute_success_criteria() → True AND NOT termination
      (success; success_counter++)
  └── sim returns terminated=True
```

## What Each Component Owns

| Stage | Component | File to Build |
|---|---|---|
| Sim interface | `ServerAmesa` subclass | `server_impl.py` |
| Sensor mapping | `Sensor` definitions | `build_agent.py` or `sensors.py` |
| Feature engineering | `PerceptorImpl` subclass | `perceptors.py` |
| RL reward/success | `SkillTeacher` subclass | `teacher.py` |
| Deterministic policy | `SkillController` subclass | `controller.py` |
| Multi-agent coordination | `SkillCoach` subclass | `coach.py` |
| Skill routing | `SkillSelector` + impl | `build_agent.py` |
| Agent wiring | `Agent` builder function | `build_agent.py` |
| Training situations | `Scenario` dicts | `scenarios.py` |
