# Agent Reference

Agent-facing reference for building components with the AMESA SDK (v1).

## Quick Index

| Doc                                            | What it covers                                                  |
| ---------------------------------------------- | --------------------------------------------------------------- |
| [sensors.md](sensors.md)                       | `Sensor` — extract values from sim observations                 |
| [perceptors.md](perceptors.md)                 | `PerceptorImpl` — transform observations before skills see them |
| [scenarios.md](scenarios.md)                   | `Scenario` — define initial conditions for training episodes    |
| [spaces.md](spaces.md)                         | `Box`, `Discrete`, `Dict`, … — action/sensor space types        |
| [skills/overview.md](skills/overview.md)       | Skill types and implementation types                            |
| [skills/teacher.md](skills/teacher.md)         | `SkillTeacher` — ML-trained skill                               |
| [skills/controller.md](skills/controller.md)   | `SkillController` — deterministic/rule-based skill              |
| [skills/selector.md](skills/selector.md)       | `SkillSelector` — route to the right skill                      |
| [skills/group.md](skills/group.md)             | `SkillGroup` — pipeline two skills in sequence                  |
| [skills/coordinated.md](skills/coordinated.md) | `SkillCoach` / `SkillCoordinated` — multi-agent coordination    |
| [agent.md](agent.md)                           | `Agent` — assemble everything                                   |
| [simulator.md](simulator.md)                   | `ServerAmesa` — implement a simulator                           |
| [trainer.md](trainer.md)                       | `Trainer` + v1 config — run training                            |
| [goals.md](goals.md)                           | `ApproachGoal`, `MaximizeGoal`, … — v1 goal system              |

## Minimal v1 training checklist

1. Implement your sim (`ServerAmesa` subclass) and expose it via `server_make.make()`
2. Define `Sensor`s that map sim observations to named values
3. Implement a `SkillTeacher` (or `SkillController`) for each skill
4. Optionally add `PerceptorImpl`s for observation pre-processing
5. Create `Scenario`s for training variety
6. Call `build_agent()` to assemble the `Agent`
7. Run `Trainer(config).train(agent, train_cycles=N)` with a v1 config dict

> **v1 goal system** — v1 includes `ApproachGoal`, `MaximizeGoal`, `MinimizeGoal`, `AvoidGoal`, `MaintainGoal`, and `CoordinatedGoal` classes as a higher-level alternative to implementing `SkillTeacher` reward methods manually. See [goals.md](goals.md).

> **Coordinated skills** — v1 fully supports `SkillCoordinatedSet` and `SkillCoordinatedPopulation` for multi-agent coordination. See [skills/coordinated.md](skills/coordinated.md).

## Import cheatsheet

```python
from amesa_core.agent.agent import Agent
from amesa_core.agent.sensors.sensor import Sensor
from amesa_core.agent.perceptor.perceptor import Perceptor
from amesa_core.agent.perceptor.perceptor_impl import PerceptorImpl
from amesa_core.agent.skill.skill import Skill
from amesa_core.agent.skill.skill_teacher import SkillTeacher
from amesa_core.agent.skill.skill_controller import SkillController
from amesa_core.agent.skill.skill_selector import SkillSelector
from amesa_core.agent.skill.skill_group import SkillGroup
from amesa_core.agent.skill.skill_coach import SkillCoach
from amesa_core.agent.scenario.scenario import Scenario
from amesa_core.spaces import Box, Discrete, Dict
from amesa_core.networking.sim import server as server_make
from amesa_core.networking.sim.server_amesa import ServerAmesa
from amesa_train import Trainer
```
