# AMESA API Schemas

Precise interface contracts for every buildable component in the AMESA SDK. Each schema
defines: import path, required methods with signatures and return types, optional hooks,
registration pattern, and critical behavioral notes. Use these as ground truth when
generating or reviewing AMESA component code.

## Component Index

| Schema | What it defines |
|---|---|
| [data-flow.md](data-flow.md) | End-to-end data pipeline — read this first |
| [simulator.md](simulator.md) | `ServerAmesa` — the sim/environment interface |
| [sensor.md](sensor.md) | `Sensor` — sensor definitions and registration |
| [perceptor.md](perceptor.md) | `PerceptorImpl` — perception / feature engineering layer |
| [teacher.md](teacher.md) | `SkillTeacher` — RL reward / success / termination contract |
| [goals.md](goals.md) | `Goal` subclasses — declarative alternative to Teacher |
| [controller.md](controller.md) | `SkillController` — deterministic / hand-coded policies |
| [coach.md](coach.md) | `SkillCoach` — multi-agent coordination coordinator |
| [skill.md](skill.md) | `Skill`, `SkillSelector`, `SkillGroup` — skill wrappers |
| [coordinated-skills.md](coordinated-skills.md) | `SkillCoordinatedSet`, `SkillCoordinatedPopulation` |
| [scenario.md](scenario.md) | `Scenario` — curriculum / training variable configuration |
| [agent.md](agent.md) | `Agent` — top-level builder that wires everything together |

## Quick Decision Guide

```
Need to…                                    Use
─────────────────────────────────────────── ──────────────────────────────
Connect a simulator                         ServerAmesa (simulator.md)
Define what sensors the sim provides        Sensor (sensor.md)
Enrich observations before skills see them  PerceptorImpl (perceptor.md)
Train a skill with RL                       SkillTeacher (teacher.md)
Declare objective without custom reward     Goal subclass (goals.md)
Write deterministic / PID / rule-based ctrl SkillController (controller.md)
Run multiple skills in parallel             SkillCoach + Coordinated (coach.md)
Route between skills based on conditions    SkillSelector (skill.md)
Chain plan→execute between two skills       SkillGroup (skill.md)
Define training situations / curricula      Scenario (scenario.md)
Compose everything into a runnable agent    Agent (agent.md)
```

## Module Roots

| Package | Purpose |
|---|---|
| `amesa_core` | Core SDK — Agent, Skill, Sensor, Scenario, SkillTeacher, SkillController, SkillCoach |
| `amesa_core.networking.sim` | Simulator server base (`ServerAmesa`) |
| `amesa_core.agent.skill.goals` | Goal subclasses (MaintainGoal, MaximizeGoal, etc.) |
| `composabl_core` | Alternate import root used by some older SDK paths (same package) |
