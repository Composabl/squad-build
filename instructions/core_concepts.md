## Core Concepts

### What AMESA Builds

AMESA is a platform for building **multi-agent systems** that use Machine Teaching. An agent system is composed of:

- **Skill agents** — the decision-making units. Each skill handles a specific sub-task.
- **Orchestrators (selectors)** — special skills that route control to the right skill agent based on current conditions.
- **Perceptors** — a perception layer that processes raw sensor data into richer variables before it reaches the skills layer.
- **Scenarios** — defined situations (variable configurations) that the agent system must learn to handle.
- **A simulation environment** — the training ground where agents practice via episodes.

The workflow is: connect a simulator → define perception → define skills → orchestrate skills → configure scenarios → train → evaluate → deploy.

### Skill Agent Types

| Type                              | Also Called            | How It Decides                                                    |
| --------------------------------- | ---------------------- | ----------------------------------------------------------------- |
| **Teacher** (learned skill)       | Learned skill agent    | Deep reinforcement learning                                       |
| **Controller** (programmed skill) | Programmed skill agent | Code — math, rules, optimization, MPC, PID, heuristics, API calls |
| **Selector** (orchestrator)       | Orchestrator           | Learned (DRL) or programmed                                       |

Teachers learn by practicing in simulation. Controllers execute predetermined logic. Selectors decide which child skill should be active at any given moment.

### Terminology Quick Reference

| Term                 | Definition                                                               |
| -------------------- | ------------------------------------------------------------------------ |
| **Sensor variables** | Observations coming from the simulator — the agent's inputs              |
| **Action space**     | The set of possible actions the agent can take                           |
| **Episode**          | One complete run of the simulation from reset to termination             |
| **Reward**           | Numeric feedback after each action telling the agent how well it did     |
| **Perceptor**        | A module that transforms raw sensors into new, derived variables         |
| **Scenario**         | A named configuration of variable values/ranges representing a situation |
| **Coach**            | The teaching construct used for coordinated (multi-agent) skills         |

## Data Flow Summary

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
