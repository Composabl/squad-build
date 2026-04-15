## What AMESA Builds

AMESA is a platform for building **multi-agent systems** that use Machine Teaching. An agent system is composed of:

- **Skill agents** — the decision-making units. Each skill handles a specific sub-task.
- **Orchestrators (selectors)** — special skills that route control to the right skill agent based on current conditions.
- **Perceptors** — a perception layer that processes raw sensor data into richer variables before it reaches the skills layer.
- **Scenarios** — defined situations (variable configurations) that the agent system must learn to handle.
- **A simulation environment** — the training ground where agents practice via episodes.

The workflow is: connect a simulator → define perception → define skills → orchestrate skills → configure scenarios → train → evaluate → deploy.

## Instructions overview

#### Programmed Skills (Controllers)

Controllers are deterministic, code-based skill agents. They are useful for well-understood sub-tasks where you want to use optimization, PID control, MPC, heuristics, or API calls.

#### Orchestrating Skill Agents

Orchestration defines how multiple skill agents work together.

#### Perception Layer (Perceptors)

The perception layer sits before the skills layer in the agent system architecture. Each perceptor inputs sensor variables, processes them, and outputs **new derived variables** that are automatically added to the sensor list.

Perceptors can use any Python function or library, including ML models and LLM APIs.

#### Scenarios

Scenarios carve the simulation space into named situations defined by specific variable configurations. They enable targeted training — each skill practices only the scenarios relevant to it — and help orchestrators learn which skill to activate under which conditions.

#### Simulation Environment

AMESA agents train inside a simulation. The simulation API extends the **Gymnasium `gymnasium.Env`** standard. You integrate your simulator by implementing the `ServerAmesa` class, which defines how the SDK talks to your simulator.

#### Learned Skills (Teachers)

Learned skills use DRL. You configure a **teacher** that provides reward signals, termination conditions, success criteria, and optional rules. The agent practices in simulation until it achieves competence.

#### Trainer API & Configuration

The **Trainer** orchestrates the entire training lifecycle: spinning up simulation workers, collecting experience, updating policies, and managing checkpoints. You configure training via a config dict or TrainerConfig object, then call `trainer.train(agent, train_cycles=N)`. **This framework only builds V2 agents**, so examples below assume `target.v2`.

## Terminology Quick Reference

| Term                  | Definition                                                           |
| --------------------- | -------------------------------------------------------------------- |
| **Sensor variables**  | Observations coming from the simulator — the agent's inputs          |
| **Action space**      | The set of possible actions the agent can take                       |
| **Observation space** | The set of possible observations the sim can return                  |
| **Episode**           | One complete run of the simulation from reset to termination         |
| **Reward**            | Numeric feedback after each action telling the agent how well it did |
| **Coach**             | The teaching construct used for coordinated (multi-agent) skills     |

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
