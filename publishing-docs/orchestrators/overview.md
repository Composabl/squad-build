# Orchestrator Overview

`AgentOrchestrator` chooses one child agent per step and routes execution to that selected child.

## What an orchestrator does

- Builds an orchestrator action space of `Discrete(len(children))`.
- Produces a child index (0-based) via orchestrator logic.
- Executes the selected child agent and returns the **child action**.

## Orchestrator implementation modes

- **Teacher orchestrator** (`AgentOrchestrator`): ML policy learns which child to choose.
- **Controller orchestrator** (`AgentOrchestratorController`): deterministic logic chooses child index.

## SDK compliance essentials

- Orchestrator must have at least one child agent.
- `children` must be agent names that exist in the orchestration graph.
- Orchestrator output must resolve to a valid child index (`0..len(children)-1`).
- Portable package type must be:
  - `orchestrator-teacher` for teacher orchestrators
  - `orchestrator-controller` for controller orchestrators
