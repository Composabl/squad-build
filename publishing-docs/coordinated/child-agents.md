# Managing Child Agents

`AgentCoordinatedSet` and `AgentCoordinatedPopulation` manage child agents with explicit registration.

## Common operations

```python
coordinated.add_agent(child_agent)
coordinated.get_agents()
coordinated.get_agent_names()
```

## Registration rule

Child names must match the keys returned by `AgentCoach.compute_reward(...)`.

## Population variant

`AgentCoordinatedPopulation` extends coordinated training with per-agent population counts while preserving coach/key alignment.
