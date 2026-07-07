# Coordinated Agents Overview

Coordinated agents orchestrate multiple child agents under a coach.

Core types:

- `AgentCoach`: reward/success/termination coordinator
- `AgentCoordinatedSet`: fixed set of child agents
- `AgentCoordinatedPopulation`: child agents with population counts

Use coordinated agents when sub-policies should learn independently but act jointly.
