# Orchestrator Quirks

- Orchestrator package type names (`orchestrator-teacher`, `orchestrator-controller`) differ from runtime JSON type names (`AgentOrchestrator`, `AgentOrchestratorController`).
- Orchestrator action space is derived from children count (`Discrete(n)`), so child list order directly defines index semantics.
- Orchestrators require at least one child; empty `children` fails at initialization.
- Teacher orchestrator masks must match child count; mismatched shapes break routing.
- Runtime output is the selected child's action, not the orchestrator index itself.
