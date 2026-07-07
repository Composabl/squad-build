# Coordinated Job JSON Schema

Coordinated entries are serialized in `agents_coordinated[]` (e.g., `AgentCoordinatedSchema`-style blocks).

## Core shape

```json
{
  "name": "team-agent",
  "type": "AgentCoordinatedSet",
  "config": {
    "remote_address": null,
    "impl_cls": { "cls_name": "MyCoach", "cls_module": "my_orchestration.coach", "cls_src": "<base64-pickle>", "cls_deps": [] },
    "agents": [],
    "scenarios": []
  }
}
```

## Key meaning

- `type`: `"AgentCoordinatedSet"` or `"AgentCoordinatedPopulation"`.
- `impl_cls`: serialized coach.
- `agents`: child agent metadata list.
- `scenarios`: coordinated scenario list.
