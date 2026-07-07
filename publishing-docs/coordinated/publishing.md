# Coordinated Publishing

Portable coordinated package example:

```toml
[project]
name = "my-coordinated-agent"
version = "0.1.0"
dependencies = ["amesa-core"]

[amesa]
type = "agent-coordinated-set"
entrypoint = "my_module.coach:MyCoach"
```

## Requirements

- `type` must be `"agent-coordinated-set"` or `"agent-coordinated-population"`.
- `entrypoint` points to the `AgentCoach` subclass.
- Use top-level `[amesa]` (not `[tool.amesa]`).
