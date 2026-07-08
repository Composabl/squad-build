# Orchestrator Publishing

Use `amesa orchestrator` commands for orchestrator lifecycle: scaffold, package, and publish.

## Portable package config (`pyproject.toml`)

Teacher orchestrator:

```toml
[project]
name = "my-orchestrator"
version = "0.1.0"
dependencies = ["amesa-core"]

[amesa]
type = "orchestrator-teacher"
entrypoint = "my_orchestrator.teacher:Teacher"
```

Controller orchestrator:

```toml
[project]
name = "my-orchestrator"
version = "0.1.0"
dependencies = ["amesa-core"]

[amesa]
type = "orchestrator-controller"
entrypoint = "my_orchestrator.controller:Controller"
```

## Compliance checklist

- `[amesa].type` is exactly `orchestrator-teacher` or `orchestrator-controller`.
- `[amesa].entrypoint` is `module.path:ClassName`.
- Orchestrator class implements the correct base interface (teacher/controller).
- Orchestrator chooses valid child indices and is configured with at least one child.
