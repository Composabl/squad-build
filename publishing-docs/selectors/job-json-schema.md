# Orchestrator Job JSON Schema

Serialized orchestrators use `AgentOrchestratorSchema`-style blocks.

## Teacher orchestrator shape (`AgentOrchestrator`)

```json
{
  "name": "my-orchestrator",
  "type": "AgentOrchestrator",
  "config": {
    "remote_address": null,
    "impl_cls": { "cls_name": "MyOrchestratorTeacher", "cls_module": "my_orchestrator.teacher", "cls_src": "<base64-pickle>", "cls_deps": [] },
    "impl_cls_data": { "guidance": null, "goals": [], "constraints": null },
    "order_type": "fixed",
    "repeat_type": "once",
    "children": ["child-agent-a", "child-agent-b"],
    "learning": { "training_cycles": 100, "train_batch_size": 4000, "rl_algo": "PPO" },
    "resources": { "workers": 1, "learner_workers": 0, "envs_per_worker": 1 },
    "model": { "checkpoint_uri": "/tmp/amesa", "fc_layers": [256, 256] },
    "model_io": {},
    "scenarios": []
  }
}
```

## Controller orchestrator shape (`AgentOrchestratorController`)

```json
{
  "name": "my-orchestrator",
  "type": "AgentOrchestratorController",
  "config": {
    "remote_address": null,
    "impl_cls": { "cls_name": "MyOrchestratorController", "cls_module": "my_orchestrator.controller", "cls_src": "<base64-pickle>", "cls_deps": [] },
    "impl_cls_data": { "guidance": null, "goals": [], "constraints": null },
    "order_type": "fixed",
    "repeat_type": "once",
    "children": ["child-agent-a", "child-agent-b"],
    "learning": { "needs_warmup": false },
    "resources": {},
    "model": {},
    "model_io": {},
    "scenarios": []
  }
}
```

## Field intent

- `type`: orchestrator runtime type (`AgentOrchestrator` or `AgentOrchestratorController`).
- `children`: ordered child agent names; selected index maps to this order.
- `order_type`: child selection ordering mode (`fixed` or `variable`).
- `repeat_type`: replay behavior (`once` or `repeat`).
- `impl_cls` / `remote_address`: local serialized implementation or remote plugin source.
