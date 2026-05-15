# Controller Skill

## Overview

A **Controller skill** is a hand-coded, deterministic (or stochastic) policy that runs in the same per-timestep simulation loop as a trained RL skill — but is **never trained**. It encodes explicit logic: PID controllers, rule-based policies, mathematical expressions, scripted heuristics. The AMESA runtime treats it as a first-class skill: it receives sensor observations, produces actions, and occupies a node in the agent's DAG. It is never touched by the RL machinery.

**When to use a Controller instead of a Teacher:**

| Use Case | Controller | Teacher |
|---|---|---|
| You have a known policy (PID, rule-based, expert system) | ✓ | |
| You need a benchmark baseline for RL to beat | ✓ | |
| You are building a combined expert + RL agent | ✓ | |
| You need learned reward shaping and RL training | | ✓ |
| You need ONNX model export and checkpointing | | ✓ |

Controllers are commonly used as the **baseline** an RL agent is benchmarked against. They can also be combined with RL skills in the same agent DAG — the DAG trains RL skills in topological order while controllers run unchanged.

---

## Architecture

### `SkillController` Class Hierarchy

`SkillController` is an **abstract base class (ABC)**. It does **not** inherit from `SkillTeacher` — they are peer concepts on completely separate execution paths.

```
SkillController(ABC)
    └── (your subclass)
SkillControllerJson(SkillController)   ← dead stub, do not use (see Gotchas)
```

**Module path:** `amesa_core.agent.skill.skill_controller`

**Import:**

```python
from amesa_core import SkillController
# or explicit path:
from amesa_core.agent.skill.skill_controller import SkillController
```

**Base class definition:**

```python
class SkillController(ABC):
    def __init__(self, *args, **kwargs):
        self.action_space: "Space" = None
        self.sensor_space: "Space" = None
        self.scenario: Scenario = None
```

The constructor is intentionally permissive (`*args, **kwargs`). Your subclass `__init__` may accept any arguments.

### Relationship to Other SDK Concepts

| Concept | Relationship |
|---|---|
| **SkillTeacher** | Peer concept, mutually exclusive. Use Teacher for RL, Controller for hand-coded logic. |
| **SkillSelector** | A controller can occupy a `SkillSelector` node — returning an integer child index instead of a sim action. Registered as `SkillSelectorController`. |
| **SkillGroup** | Controllers can appear in a skill group, but the group pipeline for controllers is **stubbed** (see Gotchas §11). |
| **Training** | Controllers are excluded from the training order entirely — no Ray algorithm, no ONNX export, no checkpoint. |
| **Perceptors** | Applied to controller inputs. `convert_sim_sensors_to_amesa_sensors()` runs the full perceptor pipeline before `compute_action` is called. |
| **Scenarios** | Controllers accept scenarios for sensor space scoping at episode start, but do not train on them. |

### Processor Layer

The framework uses `make_skill_processor()` to dispatch between processor types at runtime:

```
SkillProcessorBase
├── SkillProcessor                       ← wraps SkillTeacher (DRL)
├── SkillControllerProcessor             ← wraps SkillController (local or remote)
│   └── SkillSelectorControllerProcessor ← controller returns child index
```

When `agent.add_skill()` is called with a `SkillController` subclass, `get_type_from_impl_cls()` detects `issubclass(impl_cls, SkillController)` and assigns `SkillDirector.CONTROLLER`. The correct processor is created automatically.

---

## Required Methods

All four methods are abstract — your subclass must implement all of them. All are `async`.

---

### `compute_action`

```python
@abstractmethod
async def compute_action(self, transformed_sensors: Dict, action) -> any:
```

The core control logic. Called once per simulation step.

| Parameter | Type | Description |
|---|---|---|
| `transformed_sensors` | `Dict[str, np.ndarray]` | Sensor dict keyed by sensor name. Values are numpy arrays from the sim, after lambda mapping and perceptors. |
| `action` | any | The action sent to the sim on the **previous** step. May be `None` on the first step of each episode. |

**Returns:** The action to send to the simulator. Must match the sim's action space type:

| Action Space | Expected Return |
|---|---|
| `Box` | `np.array([...], dtype=np.float32)` or `float` |
| `Discrete` | `int` |
| `MultiDiscrete` | `List[int]` |
| `Dict` | `dict` keyed by sub-space names |
| `Tuple` | `tuple` of sub-space values |

The return value is passed through `sim_action_space.coerce_sample()` — the framework performs type coercion and clamping to the action space for you. Teachers do not get this coercion; controllers do.

**Called by:** `SkillControllerProcessor.compute_single_action()` once per sim step.

---

### `filtered_sensor_space`

```python
@abstractmethod
async def filtered_sensor_space(self) -> List[str]:
```

Declares which sensors this controller needs to observe.

**Returns:** `List[str]` — sensor name keys this controller wants in its `transformed_sensors` dict.

**Called by:** `SkillControllerProcessor` at initialization. The result is cached — returning a dynamic list based on runtime state will only use the first call's result.

---

### `compute_success_criteria`

```python
@abstractmethod
async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
```

Returns whether the current episode should be considered a success.

| Parameter | Type | Description |
|---|---|---|
| `transformed_sensors` | `Dict[str, np.ndarray]` | Current sensor observations. |
| `action` | any | The current action. |

**Returns:** `bool`

> ⚠️ **Dead code in V1.** The V1 processor (`SkillControllerProcessor`) stubs this method with `pass` — returning `None`. Your implementation is never called in V1. This method **is** called in V2 (`EventControllerProcessor`). See [V1 vs V2 Differences](#v1-vs-v2-differences).

---

### `compute_termination`

```python
@abstractmethod
async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
```

Returns whether the current episode should be terminated early.

| Parameter | Type | Description |
|---|---|---|
| `transformed_sensors` | `Dict[str, np.ndarray]` | Current sensor observations. |
| `action` | any | The current action. |

**Returns:** `bool`

> ⚠️ **Dead code in V1.** Same as `compute_success_criteria` — the V1 processor stubs this with `pass`. Your early-termination logic will silently have no effect on episode control in V1. **It works in V2.**

---

## Optional Methods

### `transform_sensors`

```python
async def transform_sensors(self, sensors) -> Dict:
    return sensors   # base class default: pass-through
```

Intended to preprocess sensor observations before they are passed to `compute_action`.

**Returns:** Modified sensor dict.

> ⚠️ **Dead code in V1 and V2.** `SkillControllerProcessor.compute_single_action()` calls `convert_sim_sensors_to_amesa_sensors()` directly, then passes the result straight to `compute_action()`. `transform_sensors` is never invoked. Overriding this method will produce no effect in either V1 or V2. If you need to transform sensors, do it inside `compute_action`.
>
> Additionally, the base class signature is `transform_sensors(self, sensors)` (one argument), but the CLI template scaffold generates `transform_sensors(self, sensors, action)` (two arguments). These are inconsistent; the method is dead regardless.

---

## Registration

Register a controller skill the same way as any other skill — `agent.add_skill()` handles everything via type inference:

```python
from amesa_core import Agent
from amesa_core.agent.skill import Skill

agent = Agent()

skill = Skill("my-controller", MyController)
agent.add_skill(skill)
```

`Skill.__init__` calls `get_type_from_impl_cls(MyController)`, detects `SkillController`, and assigns `SkillDirector.CONTROLLER`. **No special registration method exists for controllers.**

### Selector Controller

If the controller's `compute_action` should return a **child index** (routing to one of N child skills), use `SkillSelector` instead:

```python
from amesa_core.agent.skill.skill_selector import SkillSelector

selector = SkillSelector(
    "my-selector",
    MySelectorController,
    children=["child-skill-1", "child-skill-2"]
)
agent.add_selector_skill(selector, ...)
```

The selector controller's `compute_action` must return an integer (or something castable to `int`) that is the 0-based index of the child skill to execute.

---

## `SkillControllerOptions`

When a controller is registered, the framework uses `SkillControllerOptions` — not `SkillDRLOptions`. This configuration has no learning, model, or resource fields.

**File:** `amesa_core/amesa_core/config/skill_config.py`

```python
class SkillControllerOptions(BaseModel):
    model_io: ModelIOConfig = Field(default_factory=ModelIOConfig)
    remote_address: Optional[str] = None
    impl_cls: Optional[Union[str, CLSSchema]] = None
    impl_cls_data: SkillImplDataSchema = SkillImplDataSchema()
    scenarios: Optional[List[ScenarioSchema]] = []
    scenarios_current_idx: Optional[int] = 0
```

There is no `LearningConfig`, `ModelConfig`, or `ResourcesConfig`. Controllers do not train.

> ⚠️ **Training kwargs are silently dropped.** Passing any training-related kwargs to `Skill()` when the impl class is a controller does nothing — no error, no warning:
>
> ```python
> # These kwargs are silently ignored for controllers:
> skill = Skill("my-ctrl", MyController, training_cycles=500, workers=4, fc_layers=[64, 64])
> ```
>
> `Skill.process_kwargs()` contains an early-return guard that skips all training configuration when `config.type == SkillDirector.CONTROLLER`.

---

## Runtime Behavior

### Per-Step Execution (V1)

```
sim step
  │
  ▼
SkillControllerProcessor._execute(sim_sensors, ...)
  │
  ▼
compute_single_action(sim_sensors, previous_action)
  │
  ▼
convert_sim_sensors_to_amesa_sensors()   ← sensor name mapping + perceptors applied
  │
  ▼
controller.compute_action(amesa_obs, action)   ← YOUR CODE called here
  │
  ▼
sim_action_space.coerce_sample(action)   ← type coercion / clamping
  │
  ▼
return action → sim
```

**What is NOT called in V1:**
- `controller.transform_sensors()` — never invoked
- `controller.compute_success_criteria()` — processor stubs it with `pass`
- `controller.compute_termination()` — processor stubs it with `pass`

### Episode Reset

For **local** controllers, reset means **full re-instantiation** every episode:

```python
# SkillControllerProcessor.reset() — local path
self.controller = skill.get_impl_cls_instance()   # new instance, __init__ re-runs
```

Any state accumulated during an episode is destroyed. If your controller has expensive `__init__` work (connections, file loads), it runs again on every episode. For cross-episode state, use class-level variables or external storage.

For **remote** controllers, `reset()` is called via RPC — the remote server recreates its instance properly.

### Training Integration

Controllers are completely excluded from the training order. `Agent.get_training_order()` filters by `skill.is_teacher()` — controllers never appear. What does happen during a training run:

- **Pre-training validation** (`inference_episode_benchmark`) includes controllers — they execute in sim episodes during the validation phase
- `adjust_resources()` is explicitly skipped for controllers
- After the first trained skill initializes and gets its sensor space, the framework propagates `sim_sensor_space` to all controller skills

> ⚠️ **Sensor space is inherited from the trained skill.** If the agent has no trainable skills, the controller's `sim_sensor_space` is never set. For mixed agents, the controller's sensor space is set from the first trained skill's space — not directly from the sim.

### Selector Controller Execution

When `SkillSelectorControllerProcessor` dispatches, the controller's `compute_action` return value is cast to `int` and used as an index:

```
SkillSelectorControllerProcessor._execute(sim_sensors)
  │
  ▼
super()._execute()  → controller.compute_action() → action (the child index)
  │
  ▼
int(action)   ← must be castable to int
  │
  ▼
child_skill_processors[index]._execute(sim_sensors)   ← selected child executes
  │
  ▼
return child_action → sim
```

The selector controller's `compute_action` must return an integer index. The selected child receives the same `sim_sensors` as the selector.

---

## V1 vs V2 Differences

| Aspect | V1 (Ray/RLlib) | V2 (Redis Streams) |
|---|---|---|
| **Processor class** | `SkillControllerProcessor` | `EventControllerProcessor` + `ControllerComponent` |
| **Reset mechanism** | Re-instantiates class each episode | Calls `reset()` on existing `ControllerComponent` per episode ID |
| **Multi-episode isolation** | Single instance, re-created on reset | Separate `ControllerComponent` per `unique_id` (episode ID) |
| **`compute_success_criteria`** | **Never called** — processor stubs `pass` | **Called** via `ControllerComponent.compute_success_criteria()` |
| **`compute_termination`** | **Never called** — processor stubs `pass` | **Called** via `ControllerComponent.compute_termination()` |
| **`transform_sensors`** | Never called | Never called (same dead code status) |
| **`transform_action`** | Never called | Never called (not in base class; see Gotchas) |
| **Files** | `amesa_train/amesa_train/skill_processors/skill_controller_processor.py` | `amesa_train/amesa_train/v2/controller_component.py`, `amesa_train/amesa_train/v2/event_controller_processor.py` |

**The critical practical difference:** `compute_success_criteria` and `compute_termination` control episode lifecycle in V2 but are completely inert in V1. If you write termination logic expecting it to end episodes early, it will silently do nothing in V1.

---

## ⚠️ Gotchas

### G1 — `transform_sensors` is never called (V1 and V2)

`SkillControllerProcessor.compute_single_action()` calls `convert_sim_sensors_to_amesa_sensors()` directly and passes the result to `compute_action()` — bypassing `transform_sensors` entirely. The base class default (`return sensors`) makes this silently harmless, but **any override you write will never execute**. Do sensor preprocessing inside `compute_action` instead.

### G2 — `transform_action` does not exist in the base class

`SkillController` ABC has no `transform_action` method. However:
- The CLI scaffold template generates `async def transform_action(self, transformed_sensors, action) -> float`
- The remote gRPC server exposes a `transform_action` RPC endpoint
- The remote isolation server calls `self.impl_cls_instance.transform_action()`

If your controller is local, defining `transform_action` is harmless dead code. If remote, the method is exposed via gRPC but **`SkillControllerProcessor` never calls it remotely either**. Do not rely on `transform_action` for any production logic.

### G3 — `compute_success_criteria` and `compute_termination` are dead code in V1

In `SkillControllerProcessor`, both methods are stubbed:

```python
@ensure_is_initialized
def compute_success_criteria(self, obs, action):
    pass   # returns None — your method is never called

@ensure_is_initialized
def compute_termination(self, obs, action):
    pass   # returns None — your method is never called
```

Writing episode-termination logic in these methods will have **zero effect on episode control in V1**. They work correctly in V2. If you are using V1 and need episode termination, it must be handled by the simulator.

### G4 — Training kwargs are silently dropped

```python
# All of these kwargs are silently ignored for controller skills:
Skill("ctrl", MyController, training_cycles=500, workers=4, fc_layers=[64, 64])
```

`Skill.process_kwargs()` skips all training configuration when `config.type == SkillDirector.CONTROLLER`. No exception, no warning, no log line. The kwargs disappear without trace.

### G5 — Local reset is full re-instantiation every episode

Every episode reset destroys the controller instance and calls `__init__` again:

```python
self.controller = skill.get_impl_cls_instance()
```

Consequences:
- All per-episode state is wiped at reset (correct behavior for stateless controllers)
- Expensive `__init__` work — database connections, file loads, heavy initialization — runs every episode
- Cross-episode state requires class-level (static) variables or external storage

For remote controllers, `reset()` is called via RPC — the server recreates the instance, which is the same net effect.

### G6 — `SkillControllerJson` is a dead stub

`amesa_core/amesa_core/agent/skill/skill_controller_json.py` defines `SkillControllerJson(SkillController)` with a TODO comment: *"Awaiting Logic Builder Implementation."* All methods are empty no-ops. This class is a placeholder for a planned no-code UI feature. **Do not use it in production.**

### G7 — Sensor space comes from the first trained skill

During training, `SkillTrainerBase` propagates `sim_sensor_space` to controller skills from the first trained skill — not from the sim directly. If the agent has no trainable skills, the controller's sensor space is never set. If your controller's sensor needs differ from the co-resident RL skill, verify the spaces match.

### G8 — Selector controller must return an integer

`SkillSelectorControllerProcessor` casts `compute_action`'s return to `int`:

```python
processed_selector_action = int(processed_selector_action)
```

If `compute_action` returns something uncastable to `int`, this raises a `TypeError` at runtime with no context about which controller caused it. Ensure selector controllers always return a valid integer index.

### G9 — Skill groups for controllers are stubbed

`SkillControllerProcessor.__init__` hard-codes:

```python
# TODO: Skill groups for controllers are valid. Need to implement
self.skill_group_skill_processors = []
```

The list is always empty. If a controller is placed in a skill group, the group pipeline (where one skill's output feeds into another's input) does not work. There is no error — it silently uses an empty processors list.

---

## Complete Working Example

A minimal but realistic controller that implements all required methods. This is a proportional setpoint controller suitable for use as a benchmark baseline.

```python
from amesa_core import SkillController
from typing import Dict, List
import numpy as np


class SetpointController(SkillController):
    """
    Proportional controller that drives position toward a target setpoint.
    Serves as a hand-coded baseline for benchmarking against RL skills.
    """

    def __init__(self, *args, **kwargs):
        # Initialize any per-episode state here.
        # Note: __init__ is called fresh on every episode reset (local controllers).
        self.setpoint = 0.0
        self.success_threshold = 0.05
        self.out_of_bounds_limit = 10.0
        self.kp = 0.8   # proportional gain

    async def compute_action(self, transformed_sensors: Dict, action) -> np.ndarray:
        """Core control logic. Returns action to send to sim."""
        # action may be None on the first step — handle gracefully
        position = transformed_sensors["position"][0]
        error = self.setpoint - position
        control_output = self.kp * error
        # Clip to valid action range
        control_output = float(np.clip(control_output, -1.0, 1.0))
        return np.array([control_output], dtype=np.float32)

    async def filtered_sensor_space(self) -> List[str]:
        """Declare which sensors this controller reads."""
        return ["position", "velocity"]

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        """
        Returns True if position is within threshold of setpoint.

        ⚠️ V1 NOTE: This method is never called by the V1 processor.
        It IS called in V2 (EventControllerProcessor).
        """
        position = transformed_sensors["position"][0]
        return abs(position - self.setpoint) < self.success_threshold

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        """
        Returns True if position has gone out of bounds.

        ⚠️ V1 NOTE: This method is never called by the V1 processor.
        It IS called in V2 (EventControllerProcessor).
        """
        position = transformed_sensors["position"][0]
        return abs(position) > self.out_of_bounds_limit
```

### Registration

```python
from amesa_core import Agent
from amesa_core.agent.skill import Skill

agent = Agent()
agent.add_sensors([...])

# Controller detected automatically by type inference — no special method needed
skill = Skill("setpoint-ctrl", SetpointController)
agent.add_skill(skill)
```

### Selector Controller Example

```python
from amesa_core import SkillController
from amesa_core.agent.skill.skill_selector import SkillSelector
from typing import Dict, List


class ModeSelector(SkillController):
    """Routes to one of two child skills based on sensor state."""

    async def compute_action(self, transformed_sensors: Dict, action) -> int:
        """Return integer index of the child skill to activate."""
        position = transformed_sensors["position"][0]
        # Child 0 = stabilize, Child 1 = accelerate
        if abs(position) < 1.0:
            return 0   # near setpoint — stabilize
        return 1       # far from setpoint — accelerate

    async def filtered_sensor_space(self) -> List[str]:
        return ["position"]

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        return False   # selector itself has no success criteria

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        return False   # selector itself has no termination


# Registration:
selector = SkillSelector(
    "mode-selector",
    ModeSelector,
    children=["stabilize", "accelerate"]
)
agent.add_selector_skill(selector, ...)
```

---

## API Quick Reference

| Method | Required | Async | Called in V1 | Called in V2 |
|---|---|---|---|---|
| `compute_action` | ✓ | ✓ | ✓ | ✓ |
| `filtered_sensor_space` | ✓ | ✓ | ✓ | ✓ |
| `compute_success_criteria` | ✓ | ✓ | ✗ (stub) | ✓ |
| `compute_termination` | ✓ | ✓ | ✗ (stub) | ✓ |
| `transform_sensors` | optional | ✓ | ✗ (never called) | ✗ (never called) |
| `is_controller` | utility | — | — | — |
| `add_scenario` | utility | — | — | — |

→ See also: [Teacher API](./teacher-api.md) for the RL skill interface, [Orchestration](./orchestration.md) for `SkillSelector` routing, [Sensor Definitions](./sensor-definitions.md) for sensor setup.
