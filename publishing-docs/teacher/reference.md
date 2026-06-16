# SkillTeacher Interface

`SkillTeacher` is the ML-training interface for a `Skill`.

## Full scaffold

```python
import numpy as np
from amesa_core.agent.skill.skill_teacher import SkillTeacher
from typing import Dict, List

class MyTeacher(SkillTeacher):
    async def compute_reward(self, transformed_sensors: Dict, action, sim_reward: float) -> float:
        """Shape the learning signal used by the trainer.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Policy output shaped to the action space.
        :param sim_reward: Simulator reward for the current step.
        :type sim_reward: float
        :returns: Scalar reward for policy optimization.
        :rtype: float
        :raises NotImplementedError: if not overridden in the subclass.
        """
        error = abs(float(transformed_sensors["error"][0]))
        return float(-error)

    async def compute_success_criteria(self, transformed_sensors: Dict, action) -> bool:
        """Define when the objective has been achieved.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Policy output shaped to the action space.
        :returns: ``True`` when the episode should count as success.
        :rtype: bool
        :raises NotImplementedError: if not overridden in the subclass.
        """
        return abs(float(transformed_sensors["error"][0])) < 0.1

    async def transform_action(self, transformed_sensors: Dict, action):
        """Convert policy output into simulator-consumable action format.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Raw policy output.
        :returns: Action in the format expected by the simulator.
        :raises NotImplementedError: if not overridden in the subclass.
        """
        return action

    async def filtered_sensor_space(self) -> List[str]:
        """Declare which sensor keys this teacher reads.

        :returns: List of sensor key strings.
        :rtype: List[str]
        :raises NotImplementedError: if not overridden in the subclass.
        """
        return ["value", "target", "error"]

    async def transform_sensors(self, sensors, action) -> Dict:
        """Build derived sensor features before teacher logic runs.

        Wraps each sensor listed in :meth:`filtered_sensor_space` in a
        ``np.ndarray`` so the framework can call ``.flatten()`` when building
        the observation vector. Also derives ``error`` so reward and success
        logic can read a single key.

        :param sensors: Raw sensor dict from the environment.
        :param action: Current action (may influence feature construction).
        :returns: Transformed sensor dict.
        :rtype: Dict
        :raises AttributeError: if a key listed in :meth:`filtered_sensor_space`
            is returned as a plain ``float`` or ``int`` rather than a
            ``np.ndarray``; the framework calls ``.flatten()`` on those values
            when building the observation vector
            (``"'float' object has no attribute 'flatten'"``).
        """
        value = float(sensors["value"])
        target = float(sensors["target"])
        error = value - target
        return {
            **sensors,
            "value": np.array([value], dtype=np.float32),
            "target": np.array([target], dtype=np.float32),
            "error": np.array([error], dtype=np.float32),
        }

    async def compute_termination(self, transformed_sensors: Dict, action) -> bool:
        """End episodes on failure, safety, or timeout conditions.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Policy output shaped to the action space.
        :returns: ``True`` when the episode should terminate early.
        :rtype: bool
        """
        return float(transformed_sensors["value"][0]) > 100.0

    async def compute_action_mask(self, transformed_sensors: Dict, action) -> List[bool]:
        """Restrict valid actions for the current step.

        Disables the ``increase`` action (index 1) when the value is already
        within the success band, preventing unnecessary exploration.

        :param transformed_sensors: Post-:meth:`transform_sensors` sensor dict.
        :type transformed_sensors: Dict
        :param action: Current action.
        :returns: Boolean mask of length ``n_actions`` (``Discrete`` example).
        :rtype: List[bool]
        :raises ValueError: if the returned mask length differs from the action
            space size (e.g., ``n`` for ``Discrete(n)``, ``N*2`` for
            ``Box(shape=(N,))``).
        """
        error = float(transformed_sensors["error"][0])
        at_target = abs(error) < 0.1
        # Discrete(2): [decrease, increase]
        return [True, not at_target]
```

## Methods and intended use

### `compute_reward(self, transformed_sensors, action, sim_reward) -> float` (required)

Computes the scalar reward used for policy optimization. Use it for reward shaping, penalties, and blending simulator reward with task-specific objectives.

### `compute_success_criteria(self, transformed_sensors, action) -> bool` (required)

Declares when the current episode should count as success. Use it for thresholds, compound checks, or milestone-based completion.

### `transform_action(self, transformed_sensors, action)` (required)

Converts policy output into the exact action format expected by the simulator. Use it for scaling, clipping, remapping, and structural conversion.

### `filtered_sensor_space(self) -> list[str]` (required)

Defines the teacher's sensor dependencies. Use it to keep observation requirements explicit and minimal.

### `transform_sensors(self, sensors, action) -> dict` (optional)

Preprocesses raw sensors into transformed features shared by reward/success/termination. Use it for normalization, coordinate transforms, and feature engineering.

> **Important:** Values returned for keys listed in `filtered_sensor_space` must be array-like (e.g., `np.ndarray`), not plain Python `float` or `int`. The framework calls `.flatten()` on these values when building the observation vector — returning a bare scalar will raise `'float' object has no attribute 'flatten'`. If you need to remap or wrap a sensor value (e.g., angle wrapping), return `np.array([new_value], dtype=np.float32)` rather than a raw float. Derived keys that are *not* in `filtered_sensor_space` (e.g., normalized scalars used only in reward shaping) may be any Python type.

### `compute_termination(self, transformed_sensors, action) -> bool` (optional)

Signals episode termination for non-success reasons such as failure, timeout, or safety boundaries.

### `compute_action_mask(self, transformed_sensors, action) -> list[bool]` (optional)

Builds dynamic action constraints for the current state. Use it to disable invalid actions and enforce runtime feasibility.

### `get_custom_action_space(self)` (optional)

Returns a custom action space (e.g., `Box`, `Discrete`) for the policy to train against, instead of the simulator's native action space. Returns `None` by default (uses sim's action space). When overridden, `transform_action` is responsible for mapping the policy output back to the sim's expected format.

### `is_compute_done(self, transformed_sensors, action) -> bool` (optional)

Returns `True` when `compute_success_criteria` **and** `compute_termination` are both `True`. The default combines both checks into a single done signal for the training loop. Override only when you need custom gate logic beyond the conjunction of the two.

### `add_scenario(self, scenario)` (optional)

Associates a `Scenario` with the teacher, storing it on `self.scenario`. Called by the framework before training begins. Override to react to scenario changes at setup time; most teachers can use the default.

## Method contracts

- `transformed_sensors`: post-`transform_sensors` sensor dict.
- `action`: policy output shaped to action space.
- `sim_reward`: simulator reward for the current step.
- `compute_reward` **must** return a Python `float`.
- `filtered_sensor_space()` returns the exact sensor keys this teacher consumes.

## Action mask shape reminders

- `Discrete(n)`: length `n`
- `Box(shape=(N,))`: length `N*2` (mean/std pairs)
- `Tuple`: tuple/list per sub-space
- `Dict`: dict keyed by action-space keys
