# Spaces

Spaces define the shape and bounds of action outputs and sensor inputs for skills. They mirror `gymnasium` spaces but with AMESA extensions.

## Import

```python
from amesa_core.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
```

## Box

Continuous values in `[low, high]` of a given shape.

```python
Box(low=-1.0, high=1.0, shape=(1,))          # single continuous value
Box(low=0.0, high=100.0, shape=(3,))          # 3-element vector
Box(low=np.array([0, 0]), high=np.array([10, 10]), shape=(2,))  # per-element bounds
```

Set `self.action_space = Box(...)` in your `SkillTeacher.__init__`.

## Discrete

Integer action in `{0, 1, …, n-1}`.

```python
Discrete(4)   # 4 discrete actions: 0, 1, 2, 3
```

## Dict

A named collection of spaces, used when the action/sensor is a dictionary.

```python
Dict({
    "throttle": Box(low=0.0, high=1.0, shape=(1,)),
    "steering": Box(low=-1.0, high=1.0, shape=(1,)),
})
```

## Other spaces

| Space | Use |
|---|---|
| `Tuple(spaces)` | Ordered collection of heterogeneous spaces |
| `MultiBinary(n)` | `n` independent binary values |
| `MultiDiscrete([n1, n2, ...])` | Multiple discrete choices |

## Where spaces appear

- **`SkillTeacher.__init__`** — optionally set `self.action_space` for use within teacher methods (e.g. fallback sampling in `transform_action`). The trainer reads the action space from the sim, not the teacher.
- **`SkillController.__init__`** — same as above
- **`ServerAmesa.sensor_space_info()`** — return the obs space (gymnasium `Space` or amesa `Space` accepted)
- **`ServerAmesa.action_space_info()`** — **this is the authoritative source** the trainer uses for the action space

## Converting gym spaces

```python
from amesa_core.spaces import convert_to_amesa_space
amesa_space = convert_to_amesa_space(gym_space)
```

---

## ⚠️ Quirks

**`shape` is required for `Box`** — Always pass an explicit `shape` tuple. Omitting it can cause unexpected broadcasting.

**Action space in Teacher** — `self.action_space` must be set in `__init__`, not lazily. The trainer reads it at initialization time.

**`Box(low, high, shape)` vs gymnasium** — Positional argument order matches gymnasium: `low`, `high`, `shape`, `dtype`. dtype defaults to `float32`.
