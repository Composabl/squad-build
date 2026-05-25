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

| Space                          | Use                                        |
| ------------------------------ | ------------------------------------------ |
| `Tuple(spaces)`                | Ordered collection of heterogeneous spaces |
| `MultiBinary(n)`               | `n` independent binary values              |
| `MultiDiscrete([n1, n2, ...])` | Multiple discrete choices                  |

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

## Action masks

Each AMESA space implements `fit_mask(mask)`, which formats a raw mask into the structure required by the trainer. The skill processor calls this automatically on the value returned by `compute_action_mask`, so you generally do not call `fit_mask` directly — but you must return a mask in the correct shape for your action space type.

| Space                 | Expected mask input                                   | Output shape                                                      |
| --------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| `Discrete(n)`         | Any array-like of length `n`                          | `(n,)` int8 array                                                 |
| `Box(shape)`          | Array with last dim doubled                           | `(*shape[:-1], shape[-1] * 2)` — one mean + one std per dimension |
| `MultiDiscrete(nvec)` | Flat array, tuple, or list of per-subspace arrays     | Tuple of int8 arrays, one per subspace                            |
| `Tuple(spaces)`       | `tuple` or `list` of per-subspace masks               | Tuple of recursively fitted masks                                 |
| `Dict(spaces)`        | `dict` with one mask value per key (recursively fitted) | Dict of recursively fitted masks                                |

### Box mask shape

For a `Box(low, high, shape=(N,))`, the mask must have shape `(N * 2,)` — the trainer interprets alternating values as (mean-enabled, std-enabled) pairs per dimension. Returning a plain `list[bool]` of length `N` will raise a `ValueError`.

```python
# Box(shape=(3,)) → mask must be shape (6,): [mean0, std0, mean1, std1, mean2, std2]
async def compute_action_mask(self, transformed_sensors, action):
    return [1, 1, 1, 1, 0, 0]  # mask out last dimension
```

### Discrete mask

```python
# Discrete(4) → mask shape (4,): 1 = action allowed, 0 = masked
async def compute_action_mask(self, transformed_sensors, action):
    return [1, 1, 0, 0]  # only actions 0 and 1 are available
```

### Tuple / MultiDiscrete mask

Pass a `tuple` or `list` with one element per subspace:

```python
# Tuple([Discrete(3), Discrete(2)]) → ([mask_for_d3], [mask_for_d2])
async def compute_action_mask(self, transformed_sensors, action):
    return ([1, 0, 1], [1, 1])
```

---

## ⚠️ Quirks

**`shape` is required for `Box`** — Always pass an explicit `shape` tuple. Omitting it can cause unexpected broadcasting.

**Action space in Teacher** — `self.action_space` must be set in `__init__`, not lazily. The trainer reads it at initialization time.

**`Box(low, high, shape)` vs gymnasium** — Positional argument order matches gymnasium: `low`, `high`, `shape`, `dtype`. dtype defaults to `float32`.

**`Text` spaces are excluded from observations and training** — Sensors backed by a `Text` space are silently dropped from filtered observation spaces and stripped from the policy observation space before training. Listing a Text sensor in `filtered_sensor_space()` will not cause an error, but the key will never appear in `transformed_sensors` and cannot be used in reward or termination logic.
