## Scenarios

Scenarios carve the simulation space into named situations defined by specific variable configurations. They enable targeted training — each skill practices only the scenarios relevant to it — and help orchestrators learn which skill to activate under which conditions.

### Defining Scenarios by Variable Type

| Variable Type          | Definition Method    | Example                                    |
| ---------------------- | -------------------- | ------------------------------------------ |
| **Discrete**           | Named categories     | `windy`, `far_from_charger`, `low_battery` |
| **Continuous (exact)** | Single numeric value | `windspeed = 20 knots`                     |
| **Continuous (range)** | A range of values    | `windspeed between 20–40 knots`            |

Discrete variables often come from a perceptor (e.g., an ML classifier) that transforms raw sensors into categorical outputs.

#### SDK note: Scenario dicts (for coding agents)

When building agents in Python, you can pass either a `Scenario` object or a plain `dict` into `skill.add_scenario(...)`. Dicts are converted via `Scenario.from_json(...)`. The dict keys become scenario variables, and values can be:

- `{"data": <value>, "type": "is_equal"}` for constants
- `{"data": [low, high], "type": "is_between"}` for ranges
- `{"data": [v1, v2, ...], "type": "is_element_of"}` for discrete sets
- Short-hand lists (e.g., `[low, high]`) for ranges or discrete sets

Keep scenario dictionaries JSON-serializable so they can flow through trainer configs and be stored in historian outputs.

### Example: Restaurant Operations

Three scenarios based on recipe demand levels:

| Scenario      | Recipe A | Recipe B | Recipe C |
| ------------- | -------- | -------- | -------- |
| Low demand    | 30       | 20       | 10       |
| Normal demand | 60       | 45       | 10       |
| High demand   | 100      | 50       | 25       |

The agent trains on low demand first until success criteria are met, then normal, then high. Knowledge accumulates, so later scenarios are learned faster.

### Scenario Flows

Scenario flows define ordered sequences of scenarios for training. Without flows, AMESA connects scenarios at random. Flows are important when the order matters (e.g., practicing flying in high winds _then_ landing in the same conditions).

### Assigning Scenarios to Skill Agents

Scenarios are assigned to individual skill agents during configuration. Not every skill needs every scenario — a landing skill doesn't practice takeoff scenarios. In the skill agent configuration modal, check the boxes next to each relevant scenario for each section.
